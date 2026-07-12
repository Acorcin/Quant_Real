"""
Fetcher: Databento Tick Stream Simulator.

Reads historical Databento MBP-1 tick data files (.zst) chronologically
and replays them into the Redis stream `ticks:{instrument}`.
This simulates live data transmission for testing the live trading/execution stack.
"""
import os
import sys
import csv
import io
import time
import logging
import argparse
import zstandard as zstd
import redis
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("databento_simulator")

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", os.environ.get("DB_HOST", "localhost"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

PRICE_SCALE = 1e-9
NY_TZ = ZoneInfo("America/New_York")


def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def parse_timestamp(ts_str: str) -> datetime:
    """Parse Databento timestamp with 9 decimal places into UTC datetime."""
    ts_clean = ts_str.rstrip("Z")
    if "." in ts_clean:
        date_part, frac = ts_clean.split(".", 1)
        frac = frac[:6].ljust(6, "0")
        ts_clean = f"{date_part}.{frac}"
    return datetime.fromisoformat(ts_clean).replace(tzinfo=timezone.utc)


def parse_price(price_str: str) -> float:
    """Parse price string into float (pretty price or raw scaled)."""
    if "." in price_str:
        return float(price_str)
    return int(price_str) * PRICE_SCALE


def get_tick_files(data_dir: str, start_date: date, end_date: date) -> list[tuple[date, str]]:
    """Scan and return sorted list of (date, file_path) in the date range."""
    files = []
    for f in os.listdir(data_dir):
        if not (f.startswith("glbx-mdp3-") and f.endswith(".mbp-1.csv.zst")):
            continue
        try:
            date_str = f.split("-")[2].split(".")[0]
            file_date = datetime.strptime(date_str, "%Y%m%d").date()
            if start_date <= file_date <= end_date:
                files.append((file_date, os.path.join(data_dir, f)))
        except Exception as e:
            logger.warning(f"Skipping malformed filename {f}: {e}")
    files.sort(key=lambda x: x[0])
    return files


def simulate_stream(
    data_dir: str,
    instrument: str,
    start_date: date,
    end_date: date,
    speed_multiplier: float,
):
    """
    Read Databento ticks chronologically and write them to Redis ticks stream.
    """
    import json
    redis_client = get_redis_client()
    redis_key = f"ticks:{instrument}"
    logger.info(f"Target Redis stream: {redis_key}")

    # Load walkforward results if available
    results_by_date = {}
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "walkforward_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                wf_data = json.load(f)
                for entry in wf_data.get("daily_results", []):
                    results_by_date[entry["date"]] = entry
            logger.info(f"Successfully loaded {len(results_by_date)} walkforward model predictions from {results_path}")
        except Exception as e:
            logger.error(f"Failed to load walkforward results: {e}")
    else:
        logger.warning(f"walkforward_results.json not found at {results_path}. Simulator will run with defaults.")

    tick_files = get_tick_files(data_dir, start_date, end_date)
    if not tick_files:
        logger.error(f"No Databento tick files found in {data_dir} for range {start_date} to {end_date}")
        return

    logger.info(f"Found {len(tick_files)} tick files to simulate.")

    dctx = zstd.ZstdDecompressor()
    last_tick_time = None

    for file_date, filepath in tick_files:
        file_date_str = file_date.strftime("%Y-%m-%d")
        logger.info(f"Replaying ticks from {os.path.basename(filepath)} (date={file_date_str})...")
        
        # Sync Redis daily models for this date
        if file_date_str in results_by_date:
            entry = results_by_date[file_date_str]
            regime_label = entry.get("regime", "high_vol_choppy")
            regime_state = 0 if regime_label == "low_vol" else 2 if regime_label == "high_vol_trending" else 1
            
            regime_payload = {
                "state_id": regime_state,
                "state_label": regime_label,
                "confidence": 0.95,
                "days_in_regime": 1,
                "model_version": "walkforward_sim"
            }
            metamodel_payload = {
                "direction": entry.get("model_direction", "flat"),
                "probability": entry.get("probability", 0.50),
                "size_multiplier": entry.get("size_multiplier", 0.0),
                "model_version": "xgb_walkforward"
            }
            
            try:
                redis_client.set(f"{instrument}:regime", json.dumps(regime_payload))
                redis_client.set(f"{instrument}:metamodel", json.dumps(metamodel_payload))
                # Set dynamic daily ATR from the backtest results or fallback
                redis_client.set(f"{instrument}:daily_atr", 0.0075)
                logger.info(f"[SIMULATOR] Synchronized Redis for {file_date_str}: Regime={regime_label}, Metamodel={metamodel_payload['direction']} (prob={metamodel_payload['probability']:.4f}, size={metamodel_payload['size_multiplier']}x)")
            except Exception as e:
                logger.error(f"[SIMULATOR] Failed to sync Redis daily models: {e}")
        
        # Load all trades from the day
        trades = []
        symbol_counts = {}

        try:
            with open(filepath, "rb") as fh:
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    csv_reader = csv.DictReader(text_stream)
                    for row in csv_reader:
                        if row.get("action") != "T":
                            continue
                        symbol = row.get("symbol", "")
                        if "-" in symbol:
                            continue
                        
                        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                        trades.append(row)
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            continue

        if not trades:
            logger.warning(f"No trades found in {filepath}, moving to next day.")
            continue

        # Front-month selection
        front_month = max(symbol_counts, key=symbol_counts.get)
        logger.info(f"Selected front-month contract: {front_month} ({symbol_counts[front_month]} ticks)")

        # Filter and sort
        fm_ticks = []
        for row in trades:
            if row["symbol"] != front_month:
                continue
            
            try:
                ts_event = parse_timestamp(row["ts_event"])
                price = parse_price(row["price"])
                
                bid_px = float(row.get("bid_px_00") or 0.0) if row.get("bid_px_00") else None
                ask_px = float(row.get("ask_px_00") or 0.0) if row.get("ask_px_00") else None
                
                # Apply fallback spread if quotes are missing
                if not bid_px or not ask_px or ask_px <= bid_px:
                    spread = 0.00015  # 1.5 pips
                    bid_px = price - spread / 2.0
                    ask_px = price + spread / 2.0
                
                fm_ticks.append({
                    "ts_event": ts_event,
                    "bid": round(bid_px, 5),
                    "ask": round(ask_px, 5),
                    "time_str": row["ts_event"]
                })
            except Exception as e:
                logger.debug(f"Skipping bad row: {e}")

        # Sort chronologically
        fm_ticks.sort(key=lambda x: x["ts_event"])

        logger.info(f"Loaded {len(fm_ticks)} chronological ticks for replay.")

        for i, tick in enumerate(fm_ticks):
            ts = tick["ts_event"]
            bid = tick["bid"]
            ask = tick["ask"]
            time_str = tick["time_str"]

            # Compute and apply time delay to match real-time (adjusted by multiplier)
            if last_tick_time is not None and speed_multiplier > 0:
                delta = (ts - last_tick_time).total_seconds()
                if delta > 0:
                    sleep_time = delta / speed_multiplier
                    # Cap sleep time at 5.0 seconds to prevent simulated weekend/holiday halts
                    sleep_time = min(sleep_time, 5.0)
                    time.sleep(sleep_time)

            last_tick_time = ts

            # Construct Redis payload matching live OANDA structure
            tick_payload = {
                "bid": bid,
                "ask": ask,
                "time": time_str,
                "received_at": datetime.now(timezone.utc).isoformat()
            }

            try:
                # Add to Redis raw ticks stream
                redis_client.xadd(redis_key, tick_payload, maxlen=50000)
                # Update streamer liveness
                redis_client.set(f"liveness:{instrument}:streamer", time_str)
                
                if (i + 1) % 500 == 0 or i == 0 or i == len(fm_ticks) - 1:
                    logger.info(f"[SIMULATED STREAM] Tick {i+1}/{len(fm_ticks)}: Bid={bid:.5f} Ask={ask:.5f} Time={time_str}")
            except Exception as e:
                logger.error(f"Error publishing tick to Redis: {e}")
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description="Databento Tick Stream Simulator for replaying trades to Redis."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\Users\angel\OneDrive\Apps\Quant EOD\GLBX-20260405-TX6GF64XBR",
        help="Directory containing Databento .zst files",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="EUR_USD",
        help="Instrument name (default: EUR_USD)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-04-05",
        help="Start date in YYYY-MM-DD (default: 2025-04-05)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-04-03",
        help="End date in YYYY-MM-DD (default: 2026-04-03)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=10.0,
        help="Replay speed multiplier. e.g. 1.0=realtime, 10.0=10x speed, 0.0=max speed (default: 10.0)",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info("==================================================")
    logger.info("DATABENTO TICK STREAM SIMULATOR STARTING")
    logger.info(f"Start Date       : {start_date}")
    logger.info(f"End Date         : {end_date}")
    logger.info(f"Speed Multiplier : {args.speed}x")
    logger.info("==================================================")

    simulate_stream(
        data_dir=args.data_dir,
        instrument=args.instrument,
        start_date=start_date,
        end_date=end_date,
        speed_multiplier=args.speed,
    )


if __name__ == "__main__":
    main()
