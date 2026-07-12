#!/usr/bin/env python3
"""
process_databento.py — Ingest Databento MBP-1 tick data and aggregate into OHLCV bars.

Reads zstandard-compressed CSV files containing CME Micro EUR/USD Futures (M6E)
tick data, filters for trades on the front-month outright contract, and aggregates
into Daily (D) and 4-Hour (H4) bars aligned to Forex trading sessions (5 PM NY).

Bars are upserted into the PostgreSQL `bars` table via direct psycopg2 for
batch performance.

Usage:
    python process_databento.py
    python process_databento.py --data-dir "C:\\path\\to\\data"
"""

import sys
import os

# Ensure project root is on sys.path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import glob
import io
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import zstandard as zstd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = r"C:\Users\angel\OneDrive\Apps\Quant EOD\GLBX-20260405-TX6GF64XBR"
FILE_GLOB = "glbx-mdp3-*.mbp-1.csv.zst"

INSTRUMENT = "EUR_USD"
NY_TZ = ZoneInfo("America/New_York")

# Databento fixed-point price divisor for CME MBP-1 data.
# CME prices are stored as integers with a 1e-9 scale factor.
PRICE_SCALE = 1e-9

# H4 bar boundary hours (NY time): 17, 21, 01, 05, 09, 13
H4_BOUNDARY_HOURS = [17, 21, 1, 5, 9, 13]

BATCH_COMMIT_SIZE = 500

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("process_databento")

# ---------------------------------------------------------------------------
# Timezone-aware bar alignment helpers
# ---------------------------------------------------------------------------


def _ny_datetime(dt_utc: datetime) -> datetime:
    """Convert a UTC datetime to America/New_York."""
    return dt_utc.astimezone(NY_TZ)


def _forex_trading_day(dt_utc: datetime) -> datetime:
    """
    Determine the Forex trading day for a given UTC timestamp.

    A Forex trading day runs from 5:00 PM NY of the *previous* calendar day
    to 5:00 PM NY of the *current* calendar day.

    Returns a timezone-aware datetime at 5:00 PM NY representing the trading
    day boundary (the *end* of the trading day / bar_time).
    """
    ny_dt = _ny_datetime(dt_utc)
    # If it's before 5 PM NY, this trade belongs to "today's" trading day
    # (which started at yesterday 5 PM).  The bar_time label is today's 5 PM.
    # If it's at or after 5 PM NY, the trade belongs to "tomorrow's" trading day.
    cutoff = ny_dt.replace(hour=17, minute=0, second=0, microsecond=0)
    if ny_dt >= cutoff:
        # After 5 PM → belongs to next trading day
        bar_date = (ny_dt + timedelta(days=1)).date()
    else:
        bar_date = ny_dt.date()

    # Return 5 PM NY on the bar_date
    return datetime(bar_date.year, bar_date.month, bar_date.day, 17, 0, 0,
                    tzinfo=NY_TZ)


def _h4_bar_start(dt_utc: datetime) -> datetime:
    """
    Determine the H4 bar start time for a given UTC timestamp.

    H4 boundaries (NY time): 17:00, 21:00, 01:00, 05:00, 09:00, 13:00.
    A trade at exactly a boundary belongs to the bar starting at that boundary.

    Returns a timezone-aware datetime (NY) at the bar's start boundary.
    """
    ny_dt = _ny_datetime(dt_utc)
    ny_hour = ny_dt.hour
    ny_minute = ny_dt.minute
    ny_second = ny_dt.second
    ny_micro = ny_dt.microsecond

    # Find which boundary the timestamp falls into.
    # Boundaries sorted for a 24h cycle starting at 17:00:
    # 17, 21, 1, 5, 9, 13 — then wraps to 17 next day.
    # For comparison, map everything to "hours since 17:00" (mod 24).
    hours_since_17 = (ny_hour - 17) % 24

    # H4 boundaries in "hours since 17:00": 0, 4, 8, 12, 16, 20
    # The bar start is the largest boundary <= hours_since_17.
    boundary_offsets = [0, 4, 8, 12, 16, 20]
    bar_offset = 0
    for offset in boundary_offsets:
        if hours_since_17 >= offset:
            bar_offset = offset
        else:
            break

    bar_hour = (17 + bar_offset) % 24

    # Determine the calendar date for the bar start
    bar_date = ny_dt.date()

    # If bar_hour > ny_hour (wraps through midnight), bar started yesterday
    # This happens when ny_hour < 17 and bar_offset puts us back to yesterday.
    # E.g., ny_hour=2, bar_hour=1 → same day.
    #        ny_hour=0, bar_hour=21 → previous day.
    if bar_hour > ny_hour:
        bar_date -= timedelta(days=1)

    return datetime(bar_date.year, bar_date.month, bar_date.day,
                    bar_hour, 0, 0, tzinfo=NY_TZ)


# ---------------------------------------------------------------------------
# CSV / zstd file reading
# ---------------------------------------------------------------------------


def iter_zst_csv_rows(filepath: str):
    """
    Yield rows (as dicts) from a zstandard-compressed CSV file.

    Uses streaming decompression to keep memory usage low.
    """
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            csv_reader = csv.DictReader(text_stream)
            for row in csv_reader:
                yield row


# ---------------------------------------------------------------------------
# Trade parsing and filtering
# ---------------------------------------------------------------------------


def _parse_timestamp(ts_str: str) -> datetime:
    """
    Parse a Databento ISO 8601 timestamp string into a timezone-aware datetime.

    Example input: '2022-03-06T20:00:24.116267233Z'
    Python's fromisoformat handles up to 6 decimal places; Databento uses 9.
    We truncate to microsecond precision.
    """
    # Strip trailing 'Z' and truncate nanoseconds to microseconds
    ts_clean = ts_str.rstrip("Z")
    # Split on '.' to handle fractional seconds
    if "." in ts_clean:
        date_part, frac = ts_clean.split(".", 1)
        # Truncate to 6 digits (microseconds)
        frac = frac[:6].ljust(6, "0")
        ts_clean = f"{date_part}.{frac}"
    return datetime.fromisoformat(ts_clean).replace(tzinfo=timezone.utc)


def _is_spread_symbol(symbol: str) -> bool:
    """Return True if the symbol is a spread (contains '-')."""
    return "-" in symbol


def _parse_price(price_str: str) -> float:
    """
    Parse a Databento MBP-1 price field.

    When pretty_px=true (as in our dataset), prices are already human-readable
    floats like '1.096800000'.  When pretty_px=false they would be raw integers
    requiring a 1e-9 scale — but we handle both cases defensively.
    """
    # If the string contains a decimal point it's already a pretty price
    if "." in price_str:
        return float(price_str)
    raw = int(price_str)
    return raw * PRICE_SCALE


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------


def _collect_trades_from_file(filepath: str) -> list[dict]:
    """
    Read a single .zst file and return a list of parsed trade dicts.

    Filters for action='T' (trades) and excludes spread symbols.
    """
    trades = []
    row_count = 0
    skipped_spread = 0
    skipped_non_trade = 0

    for row in iter_zst_csv_rows(filepath):
        row_count += 1

        # Filter: trades only
        if row.get("action") != "T":
            skipped_non_trade += 1
            continue

        symbol = row.get("symbol", "")

        # Filter: outright contracts only (no spreads)
        if _is_spread_symbol(symbol):
            skipped_spread += 1
            continue

        ts_event = _parse_timestamp(row["ts_event"])
        price = _parse_price(row["price"])
        size = int(row["size"])

        trades.append({
            "ts_event": ts_event,
            "symbol": symbol,
            "price": price,
            "size": size,
        })

    logger.debug(
        f"  Rows: {row_count:,} | Trades: {len(trades):,} | "
        f"Skipped non-trade: {skipped_non_trade:,} | Skipped spreads: {skipped_spread:,}"
    )
    return trades


def _determine_front_month(trades: list[dict]) -> dict[str, str]:
    """
    For each calendar date, determine the front-month contract (symbol with
    the highest trade count).

    Returns {date_str: symbol}.
    """
    # Group trade counts by (calendar_date, symbol)
    date_symbol_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for t in trades:
        cal_date = _ny_datetime(t["ts_event"]).date().isoformat()
        date_symbol_counts[cal_date][t["symbol"]] += 1

    front_month = {}
    for cal_date, symbol_counts in date_symbol_counts.items():
        best_symbol = max(symbol_counts, key=symbol_counts.get)
        front_month[cal_date] = best_symbol
        logger.debug(f"  Front month {cal_date}: {best_symbol} ({symbol_counts[best_symbol]:,} trades)")

    return front_month


def _filter_front_month_trades(trades: list[dict], front_month: dict[str, str]) -> list[dict]:
    """Keep only trades for the front-month contract on each calendar date."""
    filtered = []
    for t in trades:
        cal_date = _ny_datetime(t["ts_event"]).date().isoformat()
        if t["symbol"] == front_month.get(cal_date):
            filtered.append(t)
    return filtered


def _aggregate_daily_bars(trades: list[dict]) -> list[dict]:
    """
    Aggregate trades into daily OHLCV bars aligned to 5 PM NY.

    Returns list of bar dicts ready for database insertion.
    """
    # Group trades by trading day
    day_buckets: dict[datetime, list[dict]] = defaultdict(list)
    for t in trades:
        bar_time = _forex_trading_day(t["ts_event"])
        day_buckets[bar_time].append(t)

    bars = []
    for bar_time in sorted(day_buckets):
        bucket = day_buckets[bar_time]
        # Sort by timestamp to get correct open/close
        bucket.sort(key=lambda x: x["ts_event"])
        prices = [t["price"] for t in bucket]
        bars.append({
            "instrument": INSTRUMENT,
            "granularity": "D",
            "bar_time": bar_time.isoformat(),
            "open": bucket[0]["price"],
            "high": max(prices),
            "low": min(prices),
            "close": bucket[-1]["price"],
            "volume": sum(t["size"] for t in bucket),
            "complete": True,
        })

    return bars


def _aggregate_h4_bars(trades: list[dict]) -> list[dict]:
    """
    Aggregate trades into 4-hour OHLCV bars.

    H4 boundaries (NY time): 17:00, 21:00, 01:00, 05:00, 09:00, 13:00.

    Returns list of bar dicts ready for database insertion.
    """
    h4_buckets: dict[datetime, list[dict]] = defaultdict(list)
    for t in trades:
        bar_start = _h4_bar_start(t["ts_event"])
        h4_buckets[bar_start].append(t)

    bars = []
    for bar_start in sorted(h4_buckets):
        bucket = h4_buckets[bar_start]
        bucket.sort(key=lambda x: x["ts_event"])
        prices = [t["price"] for t in bucket]
        bars.append({
            "instrument": INSTRUMENT,
            "granularity": "H4",
            "bar_time": bar_start.isoformat(),
            "open": bucket[0]["price"],
            "high": max(prices),
            "low": min(prices),
            "close": bucket[-1]["price"],
            "volume": sum(t["size"] for t in bucket),
            "complete": True,
        })

    return bars


# ---------------------------------------------------------------------------
# Database upsert
# ---------------------------------------------------------------------------

UPSERT_SQL = """
    INSERT INTO bars (instrument, granularity, bar_time, open, high, low, close, volume, complete)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (instrument, granularity, bar_time)
    DO UPDATE SET
        open     = EXCLUDED.open,
        high     = EXCLUDED.high,
        low      = EXCLUDED.low,
        close    = EXCLUDED.close,
        volume   = EXCLUDED.volume,
        complete = EXCLUDED.complete,
        fetched_at = NOW()
"""


def _upsert_bars(conn, bars: list[dict]) -> int:
    """
    Batch-upsert bars into the database.

    Commits every BATCH_COMMIT_SIZE rows for performance.
    Returns the number of rows upserted.
    """
    if not bars:
        return 0

    total = 0
    try:
        with conn.cursor() as cur:
            for i, bar in enumerate(bars, 1):
                cur.execute(UPSERT_SQL, (
                    bar["instrument"],
                    bar["granularity"],
                    bar["bar_time"],
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    bar["volume"],
                    bar["complete"],
                ))
                if i % BATCH_COMMIT_SIZE == 0:
                    conn.commit()
                    logger.debug(f"  Committed batch at row {i}")

            # Final commit for remaining rows
            conn.commit()
            total = len(bars)

    except Exception as e:
        conn.rollback()
        logger.error(f"Error upserting bars: {e}")
        raise

    return total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Databento MBP-1 tick data into OHLCV bars."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing .zst files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Discover files
    pattern = os.path.join(data_dir, FILE_GLOB)
    zst_files = sorted(glob.glob(pattern))
    if not zst_files:
        logger.error(f"No files matching '{FILE_GLOB}' found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(zst_files)} .zst files in {data_dir}")

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="quant_eod",
            user="postgres",
            password="postgres",
        )
        logger.info("Connected to PostgreSQL (quant_eod)")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    # --------------- Processing loop ---------------
    total_trades = 0
    total_daily_bars = 0
    total_h4_bars = 0

    try:
        for file_idx, filepath in enumerate(zst_files, 1):
            filename = os.path.basename(filepath)
            
            # Check if file has already been ingested to make script resumable
            try:
                date_part = filename.split("glbx-mdp3-")[1].split(".")[0]
                file_date = datetime.strptime(date_part, "%Y%m%d").date()
            except Exception:
                file_date = None
                
            if file_date:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS(SELECT 1 FROM bars WHERE instrument = %s AND granularity = 'D' AND complete = TRUE AND date(bar_time) = %s)",
                        (INSTRUMENT, file_date)
                    )
                    exists = cur.fetchone()[0]
                    if exists:
                        logger.info(f"[{file_idx}/{len(zst_files)}] Skipping {filename} — already ingested")
                        continue

            logger.info(f"[{file_idx}/{len(zst_files)}] Processing {filename} ...")

            # Step 1: Read and filter trades
            trades = _collect_trades_from_file(filepath)
            if not trades:
                logger.warning(f"  No trades found in {filename}, skipping.")
                continue

            logger.info(f"  Extracted {len(trades):,} outright trades")

            # Step 2: Determine front-month contract per calendar date
            front_month = _determine_front_month(trades)
            for cal_date, symbol in sorted(front_month.items()):
                logger.info(f"  Front-month {cal_date}: {symbol}")

            # Step 3: Filter to front-month trades only
            fm_trades = _filter_front_month_trades(trades, front_month)
            logger.info(f"  Front-month trades: {len(fm_trades):,} (of {len(trades):,})")
            total_trades += len(fm_trades)

            # Step 4: Aggregate into Daily bars
            daily_bars = _aggregate_daily_bars(fm_trades)
            logger.info(f"  Daily bars: {len(daily_bars)}")

            # Step 5: Aggregate into H4 bars
            h4_bars = _aggregate_h4_bars(fm_trades)
            logger.info(f"  H4 bars: {len(h4_bars)}")

            # Step 6: Upsert into database
            n_daily = _upsert_bars(conn, daily_bars)
            n_h4 = _upsert_bars(conn, h4_bars)
            total_daily_bars += n_daily
            total_h4_bars += n_h4

            logger.info(
                f"  Upserted {n_daily} daily + {n_h4} H4 bars "
                f"({total_daily_bars + total_h4_bars} cumulative)"
            )

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        conn.close()
        logger.info("Database connection closed.")

    # --------------- Summary ---------------
    logger.info("=" * 60)
    logger.info("Processing complete.")
    logger.info(f"  Files processed:  {len(zst_files)}")
    logger.info(f"  Total trades:     {total_trades:,}")
    logger.info(f"  Daily bars:       {total_daily_bars}")
    logger.info(f"  H4 bars:          {total_h4_bars}")
    logger.info(f"  Total bars:       {total_daily_bars + total_h4_bars}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
