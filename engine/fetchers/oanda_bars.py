"""
Fetcher: OANDA V20 — Daily and 4H candle bars.

Pulls completed bars for all configured instruments.
OANDA's dailyAlignment defaults to 17 (5 PM) in America/New_York,
which aligns perfectly with the Forex daily rollover.
"""
import requests
import logging
from datetime import datetime, timezone
from config.settings import OANDA_API_TOKEN, OANDA_BASE_URL, INSTRUMENTS
from models.database import get_connection

logger = logging.getLogger(__name__)


def fetch_candles(instrument: str, granularity: str, count: int) -> list[dict]:
    """
    Fetch candle data from OANDA V20 API.

    Args:
        instrument: e.g. "EUR_USD"
        granularity: "D" for daily, "H4" for 4-hour
        count: number of bars to fetch (max 5000)

    Returns:
        List of candle dicts with OHLCV data.
    """
    url = f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles"
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M",  # mid prices
        # dailyAlignment defaults to 17 (5 PM EST) — no override needed
    }
    
    headers = {
        "Authorization": f"Bearer {OANDA_API_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    candles = []
    for c in data.get("candles", []):
        if not c.get("complete", False):
            continue  # skip incomplete (current) bar
        mid = c["mid"]
        candles.append({
            "instrument": instrument,
            "granularity": granularity,
            "bar_time": c["time"],
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": int(c.get("volume", 0)),
            "complete": True,
        })

    logger.info(f"Fetched {len(candles)} {granularity} bars for {instrument}")
    return candles


def store_candles(candles: list[dict]):
    """Upsert candles into the bars table."""
    if not candles:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for c in candles:
                cur.execute("""
                    INSERT INTO bars (instrument, granularity, bar_time, open, high, low, close, volume, complete)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (instrument, granularity, bar_time)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        complete = EXCLUDED.complete,
                        fetched_at = NOW()
                """, (
                    c["instrument"], c["granularity"], c["bar_time"],
                    c["open"], c["high"], c["low"], c["close"],
                    c["volume"], c["complete"],
                ))
        conn.commit()
        logger.info(f"Stored {len(candles)} bars")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing candles: {e}")
        raise
    finally:
        conn.close()


def fetch_and_store_all():
    """Fetch daily + 4H bars for all instruments and store them."""
    results = {}
    for instrument in INSTRUMENTS:
        try:
            # Daily bars — 210 days for indicator calculation (MA-200 needs 200+)
            daily = fetch_candles(instrument, "D", 210)
            store_candles(daily)
            results[f"{instrument}_D"] = len(daily)

            # 4H bars — 120 bars = 20 days of 4H data
            h4 = fetch_candles(instrument, "H4", 120)
            store_candles(h4)
            results[f"{instrument}_H4"] = len(h4)

        except Exception as e:
            logger.error(f"Failed to fetch bars for {instrument}: {e}")
            results[f"{instrument}_error"] = str(e)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_and_store_all()
    print(result)
