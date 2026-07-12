#!/usr/bin/env python3
"""
Historical Backfill Script.

Fetches 2 years of daily OANDA candle data and stores it in the
database. This bootstraps the HMM regime detector and provides
enough history for the meta-model to train on.

Usage:
    python backfill.py [--days 504]
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import OANDA_API_TOKEN, OANDA_BASE_URL, INSTRUMENTS
from fetchers.oanda_bars import fetch_candles, store_candles
from fetchers.fred_yields import fetch_historical_yields, store_yields_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backfill")


def fetch_candles_yahoo(instrument: str, days: int) -> list[dict]:
    """Fetch daily candle data from Yahoo Finance as a fallback."""
    import yfinance as yf
    from datetime import date, timedelta
    import pandas as pd

    ticker_map = {
        "EUR_USD": "EURUSD=X",
        "GBP_USD": "GBPUSD=X",
        "USD_JPY": "USDJPY=X"
    }
    ticker = ticker_map.get(instrument, f"{instrument.replace('_', '')}=X")

    end_date = date.today()
    # Fetch extra days to ensure we get 'days' worth of trading days
    start_date = end_date - timedelta(days=int(days * 1.5))

    logger.info(f"Downloading {instrument} from yfinance (ticker: {ticker}) from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if df.empty:
        raise ValueError(f"No data returned from yfinance for ticker {ticker}")

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    # Keep only the requested count from the tail
    df = df.tail(days)

    def _get_val(row, col):
        val = row[col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val)

    candles = []
    for dt, row in df.iterrows():
        # format dt to ISO string TIMESTAMPTZ expects
        time_str = dt.strftime("%Y-%m-%dT21:00:00.000000000Z")
        candles.append({
            "instrument": instrument,
            "granularity": "D",
            "bar_time": time_str,
            "open": _get_val(row, "Open"),
            "high": _get_val(row, "High"),
            "low": _get_val(row, "Low"),
            "close": _get_val(row, "Close"),
            "volume": int(_get_val(row, "Volume")) if "Volume" in df.columns else 0,
            "complete": True
        })
    logger.info(f"Fetched {len(candles)} daily bars from Yahoo Finance for {instrument}")
    return candles


def backfill_bars(days: int = 756, force_yahoo: bool = False):
    """
    Backfill daily bars for all instruments.
    Uses Yahoo Finance if OANDA API token is missing or if force_yahoo is True.
    """
    use_yahoo = force_yahoo or not OANDA_API_TOKEN
    source_name = "Yahoo Finance" if use_yahoo else "OANDA"
    logger.info(f"Starting backfill: {days} daily bars for {INSTRUMENTS} using {source_name}")

    for instrument in INSTRUMENTS:
        try:
            if use_yahoo:
                candles = fetch_candles_yahoo(instrument, days)
                store_candles(candles)
            else:
                logger.info(f"Fetching {days} daily bars from OANDA for {instrument}...")
                candles = fetch_candles(instrument, "D", days)
                store_candles(candles)
                logger.info(f"  Stored {len(candles)} daily bars for {instrument}")

                # Also fetch extended 4H history (6 bars/day * days)
                h4_count = min(days * 6, 5000)
                logger.info(f"Fetching {h4_count} 4H bars from OANDA for {instrument}...")
                h4_candles = fetch_candles(instrument, "H4", h4_count)
                store_candles(h4_candles)
                logger.info(f"  Stored {len(h4_candles)} 4H bars for {instrument}")

        except Exception as e:
            logger.error(f"Backfill failed for {instrument}: {e}")

    # Backfill FRED yield data
    logger.info("Backfilling FRED yield data...")
    try:
        yield_records = fetch_historical_yields(days)
        if yield_records:
            store_yields_batch(yield_records)
        else:
            logger.warning("No historical yields fetched from FRED.")
    except Exception as e:
        logger.error(f"Failed to backfill FRED yields: {e}")

    logger.info("Backfill complete.")


def backfill_and_fit_hmm(days: int = 756, force_yahoo: bool = False):
    """Backfill bars then fit the HMM on the history."""
    from models.database import init_schema
    from models.hmm_regime import RegimeDetector

    # Ensure Phase 2 schema exists
    schema_path = os.path.join(os.path.dirname(__file__), "sql", "schema_phase2.sql")
    if os.path.exists(schema_path):
        from models.database import get_connection
        conn = get_connection()
        try:
            with open(schema_path) as f:
                sql = f.read()
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("Phase 2 schema initialized")
        except Exception as e:
            conn.rollback()
            logger.warning(f"Phase 2 schema init: {e}")
        finally:
            conn.close()

    # Backfill bars
    backfill_bars(days, force_yahoo)

    # Fit HMM
    logger.info("Fitting HMM regime detector on backfilled data...")
    detector = RegimeDetector()
    version = detector.fit("EUR_USD")
    regime = detector.predict_regime("EUR_USD")
    logger.info(f"HMM fitted: version={version}")
    logger.info(f"Current regime: {regime['state_label']} (conf={regime['confidence']:.3f})")

    return regime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical data")
    parser.add_argument("--days", type=int, default=756, help="Days of history (default 756)")
    parser.add_argument("--hmm", action="store_true", help="Also fit HMM after backfill")
    parser.add_argument("--from-yahoo", action="store_true", help="Force backfill from Yahoo Finance")
    args = parser.parse_args()

    if args.hmm:
        backfill_and_fit_hmm(args.days, args.from_yahoo)
    else:
        backfill_bars(args.days, args.from_yahoo)
