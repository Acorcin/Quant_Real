"""
Fetcher: FRED API — US and German 2-Year bond yields.

Computes the yield spread (US 2Y - DE 2Y) and rate-of-change metrics,
including spread momentum (EUR–USD carry / rate differential dynamics).
Uses the fredapi library for clean data access.
"""
import logging
from datetime import date, timedelta

import pandas as pd
from fredapi import Fred

from config.settings import FRED_API_KEY, FRED_US_2Y_SERIES, FRED_DE_2Y_SERIES
from models.database import get_connection

logger = logging.getLogger(__name__)


def fetch_yields(lookback_days: int = 30) -> dict:
    """
    Fetch US 2Y and German 2Y yields from FRED.
    Attempts API first if key exists, otherwise falls back to public CSV.
    """
    if not FRED_API_KEY:
        logger.info("FRED_API_KEY not set. Falling back to public CSV download.")
        return fetch_yields_public(lookback_days)

    fred = Fred(api_key=FRED_API_KEY)
    end = date.today()
    start = end - timedelta(days=lookback_days + 10)  # extra buffer for weekends/holidays

    try:
        us_2y = fred.get_series(FRED_US_2Y_SERIES, observation_start=start, observation_end=end)
        us_2y = us_2y.dropna()
        logger.info(f"Fetched {len(us_2y)} US 2Y observations")
    except Exception as e:
        logger.error(f"Failed to fetch US 2Y yield: {e}")
        us_2y = None

    de_2y = None
    try:
        de_2y = fred.get_series(FRED_DE_2Y_SERIES, observation_start=start, observation_end=end)
        de_2y = de_2y.dropna()
        logger.info(f"Fetched {len(de_2y)} DE 2Y observations")
    except Exception as e:
        logger.warning(f"German 2Y series unavailable from FRED: {e}. Using fallback.")

    if us_2y is None or us_2y.empty:
        logger.warning("FRED API failed or returned empty. Falling back to public CSV.")
        return fetch_yields_public(lookback_days)

    latest_us = float(us_2y.iloc[-1])
    latest_date = us_2y.index[-1].date()

    latest_de = None
    if de_2y is not None and not de_2y.empty:
        latest_de = float(de_2y.iloc[-1])
    else:
        logger.warning("German 2Y yield unavailable — spread calculations will be partial")

    spread_bps = None
    spread_change_5d = None
    spread_change_20d = None

    # US-only changes (legacy / diagnostics)
    change_1d = None
    change_5d = None
    change_20d = None

    if len(us_2y) >= 2:
        change_1d = round((latest_us - float(us_2y.iloc[-2])) * 100, 2)
    if len(us_2y) >= 6:
        change_5d = round((latest_us - float(us_2y.iloc[-6])) * 100, 2)
    if len(us_2y) >= 21:
        change_20d = round((latest_us - float(us_2y.iloc[-21])) * 100, 2)

    # Aligned EUR–USD spread series (common dates only)
    if de_2y is not None and not de_2y.empty:
        aligned = pd.DataFrame({"us": us_2y, "de": de_2y}).ffill().dropna()
        if not aligned.empty:
            aligned["spread_bps"] = (aligned["us"] - aligned["de"]) * 100.0
            spread_series = aligned["spread_bps"]
            spread_bps = round(float(spread_series.iloc[-1]), 2)
            if len(spread_series) >= 6:
                spread_change_5d = round(
                    float(spread_series.iloc[-1] - spread_series.iloc[-6]), 2
                )
            if len(spread_series) >= 21:
                spread_change_20d = round(
                    float(spread_series.iloc[-1] - spread_series.iloc[-21]), 2
                )
    elif latest_de is not None:
        spread_bps = round((latest_us - latest_de) * 100, 2)

    result = {
        "date": str(latest_date),
        "us_2y_yield": latest_us,
        "de_2y_yield": latest_de,
        "yield_spread_bps": spread_bps,
        "spread_change_5d_bps": spread_change_5d,
        "spread_change_20d_bps": spread_change_20d,
        "us_2y_change_1d_bps": change_1d,
        "us_2y_change_5d_bps": change_5d,
        "us_2y_change_20d_bps": change_20d,
        "source": "fred",
    }

    logger.info(
        f"Yield data: US 2Y={latest_us}, spread={spread_bps} bps, "
        f"spread Δ5d={spread_change_5d}"
    )
    return result


def fetch_yields_public(lookback_days: int = 30) -> dict:
    """Fallback fetcher downloading public CSVs from FRED without an API key."""
    try:
        df_us = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2")
        df_us["DGS2"] = pd.to_numeric(df_us["DGS2"], errors="coerce")
        df_us["observation_date"] = pd.to_datetime(df_us["observation_date"])
        df_us = df_us.dropna()

        df_de = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01DEM156N")
        df_de["IRLTLT01DEM156N"] = pd.to_numeric(df_de["IRLTLT01DEM156N"], errors="coerce")
        df_de["observation_date"] = pd.to_datetime(df_de["observation_date"])
        df_de = df_de.dropna()

        # filter to lookback
        import datetime
        end = datetime.date.today()
        start = end - datetime.timedelta(days=lookback_days + 10)
        start_ts = pd.Timestamp(start)

        df_us = df_us[df_us["observation_date"] >= start_ts]
        df_de = df_de[df_de["observation_date"] >= start_ts]

        if df_us.empty:
            return {"error": "US 2Y yield data unavailable"}

        # merge
        df_merged = pd.merge(df_us, df_de, on="observation_date", how="left")
        df_merged = df_merged.sort_values("observation_date").reset_index(drop=True)
        df_merged["IRLTLT01DEM156N"] = df_merged["IRLTLT01DEM156N"].ffill()
        df_merged = df_merged.dropna()

        if df_merged.empty:
            return {"error": "No aligned yield data found"}

        df_merged["yield_spread_bps"] = (df_merged["DGS2"] - df_merged["IRLTLT01DEM156N"]) * 100.0
        df_merged["spread_change_5d_bps"] = df_merged["yield_spread_bps"].diff(5)
        df_merged["spread_change_20d_bps"] = df_merged["yield_spread_bps"].diff(20)

        # US-only changes
        df_merged["us_change_1d"] = df_merged["DGS2"].diff(1) * 100.0
        df_merged["us_change_5d"] = df_merged["DGS2"].diff(5) * 100.0
        df_merged["us_change_20d"] = df_merged["DGS2"].diff(20) * 100.0

        latest = df_merged.iloc[-1]
        
        return {
            "date": str(latest["observation_date"].date()),
            "us_2y_yield": float(latest["DGS2"]),
            "de_2y_yield": float(latest["IRLTLT01DEM156N"]),
            "yield_spread_bps": round(float(latest["yield_spread_bps"]), 2),
            "spread_change_5d_bps": round(float(latest["spread_change_5d_bps"]), 2) if pd.notna(latest["spread_change_5d_bps"]) else None,
            "spread_change_20d_bps": round(float(latest["spread_change_20d_bps"]), 2) if pd.notna(latest["spread_change_20d_bps"]) else None,
            "us_2y_change_1d_bps": round(float(latest["us_change_1d"]), 2) if pd.notna(latest["us_change_1d"]) else None,
            "us_2y_change_5d_bps": round(float(latest["us_change_5d"]), 2) if pd.notna(latest["us_change_5d"]) else None,
            "us_2y_change_20d_bps": round(float(latest["us_change_20d"]), 2) if pd.notna(latest["us_change_20d"]) else None,
            "source": "fred",
        }
    except Exception as e:
        logger.error(f"Failed to fetch public FRED yields: {e}")
        return {"error": str(e)}


def fetch_historical_yields(lookback_days: int = 1000) -> list[dict]:
    """Fetch the entire yield history and align it, return list of dicts for DB storage."""
    try:
        logger.info(f"Fetching public FRED yield history for the past {lookback_days} days...")
        df_us = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2")
        df_us["DGS2"] = pd.to_numeric(df_us["DGS2"], errors="coerce")
        df_us["observation_date"] = pd.to_datetime(df_us["observation_date"])
        df_us = df_us.dropna()

        df_de = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01DEM156N")
        df_de["IRLTLT01DEM156N"] = pd.to_numeric(df_de["IRLTLT01DEM156N"], errors="coerce")
        df_de["observation_date"] = pd.to_datetime(df_de["observation_date"])
        df_de = df_de.dropna()

        # filter to lookback
        import datetime
        end = datetime.date.today()
        start = end - datetime.timedelta(days=lookback_days + 30)
        start_ts = pd.Timestamp(start)

        df_us = df_us[df_us["observation_date"] >= start_ts]
        df_de = df_de[df_de["observation_date"] >= start_ts]

        # merge
        df_merged = pd.merge(df_us, df_de, on="observation_date", how="left")
        df_merged = df_merged.sort_values("observation_date").reset_index(drop=True)
        df_merged["IRLTLT01DEM156N"] = df_merged["IRLTLT01DEM156N"].ffill()
        df_merged = df_merged.dropna()

        df_merged["yield_spread_bps"] = (df_merged["DGS2"] - df_merged["IRLTLT01DEM156N"]) * 100.0
        df_merged["spread_change_5d_bps"] = df_merged["yield_spread_bps"].diff(5)
        df_merged["spread_change_20d_bps"] = df_merged["yield_spread_bps"].diff(20)

        records = []
        for _, row in df_merged.iterrows():
            records.append({
                "date": str(row["observation_date"].date()),
                "us_2y_yield": float(row["DGS2"]),
                "de_2y_yield": float(row["IRLTLT01DEM156N"]),
                "yield_spread_bps": round(float(row["yield_spread_bps"]), 2),
                "spread_change_5d_bps": round(float(row["spread_change_5d_bps"]), 2) if pd.notna(row["spread_change_5d_bps"]) else None,
                "spread_change_20d_bps": round(float(row["spread_change_20d_bps"]), 2) if pd.notna(row["spread_change_20d_bps"]) else None,
                "source": "fred",
            })
        return records
    except Exception as e:
        logger.error(f"Failed to fetch public historical yields: {e}")
        return []


def store_yields(data: dict):
    """Store yield data in the database."""
    if "error" in data:
        logger.warning(f"Skipping yield storage: {data['error']}")
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO yield_data (
                    date, us_2y_yield, de_2y_yield, yield_spread_bps,
                    spread_change_5d_bps, spread_change_20d_bps,
                    source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, source)
                DO UPDATE SET
                    us_2y_yield = EXCLUDED.us_2y_yield,
                    de_2y_yield = EXCLUDED.de_2y_yield,
                    yield_spread_bps = EXCLUDED.yield_spread_bps,
                    spread_change_5d_bps = EXCLUDED.spread_change_5d_bps,
                    spread_change_20d_bps = EXCLUDED.spread_change_20d_bps,
                    fetched_at = NOW()
            """, (
                data["date"],
                data["us_2y_yield"],
                data["de_2y_yield"],
                data["yield_spread_bps"],
                data.get("spread_change_5d_bps"),
                data.get("spread_change_20d_bps"),
                data["source"],
            ))
        conn.commit()
        logger.info(f"Stored yield data for {data['date']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing yields: {e}")
        raise
    finally:
        conn.close()


def store_yields_batch(records: list[dict]):
    """Store multiple yield data rows in a single batch transaction."""
    if not records:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for r in records:
                cur.execute("""
                    INSERT INTO yield_data (
                        date, us_2y_yield, de_2y_yield, yield_spread_bps,
                        spread_change_5d_bps, spread_change_20d_bps,
                        source
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, source)
                    DO UPDATE SET
                        us_2y_yield = EXCLUDED.us_2y_yield,
                        de_2y_yield = EXCLUDED.de_2y_yield,
                        yield_spread_bps = EXCLUDED.yield_spread_bps,
                        spread_change_5d_bps = EXCLUDED.spread_change_5d_bps,
                        spread_change_20d_bps = EXCLUDED.spread_change_20d_bps,
                        fetched_at = NOW()
                """, (
                    r["date"],
                    r["us_2y_yield"],
                    r["de_2y_yield"],
                    r["yield_spread_bps"],
                    r["spread_change_5d_bps"],
                    r["spread_change_20d_bps"],
                    r["source"],
                ))
        conn.commit()
        logger.info(f"Stored {len(records)} yield data rows in batch")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing batch yields: {e}")
        raise
    finally:
        conn.close()


def fetch_and_store():
    """Fetch yields from FRED and store in DB."""
    data = fetch_yields()
    store_yields(data)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_and_store()
    print(result)
