"""
Fetcher: OANDA — Retail Position Ratios (sentiment).

Uses the ForexLabs legacy endpoint for historical position ratios.
If deprecated, falls back to a placeholder that logs the gap.
"""
import requests
import logging
from datetime import date
from config.settings import OANDA_API_TOKEN, OANDA_BASE_URL, INSTRUMENTS
from models.database import get_connection

logger = logging.getLogger(__name__)

HEADERS = {
    "Authorization": f"Bearer {OANDA_API_TOKEN}",
    "Content-Type": "application/json",
}

# ForexLabs legacy endpoint (may not be available on all account types)
FOREXLABS_URL = "https://api-fxpractice.oanda.com/labs/v1/historical_position_ratios"


def fetch_sentiment(instrument: str) -> dict | None:
    """
    Fetch retail position ratios for an instrument.

    Returns dict with pct_long, pct_short, ratio, or None on failure.
    """
    try:
        params = {
            "instrument": instrument,
            "period": 86400,  # 1 day
        }
        response = requests.get(
            FOREXLABS_URL,
            headers=HEADERS,
            params=params,
            timeout=15,
        )

        if response.status_code == 200:
            data = response.json()
            # ForexLabs returns data in various formats — parse what we get
            if isinstance(data, dict) and "data" in data:
                ratios = data["data"]
                if ratios:
                    latest = ratios[-1] if isinstance(ratios, list) else ratios
                    pct_long = float(latest.get("long_position_ratio", 0.5))
                    pct_short = 1.0 - pct_long
                    ratio = round(pct_long / pct_short, 3) if pct_short > 0 else 99.0

                    result = {
                        "instrument": instrument,
                        "date": str(date.today()),
                        "pct_long": round(pct_long, 4),
                        "pct_short": round(pct_short, 4),
                        "long_short_ratio": ratio,
                        "source": "oanda_position_ratios",
                    }
                    logger.info(f"Sentiment {instrument}: {pct_long:.1%} long / {pct_short:.1%} short")
                    return result

        # If we get here, the endpoint didn't work as expected
        logger.warning(
            f"ForexLabs sentiment endpoint returned status {response.status_code} "
            f"for {instrument}. This endpoint may have been deprecated."
        )
        return _fallback_sentiment(instrument)

    except Exception as e:
        logger.warning(f"Sentiment fetch failed for {instrument}: {e}. Using fallback.")
        return _fallback_sentiment(instrument)


def _fallback_sentiment(instrument: str) -> dict:
    """
    Fallback when ForexLabs endpoint is unavailable.
    Returns a neutral placeholder and logs the gap.
    """
    logger.info(
        f"Using fallback sentiment for {instrument}. "
        "TODO: Implement alternative source (OANDA web tool, myfxbook, etc.)"
    )
    return {
        "instrument": instrument,
        "date": str(date.today()),
        "pct_long": 0.50,
        "pct_short": 0.50,
        "long_short_ratio": 1.0,
        "source": "fallback_neutral",
    }


def store_sentiment(data: dict):
    """Store sentiment data in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sentiment (instrument, date, pct_long, pct_short, long_short_ratio, source)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (instrument, date, source)
                DO UPDATE SET
                    pct_long = EXCLUDED.pct_long,
                    pct_short = EXCLUDED.pct_short,
                    long_short_ratio = EXCLUDED.long_short_ratio,
                    fetched_at = NOW()
            """, (
                data["instrument"], data["date"],
                data["pct_long"], data["pct_short"],
                data["long_short_ratio"], data["source"],
            ))
        conn.commit()
        logger.info(f"Stored sentiment for {data['instrument']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing sentiment: {e}")
        raise
    finally:
        conn.close()


def fetch_and_store_all() -> dict:
    """Fetch sentiment for all instruments and store."""
    results = {}
    for instrument in INSTRUMENTS:
        data = fetch_sentiment(instrument)
        if data:
            store_sentiment(data)
            results[instrument] = data
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_and_store_all()
    for k, v in result.items():
        print(f"{k}: {v}")
