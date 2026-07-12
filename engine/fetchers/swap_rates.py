"""
Fetcher: OANDA V20 — Swap / Financing Rates.

Pulls the overnight financing rates for each instrument.
OANDA charges/pays financing at 5 PM EST daily.
"""
import requests
import logging
from datetime import date, datetime
from config.settings import OANDA_API_TOKEN, OANDA_BASE_URL, OANDA_ACCOUNT_ID, INSTRUMENTS
from models.database import get_connection

logger = logging.getLogger(__name__)

HEADERS = {
    "Authorization": f"Bearer {OANDA_API_TOKEN}",
    "Content-Type": "application/json",
}


def fetch_swap_rate(instrument: str) -> dict | None:
    """
    Fetch financing/swap rates from OANDA V20 instrument details.

    OANDA provides financing rates in the instrument details endpoint.
    The financing rate is annualized — we convert to daily pips.
    """
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/instruments"
    params = {"instruments": instrument}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        instruments = data.get("instruments", [])
        if not instruments:
            logger.warning(f"No instrument data for {instrument}")
            return None

        inst = instruments[0]
        financing = inst.get("financing", {})

        # OANDA provides daily financing rates
        long_rate = float(financing.get("longRate", 0))
        short_rate = float(financing.get("shortRate", 0))

        # Convert annual rate to approximate daily pip cost
        # This is a rough approximation — exact depends on position size and pip value
        # For EUR/USD: rate / 365 * price ≈ daily cost in price terms
        # We'll store the raw rates and let the decision engine do precise math
        today = date.today()
        is_wednesday = today.weekday() == 2  # triple swap day

        result = {
            "instrument": instrument,
            "date": str(today),
            "long_rate_annual": long_rate,
            "short_rate_annual": short_rate,
            "long_swap_pips": round(long_rate / 365 * 10000, 4),  # approx daily pips
            "short_swap_pips": round(short_rate / 365 * 10000, 4),
            "triple_swap_day": is_wednesday,
            "source": "oanda",
        }

        logger.info(
            f"Swap {instrument}: long={result['long_swap_pips']} pips/day, "
            f"short={result['short_swap_pips']} pips/day"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to fetch swap rates for {instrument}: {e}")
        return None


def store_swap_rate(data: dict):
    """Store swap rate data in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO swap_rates (instrument, date, long_swap_pips, short_swap_pips, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (instrument, date, source)
                DO UPDATE SET
                    long_swap_pips = EXCLUDED.long_swap_pips,
                    short_swap_pips = EXCLUDED.short_swap_pips,
                    fetched_at = NOW()
            """, (
                data["instrument"], data["date"],
                data["long_swap_pips"], data["short_swap_pips"],
                data["source"],
            ))
        conn.commit()
        logger.info(f"Stored swap rates for {data['instrument']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing swap rates: {e}")
        raise
    finally:
        conn.close()


def fetch_and_store_all() -> dict:
    """Fetch swap rates for all instruments and store."""
    results = {}
    for instrument in INSTRUMENTS:
        data = fetch_swap_rate(instrument)
        if data:
            store_swap_rate(data)
            results[instrument] = data
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_and_store_all()
    for k, v in result.items():
        print(f"{k}: {v}")
