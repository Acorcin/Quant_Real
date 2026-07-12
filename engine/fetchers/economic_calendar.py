"""
Fetcher: Economic Calendar — high-impact events.

Fetches today's and tomorrow's high-impact economic events.
Primary source: Investing.com calendar API (public endpoint).
Fallback: static high-impact event schedule.
"""
import requests
import logging
from datetime import date, datetime, timedelta
from models.database import get_connection

logger = logging.getLogger(__name__)

# Public economic calendar endpoints (no API key needed)
# These are commonly available — we'll try multiple sources
CALENDAR_SOURCES = [
    {
        "name": "nager_holidays",  # Placeholder — real implementation needs a calendar API
        "url": None,
    }
]

# Known high-impact recurring events (static fallback)
HIGH_IMPACT_EVENTS = {
    "USD": [
        "Non-Farm Payrolls", "CPI", "Core CPI", "Core PCE",
        "FOMC Rate Decision", "Fed Chair Press Conference",
        "GDP", "ISM Manufacturing PMI", "Retail Sales",
        "Initial Jobless Claims", "PPI", "Consumer Confidence",
    ],
    "EUR": [
        "ECB Rate Decision", "ECB Press Conference",
        "German CPI", "Eurozone CPI", "German GDP",
        "Eurozone GDP", "German PMI", "Eurozone PMI",
    ],
}


def fetch_calendar_events() -> dict:
    """
    Fetch economic calendar events for today and tomorrow.

    Returns a dict with 'today_events' and 'tomorrow_events' lists.

    NOTE: A fully automated calendar feed requires either:
    1. A paid API (Trading Economics, MetaTrader, FXStreet)
    2. Collecting from a public source (ForexFactory, Investing.com)
    3. Manual daily input

    For MVP, this returns the structure with a TODO for the data source.
    The Perplexity AI agent (Section 2) partially covers this by reading
    financial news and detecting event impacts.
    """
    today = date.today()
    tomorrow = today + timedelta(days=1)
    is_friday = today.weekday() == 4

    # Attempt to fetch from a public calendar API
    events = _try_fetch_online_calendar(today, tomorrow)

    if not events:
        # Return structure with empty events — the AI sentiment step
        # will catch major events via news search
        events = {
            "today_events": [],
            "tomorrow_events": [],
            "is_friday": is_friday,
            "is_pre_holiday": False,
            "source": "none_available",
            "note": "Calendar feed not configured. Perplexity AI step will catch major events via news.",
        }
        logger.warning(
            "No calendar data source configured. "
            "Add a calendar API or manually input events. "
            "The Perplexity sentiment step partially covers this."
        )

    events["is_friday"] = is_friday
    events["date"] = str(today)
    return events


def _try_fetch_online_calendar(today: date, tomorrow: date) -> dict | None:
    """
    Try to fetch calendar from an online source.

    TODO: Implement one of these options:
    1. Trading Economics API (paid, ~$50/mo, best quality)
    2. ForexFactory calendar (free, needs HTML parsing)
    3. MQL5 calendar (free, good quality)
    4. Custom: maintain a Google Sheet with this week's events
       and read it via the Google Sheets API (you have Drive connected)

    For now, returns None to trigger fallback.
    """
    return None


def store_calendar_events(events: list[dict]):
    """Store calendar events in the database."""
    if not events:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for evt in events:
                cur.execute("""
                    INSERT INTO calendar_events
                        (event_name, currency, impact, event_time, forecast, previous, actual, surprise_direction)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_name, event_time) DO UPDATE SET
                        actual = EXCLUDED.actual,
                        surprise_direction = EXCLUDED.surprise_direction,
                        fetched_at = NOW()
                """, (
                    evt.get("name", ""),
                    evt.get("currency", ""),
                    evt.get("impact", "medium"),
                    evt.get("time", datetime.now()),
                    evt.get("forecast"),
                    evt.get("previous"),
                    evt.get("actual"),
                    evt.get("surprise_direction"),
                ))
        conn.commit()
        logger.info(f"Stored {len(events)} calendar events")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing calendar events: {e}")
        raise
    finally:
        conn.close()


def fetch_and_store() -> dict:
    """Fetch calendar events and store any that have data."""
    data = fetch_calendar_events()

    all_events = data.get("today_events", []) + data.get("tomorrow_events", [])
    if all_events:
        store_calendar_events(all_events)

    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_and_store()
    print(result)
