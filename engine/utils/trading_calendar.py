"""
Trading Calendar Utilities.

Provides helper functions for handling trading day logic (weekends, holidays).
"""
from datetime import date, timedelta

# US + EU major holidays that close markets (simplified subset).
# For production, use pandas_market_calendars or a dedicated holiday library.
MARKET_HOLIDAYS_2026 = [
    date(2026, 1, 1),   # New Year
    date(2026, 4, 10),  # Good Friday
    date(2026, 5, 25),  # Memorial Day (US)
    date(2026, 7, 3),   # Independence Day observed (US)
    date(2026, 9, 7),   # Labor Day (US)
    date(2026, 11, 26), # Thanksgiving (US)
    date(2026, 12, 25), # Christmas
]


def next_trading_day(run_date: date) -> date:
    """
    Calculate the next trading day after `run_date`, skipping weekends
    and known major US/EU holidays.

    For EUR/USD, the trading week is Mon-Fri. Saturday and Sunday are
    always non-trading days. Additional holidays are checked against
    MARKET_HOLIDAYS_2026.

    Args:
        run_date: The reference date (typically today).

    Returns:
        The next calendar date that is a valid trading day.

    Example:
        If run_date is Friday 2026-04-03, returns Monday 2026-04-06.
        If run_date is Thursday before Good Friday (2026-04-09),
        returns Tuesday 2026-04-14 (skip Fri, Sat, Sun, Mon).
    """
    candidate = run_date + timedelta(days=1)
    while True:
        # Skip weekends
        if candidate.weekday() >= 5:  # 5=Sat, 6=Sun
            candidate += timedelta(days=1)
            continue
        # Skip holidays
        if candidate in MARKET_HOLIDAYS_2026:
            candidate += timedelta(days=1)
            continue
        # Found a valid trading day
        return candidate


def is_trading_day(d: date) -> bool:
    """Check if a given date is a trading day."""
    if d.weekday() >= 5:
        return False
    if d in MARKET_HOLIDAYS_2026:
        return False
    return True
