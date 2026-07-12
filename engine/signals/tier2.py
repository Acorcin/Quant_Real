"""
Tier 2 Signal Generators — Confirmation Only.

These signals never generate standalone trades. They confirm or
deny the direction proposed by Tier 1 signals, adding or
subtracting from composite confidence.

1. Daily Candle Pattern — engulfing, pin bar, inside bar
2. RSI-14 Extreme Zones — overbought/oversold
3. MA Alignment — price relative to MA-50 and MA-200
4. Multi-Timeframe — daily + 4H agreement
"""
import logging
import pandas as pd
from datetime import date
from models.database import fetch_all

logger = logging.getLogger(__name__)


def candle_pattern_confirmation(technical: dict, proposed_direction: str) -> dict:
    """
    Daily Candle Pattern Confirmation.

    Checks if today's candle pattern confirms the proposed direction.
    """
    confirmed = False
    detail_parts = []

    if proposed_direction == "long":
        if technical.get("is_engulfing_bull"):
            confirmed = True
            detail_parts.append("bullish engulfing")
        if technical.get("is_pin_bar_bull"):
            confirmed = True
            detail_parts.append("bullish pin bar")
    elif proposed_direction == "short":
        if technical.get("is_engulfing_bear"):
            confirmed = True
            detail_parts.append("bearish engulfing")
        if technical.get("is_pin_bar_bear"):
            confirmed = True
            detail_parts.append("bearish pin bar")

    # Inside bars are neutral — signal consolidation
    if technical.get("is_inside_bar"):
        detail_parts.append("inside bar (consolidation)")

    if technical.get("is_doji"):
        detail_parts.append("doji (indecision)")

    detail = ", ".join(detail_parts) if detail_parts else "no notable pattern"

    return _confirmation(
        "candle_pattern", confirmed,
        f"{'Confirmed' if confirmed else 'No confirmation'}: {detail}",
        {"patterns": detail_parts},
    )


def rsi_extreme_confirmation(technical: dict, proposed_direction: str) -> dict:
    """
    RSI-14 Extreme Zone Confirmation.

    - RSI < 30 confirms LONG (oversold)
    - RSI > 70 confirms SHORT (overbought)
    - RSI 30-70 is neutral
    """
    rsi = technical.get("rsi_14", 50.0)
    if rsi is None:
        return _confirmation("rsi_14", False, "RSI unavailable", {"rsi": None})

    confirmed = False
    if proposed_direction == "long" and rsi < 30:
        confirmed = True
    elif proposed_direction == "short" and rsi > 70:
        confirmed = True

    zone = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")

    return _confirmation(
        "rsi_14", confirmed,
        f"RSI-14 at {rsi:.1f} ({zone})" + (" — confirms" if confirmed else ""),
        {"rsi": round(rsi, 2), "zone": zone},
    )


def ma_alignment_confirmation(technical: dict, proposed_direction: str) -> dict:
    """
    Moving Average Alignment Confirmation.

    - Price above MA-50 AND MA-50 above MA-200 → confirms LONG (uptrend)
    - Price below MA-50 AND MA-50 below MA-200 → confirms SHORT (downtrend)
    """
    price_vs_50 = technical.get("price_vs_ma50")
    price_vs_200 = technical.get("price_vs_ma200")
    ma50 = technical.get("ma_50_value")
    ma200 = technical.get("ma_200_value")

    if price_vs_50 is None or price_vs_200 is None:
        return _confirmation("ma_alignment", False, "MA data insufficient (need 200 bars)", {})

    confirmed = False
    if proposed_direction == "long":
        confirmed = price_vs_50 > 0 and (ma50 > ma200 if ma50 and ma200 else price_vs_200 > 0)
    elif proposed_direction == "short":
        confirmed = price_vs_50 < 0 and (ma50 < ma200 if ma50 and ma200 else price_vs_200 < 0)

    return _confirmation(
        "ma_alignment", confirmed,
        f"Price vs MA50: {price_vs_50:+.2f}%, vs MA200: {price_vs_200:+.2f}%"
        + (" — trend aligned" if confirmed else " — no trend alignment"),
        {"price_vs_ma50": round(price_vs_50, 4), "price_vs_ma200": round(price_vs_200, 4)},
    )


def multi_timeframe_confirmation(
    run_date: date, instrument: str, proposed_direction: str
) -> dict:
    """
    Multi-Timeframe Confirmation — daily + 4H agreement.

    Checks if the last 2 completed 4H bars agree with the daily direction.
    """
    from datetime import datetime, time
    from zoneinfo import ZoneInfo
    
    ny_tz = ZoneInfo("America/New_York")
    cutoff_dt = datetime.combine(run_date, time(17, 0), tzinfo=ny_tz)

    rows = fetch_all("""
        SELECT bar_time, open, close
        FROM bars
        WHERE instrument = %s AND granularity = 'H4' AND complete = TRUE
          AND bar_time <= %s
        ORDER BY bar_time DESC
        LIMIT 2
    """, (instrument, cutoff_dt))

    if len(rows) < 2:
        return _confirmation("multi_timeframe", False, "Insufficient 4H data", {})

    # Check direction of last 2 4H bars
    h4_dirs = []
    for r in rows:
        d = 1 if float(r["close"]) > float(r["open"]) else -1
        h4_dirs.append(d)

    dir_map = {"long": 1, "short": -1}
    target = dir_map.get(proposed_direction, 0)

    # Both 4H bars agree with proposed direction
    agreement = sum(1 for d in h4_dirs if d == target)
    confirmed = agreement == 2

    return _confirmation(
        "multi_timeframe", confirmed,
        f"Last 2 4H bars: {agreement}/2 agree with {proposed_direction}",
        {"h4_directions": h4_dirs, "agreement": agreement},
    )


def generate_all_tier2(
    run_date: date, instrument: str, technical: dict, proposed_direction: str
) -> list[dict]:
    """Run all Tier 2 confirmations against the proposed direction."""
    if proposed_direction == "flat":
        return []

    confirmations = [
        candle_pattern_confirmation(technical, proposed_direction),
        rsi_extreme_confirmation(technical, proposed_direction),
        ma_alignment_confirmation(technical, proposed_direction),
        multi_timeframe_confirmation(run_date, instrument, proposed_direction),
    ]

    confirmed_count = sum(1 for c in confirmations if c["confirmed"])
    logger.info(
        f"Tier 2 confirmations: {confirmed_count}/4 confirm {proposed_direction}"
    )
    return confirmations


# ─── Helpers ─────────────────────────────────────────────

def _confirmation(detector: str, confirmed: bool, detail: str, metadata: dict = None) -> dict:
    return {
        "tier": 2,
        "detector": detector,
        "confirmed": confirmed,
        "direction": None,  # Tier 2 doesn't propose direction
        "strength": 1.0 if confirmed else 0.0,
        "detail": detail,
        "metadata": metadata or {},
    }
