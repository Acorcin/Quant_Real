"""
Tier 1 Signal Generators — Primary Trading Signals.

These are the core alpha signals. Each generates an independent
directional signal (long/short/flat) with a strength score.

1. Yield Spread Momentum — macro rate differential dynamics
2. Kronos Directional — foundation-model median forecast, conviction scaled
   inversely to its own quantile spread
3. EOD Event Reversal — institutional pattern after high-impact events

Intraday CME futures configuration: sentiment signals (OANDA retail fade,
Perplexity AI macro) are REMOVED.
"""
import logging
from datetime import date
from models.database import fetch_one, fetch_all

logger = logging.getLogger(__name__)

# Regime-adaptive thresholds
YIELD_THRESHOLDS = {
    0: 8.0,   # low_vol_trend: tighter threshold (smaller moves matter)
    1: 15.0,  # high_vol_choppy: wider (need bigger moves to be meaningful)
    2: 20.0,  # high_vol_crash: widest (extreme noise)
}

# kronos_directional_signal: |p50| below this (in sigma-scaled units) is noise
KRONOS_MIN_EDGE_SCALED = 0.05


def yield_spread_momentum(run_date: date, instrument: str, regime_state: int) -> dict:
    """
    Yield Spread Momentum Signal.

    Logic:
    - If 5d yield spread change > threshold AND favors USD → SHORT EUR/USD
    - If 5d yield spread change < -threshold AND favors EUR → LONG EUR/USD
    - Threshold is regime-adaptive (tighter in trends, wider in chop)
    """
    macro = fetch_one(
        "SELECT * FROM yield_data WHERE date <= %s ORDER BY date DESC LIMIT 1",
        (str(run_date),),
    )
    if not macro:
        return _no_signal("yield_spread_momentum", "No macro data available")

    raw_spread = macro.get("spread_change_5d_bps")
    us_only = macro.get("us_2y_change_5d_bps")
    if raw_spread is None and us_only is None:
        return _no_signal("yield_spread_momentum", "No spread or US 2Y 5d change data")
    if raw_spread is None and us_only is not None:
        logger.warning("yield_spread_momentum using US-only 5d change fallback")
    spread_change_5d = float(raw_spread if raw_spread is not None else us_only)
    threshold = YIELD_THRESHOLDS.get(regime_state, 15.0)

    if spread_change_5d > threshold:
        # US yields rising relative → USD strength → SHORT EUR/USD
        strength = min(abs(spread_change_5d) / (threshold * 2), 1.0)
        return _signal(
            "yield_spread_momentum", "short", strength,
            f"5d spread change +{spread_change_5d:.1f} bps > threshold {threshold} → USD strength",
            {"spread_change_5d_bps": spread_change_5d, "threshold": threshold},
        )
    elif spread_change_5d < -threshold:
        # US yields falling relative → EUR strength → LONG EUR/USD
        strength = min(abs(spread_change_5d) / (threshold * 2), 1.0)
        return _signal(
            "yield_spread_momentum", "long", strength,
            f"5d spread change {spread_change_5d:.1f} bps < -{threshold} → EUR strength",
            {"spread_change_5d_bps": spread_change_5d, "threshold": threshold},
        )
    else:
        return _signal(
            "yield_spread_momentum", "flat", 0.0,
            f"5d spread change {spread_change_5d:.1f} bps within ±{threshold} threshold",
            {"spread_change_5d_bps": spread_change_5d, "threshold": threshold},
        )


def kronos_directional_signal(kronos: dict | None) -> dict:
    """
    Kronos Directional Signal — foundation-model median forecast.

    Logic:
    - Vote LONG/SHORT on the sign of `kronos_p50_scaled` (the sigma-scaled
      median of the Kronos predictive distribution).
    - Conviction scales INVERSELY with `kronos_uncertainty` (the P90-P10
      quantile spread): a wide predictive distribution means the model itself
      is unsure, so the vote counts for less.

        strength = clip( |p50| / (|p50| + spread), 0, 1 )

      i.e. the fraction of the model's own uncertainty budget taken up by its
      directional edge. p50 >> spread → ~1.0; p50 << spread → ~0.
    - |p50| below KRONOS_MIN_EDGE_SCALED is treated as noise → flat.
    - A zero/negative spread is a degenerate distribution (collapsed sampler),
      not high conviction → flat. The forecasting pipeline's own gate flags
      these as DEGENERATE_FORECAST; we mirror that stance here.
    """
    kronos = kronos or {}
    if "p50_scaled" not in kronos:
        return _no_signal("kronos_directional", "No Kronos features for this origin")

    p50 = float(kronos.get("p50_scaled", 0.0) or 0.0)
    spread = float(kronos.get("uncertainty", 0.0) or 0.0)

    if spread <= 0:
        return _signal(
            "kronos_directional", "flat", 0.0,
            f"Degenerate Kronos distribution (P90-P10 spread {spread:.4f} <= 0)",
            {"p50_scaled": p50, "uncertainty": spread, "degenerate": True},
        )

    if abs(p50) < KRONOS_MIN_EDGE_SCALED:
        return _signal(
            "kronos_directional", "flat", 0.0,
            f"Kronos P50 {p50:+.4f} below noise floor ±{KRONOS_MIN_EDGE_SCALED}",
            {"p50_scaled": p50, "uncertainty": spread},
        )

    direction = "long" if p50 > 0 else "short"
    strength = min(abs(p50) / (abs(p50) + spread), 1.0)
    return _signal(
        "kronos_directional", direction, strength,
        f"Kronos P50 {p50:+.4f} (scaled), spread {spread:.4f} → {direction} "
        f"with conviction {strength:.2f}",
        {"p50_scaled": p50, "uncertainty": spread},
    )


def eod_event_reversal(run_date: date, instrument: str, technical: dict) -> dict:
    """
    EOD Event Reversal Signal.

    Logic:
    - Was there a high-impact event today?
    - Did the event surprise in one direction (e.g., bullish USD)?
    - Did the daily candle CLOSE in the OPPOSITE direction?
    - If yes → institutional signal: market absorbed the news and reversed.

    This is a well-documented institutional pattern:
    "The market tells you what it thinks of the news by the close."
    """
    events = fetch_all(
        """SELECT * FROM calendar_events
           WHERE DATE(event_time) = %s AND impact = 'high'
           ORDER BY event_time""",
        (str(run_date),),
    )

    if not events:
        return _signal(
            "eod_event_reversal", "flat", 0.0,
            "No high-impact events today",
            {"triggered": False},
        )

    def _usd_surprise_score(sd: str) -> float:
        if not sd or sd == "neutral":
            return 0.0
        if "positive_usd" in sd:
            return 1.0
        if "negative_usd" in sd:
            return -1.0
        return 0.0

    surprise_scores = [_usd_surprise_score(str(e.get("surprise_direction") or "")) for e in events]
    has_positive = any(score > 0 for score in surprise_scores)
    has_negative = any(score < 0 for score in surprise_scores)
    has_conflict = has_positive and has_negative
    net_usd = float(sum(surprise_scores))
    non_neutral = [e for e in events if (e.get("surprise_direction") or "") not in ("", "neutral")]

    if not non_neutral:
        return _signal(
            "eod_event_reversal", "flat", 0.0,
            "Events occurred but no surprise_direction set",
            {"triggered": False, "events_count": len(events)},
        )

    if net_usd == 0.0:
        return _signal(
            "eod_event_reversal", "flat", 0.0,
            "High-impact events conflict (net USD surprise ≈ 0)",
            {
                "triggered": False,
                "events_count": len(events),
                "conflicting_surprises": True,
                "per_event_scores": surprise_scores,
            },
        )

    surprise_direction = "positive_usd" if net_usd > 0 else "negative_usd"

    body_dir = technical.get("body_direction", 0)

    # Determine if there's a reversal
    # positive_usd surprise + bullish EUR candle = reversal → LONG EUR/USD
    # negative_usd surprise + bearish EUR candle = reversal → SHORT EUR/USD
    usd_positive = "positive_usd" in surprise_direction
    usd_negative = "negative_usd" in surprise_direction

    reversal_detected = False
    direction = "flat"

    if usd_positive and body_dir == 1:  # USD-positive event, but EUR candle is bullish
        reversal_detected = True
        direction = "long"
    elif usd_negative and body_dir == -1:  # USD-negative event, but EUR candle is bearish
        reversal_detected = True
        direction = "short"

    if reversal_detected:
        strength = min(0.85 + 0.02 * (len(non_neutral) - 1), 1.0)
        return _signal(
            "eod_event_reversal", direction, strength,
            f"Aggregated USD surprise ({surprise_direction}) vs candle → institutional reversal",
            {
                "triggered": True,
                "surprise": surprise_direction,
                "net_usd_score": net_usd,
                "conflicting_surprises": has_conflict,
                "events_count": len(events),
                "candle_direction": body_dir,
            },
        )
    else:
        return _signal(
            "eod_event_reversal", "flat", 0.0,
            f"Aggregated surprise '{surprise_direction}' aligned with candle — no reversal",
            {
                "triggered": False,
                "surprise": surprise_direction,
                "net_usd_score": net_usd,
                "conflicting_surprises": has_conflict,
                "candle_direction": body_dir,
            },
        )


def generate_all_tier1(run_date: date, instrument: str, regime_state: int,
                       technical: dict, kronos: dict | None = None) -> list[dict]:
    """Run all Tier 1 generators and return list of signals.

    `kronos` is the foundation-model output dict for this origin (p50_scaled,
    uncertainty, ...) from the forecasting pipeline's features parquet."""
    signals = [
        yield_spread_momentum(run_date, instrument, regime_state),
        kronos_directional_signal(kronos),
        eod_event_reversal(run_date, instrument, technical),
    ]
    logger.info(f"Tier 1 signals: {[(s['detector'], s['direction'], s['strength']) for s in signals]}")
    return signals


# ─── Helpers ─────────────────────────────────────────────

def _signal(detector: str, direction: str, strength: float, detail: str, metadata: dict = None) -> dict:
    return {
        "tier": 1,
        "detector": detector,
        "direction": direction,
        "strength": round(strength, 4),
        "detail": detail,
        "metadata": metadata or {},
    }


def _no_signal(detector: str, reason: str) -> dict:
    return _signal(detector, "flat", 0.0, reason, {"no_data": True})
