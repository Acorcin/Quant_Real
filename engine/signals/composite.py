"""
Composite Signal Assembly.

Combines Tier 1 and Tier 2 signals into a single directional
decision with a composite strength score.
"""
import json
import logging
from datetime import date
from models.database import get_connection

logger = logging.getLogger(__name__)


def compute_composite(tier1_signals: list[dict], tier2_signals: list[dict]) -> dict:
    """
    Compute composite direction and strength from all signals.

    Tier 1 signals vote on direction (weighted by strength).
    Tier 2 confirmations boost or penalize the composite strength.

    Returns:
        Dict with composite_direction, composite_strength,
        direction_encoded, signal_count, tier2_count,
        eod_event_reversal, event_surprise_magnitude.
    """
    # Tally Tier 1 directional votes
    long_score = 0.0
    short_score = 0.0
    long_count = 0
    short_count = 0
    eod_reversal = 0
    event_surprise = 0.0

    for sig in tier1_signals:
        if sig["direction"] == "long":
            long_score += sig["strength"]
            long_count += 1
        elif sig["direction"] == "short":
            short_score += sig["strength"]
            short_count += 1

        if sig["detector"] == "eod_event_reversal":
            meta = sig.get("metadata", {})
            eod_reversal = 1 if meta.get("triggered", False) else 0
            if eod_reversal:
                event_surprise = float(meta.get("net_usd_score", sig["strength"]))
            elif meta.get("conflicting_surprises"):
                event_surprise = 0.0

    active_count = long_count + short_count

    if active_count == 0:
        return {
            "composite_direction": "flat",
            "composite_strength": 0.0,
            "direction_encoded": 0,
            "signal_count": 0,
            "tier2_count": 0,
            "eod_event_reversal": 0,
            "event_surprise_magnitude": 0.0,
        }

    tie_break = False
    # Determine primary direction (average strength only among votes on the winning side)
    if long_score > short_score:
        direction = "long"
        base_strength = long_score / long_count if long_count else 0.0
    elif short_score > long_score:
        direction = "short"
        base_strength = short_score / short_count if short_count else 0.0
    else:
        tie_break = True
        direction = "flat"
        base_strength = 0.0

    # Tier 2 adjustment: each confirmation adds +0.05, each non-confirmation -0.02
    tier2_count = 0
    t2_adjustment = 0.0
    for conf in tier2_signals:
        if conf.get("confirmed"):
            t2_adjustment += 0.05
            tier2_count += 1
        else:
            t2_adjustment -= 0.02

    composite_strength = 0.0 if tie_break else max(0.0, min(1.0, base_strength + t2_adjustment))

    # If composite strength is too weak, go flat
    if composite_strength < 0.15:
        direction = "flat"
        composite_strength = 0.0

    direction_map = {"long": 1, "short": -1, "flat": 0}

    result = {
        "composite_direction": direction,
        "composite_strength": round(composite_strength, 4),
        "direction_encoded": direction_map.get(direction, 0),
        "signal_count": active_count,
        "tier2_count": tier2_count,
        "eod_event_reversal": eod_reversal,
        "event_surprise_magnitude": round(event_surprise, 4),
        "tie_break": tie_break,
    }

    logger.info(
        f"Composite: {direction.upper()} (strength={composite_strength:.3f}, "
        f"T1 signals={active_count}, T2 confirms={tier2_count})"
    )
    return result


def store_signals(run_date: date, instrument: str,
                  tier1: list[dict], tier2: list[dict]):
    """Store all signals in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for sig in tier1 + tier2:
                cur.execute("""
                    INSERT INTO signals (date, instrument, tier, detector,
                        direction, strength, detail, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, instrument, detector) DO UPDATE SET
                        direction = EXCLUDED.direction,
                        strength = EXCLUDED.strength,
                        detail = EXCLUDED.detail,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW()
                """, (
                    str(run_date), instrument,
                    sig["tier"], sig["detector"],
                    sig.get("direction") or ("confirmed" if sig.get("confirmed") else "not_confirmed"),
                    sig["strength"],
                    sig["detail"],
                    json.dumps(sig.get("metadata", {})),
                ))
        conn.commit()
        logger.info(f"Stored {len(tier1)} T1 + {len(tier2)} T2 signals for {instrument}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing signals: {e}")
        raise
    finally:
        conn.close()
