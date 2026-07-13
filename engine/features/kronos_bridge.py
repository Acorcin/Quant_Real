"""
Kronos bridge: read the forecasting pipeline's foundation-model output straight
from the `md` schema (populated by `forecasting/run.py --persist`) instead of a
hand-passed parquet snapshot.

Returns the RAW dict shape both consumers already expect:
  * features.vector.assemble_feature_vector(kronos=...)
  * signals.tier1.kronos_directional_signal(kronos)

      {p50_scaled, uncertainty, context_vol, structure_type, gate_state, ts}

Degrades gracefully: on any failure (DB down, md schema absent, no run yet for
the instrument) it returns None and logs a warning. Callers then fall through to
"no Kronos" — kronos_regime_encoded = -1, kronos_directional votes flat — so the
engine keeps running when the pipeline hasn't produced a forecast.

Source-of-truth query: the newest run for the instrument that actually carries
foundation-model values (a --no-foundation persist run leaves kronos_p50_scaled
NULL), then that run's most recent origin. This is the point-in-time forecast the
engine should act on now.

LIVE ONLY. `load_latest_kronos` returns the single most-recent forecast, so it is
correct for the daily loop but WRONG for historical backtests: applying today's
forecast to every past date is look-ahead. The historical feature builders
(walkforward_test.py, generate_historical_features.py) must join md.kronos_features
to each origin's timestamp instead — a point-in-time loader that is deliberately
not implemented here yet. They currently pass kronos=None (a safe no-op) until
that lands.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# md.kronos_features.structure -> the raw structure_type string the engine's
# KRONOS_REGIME_MAP re-encodes. (Enum values already match; kept explicit so a
# schema rename can't silently drift the bridge.)
_STRUCTURE_PASSTHROUGH = {"efficient", "linear", "vol_only", "nonlinear"}

_LATEST_KRONOS_SQL = """
    WITH latest AS (
        SELECT r.run_id
        FROM md.runs r
        WHERE r.instrument = %s
          AND EXISTS (
              SELECT 1 FROM md.kronos_features k
              WHERE k.run_id = r.run_id
                AND k.kronos_p50_scaled IS NOT NULL
          )
        ORDER BY r.started_at DESC
        LIMIT 1
    )
    SELECT
        k.kronos_p50_scaled      AS p50_scaled,
        k.kronos_spread_scaled   AS uncertainty,
        k.sigma_256              AS context_vol,
        k.structure::text        AS structure_type,
        k.gate_state::text       AS gate_state,
        k.ts                     AS ts,
        k.origin_seq             AS origin_seq
    FROM md.kronos_features k
    JOIN latest USING (run_id)
    WHERE k.kronos_p50_scaled IS NOT NULL
    ORDER BY k.origin_seq DESC
    LIMIT 1
"""


def load_latest_kronos(instrument: str) -> dict | None:
    """Latest point-in-time Kronos features for `instrument` from md, or None.

    `instrument` is the md series name (e.g. "M6EH6"), which may differ from the
    engine's trading symbol during the CME transition — resolve it via
    settings.KRONOS_INSTRUMENT at the call site."""
    try:
        from models.database import fetch_one
        row = fetch_one(_LATEST_KRONOS_SQL, (instrument,))
    except Exception as e:  # DB down / md schema missing — never kill the loop
        logger.warning("Kronos bridge unavailable (%s); proceeding without "
                       "foundation features", e)
        return None

    if not row:
        logger.info("No Kronos features in md for %s yet — engine will run "
                    "with kronos_regime_encoded=-1", instrument)
        return None

    structure = str(row.get("structure_type") or "").lower()
    if structure not in _STRUCTURE_PASSTHROUGH:
        logger.warning("Kronos bridge: unexpected structure_type %r for %s",
                       structure, instrument)

    kronos = {
        "p50_scaled": _f(row.get("p50_scaled")),
        "uncertainty": _f(row.get("uncertainty")),
        "context_vol": _f(row.get("context_vol")),
        "structure_type": structure,
        "gate_state": row.get("gate_state"),
        "ts": row.get("ts"),
    }
    logger.info("Kronos bridge %s: structure=%s p50=%+.4f spread=%.4f "
                "gate=%s (origin %s)", instrument, structure,
                kronos["p50_scaled"], kronos["uncertainty"],
                kronos["gate_state"], row.get("origin_seq"))
    return kronos


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
