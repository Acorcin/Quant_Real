"""
Feature Vector Assembly.

Combines technical indicators, macro data, Kronos foundation-model outputs,
regime state, and signal outputs into the 25-feature vector that feeds the
XGBoost meta-model.

Intraday CME futures configuration: sentiment features (OANDA retail,
Perplexity AI) and FX carry features (swap pips) are REMOVED. Foundation-model
features come from the Kronos forecasting pipeline's per-origin features
parquet; L3 microstructure features are placeholders until the offline
Databento MBO pass lands.
"""
import json
import logging
from datetime import date
from models.database import get_connection, fetch_one

logger = logging.getLogger(__name__)

# Kronos structural regime encoding (forecasting pipeline structure_type)
KRONOS_REGIME_MAP = {"efficient": 0, "linear": 1, "vol_only": 2, "nonlinear": 3}


def assemble_feature_vector(
    run_date: date,
    instrument: str,
    technical: dict,
    regime: dict,
    signals_summary: dict,
    kronos: dict | None = None,
    l3: dict | None = None,
) -> dict:
    """
    Assemble the full feature vector for the meta-model.

    Args:
        run_date: Today's date.
        instrument: e.g. "M6E" (CME symbol).
        technical: Output from features.technical.compute_all_features().
        regime: Regime dict with state_id, state_label, confidence, days_in_regime.
        signals_summary: Summary of signal outputs (direction, count, strength, etc.).
        kronos: Foundation-model outputs for this origin from the forecasting
            pipeline's features parquet:
              p50_scaled        — kronos_p50_scaled (sigma-scaled median forecast)
              uncertainty       — P90-P10 spread in scaled-return space
              context_vol       — sigma_256 (trailing volatility scaler)
              structure_type    — 'efficient'|'linear'|'vol_only'|'nonlinear'
        l3: Offline Databento L3/MBO microstructure features (placeholders until
            the offline pass lands): order_book_imbalance, vpin.

    Returns:
        Dict of feature name → value (the 25-feature vector).
    """
    # Pull macro data from DB
    macro = _get_macro_data(run_date)
    spread_5d = macro.get("spread_change_5d_bps")
    spread_20d = macro.get("spread_change_20d_bps")
    us_5d = macro.get("us_2y_change_5d_bps")
    us_20d = macro.get("us_2y_change_20d_bps")
    if spread_5d is None and us_5d is not None:
        logger.warning("Feature vector using US-only 5d change fallback (spread_change_5d_bps missing)")
    if spread_20d is None and us_20d is not None:
        logger.warning("Feature vector using US-only 20d change fallback (spread_change_20d_bps missing)")

    kronos = kronos or {}
    l3 = l3 or {}
    structure = str(kronos.get("structure_type", "")).lower()
    if structure and structure not in KRONOS_REGIME_MAP:
        logger.warning("Unknown kronos structure_type %r — encoding as -1", structure)

    vector = {
        # ─── Regime (from HMM) ───
        "regime_state": regime.get("state_id", 1),
        "days_in_regime": regime.get("days_in_regime", 1),

        # ─── Macro (from FRED) ───
        "yield_spread_bps": macro.get("yield_spread_bps", 0.0),
        "yield_spread_change_5d": float(spread_5d if spread_5d is not None else (us_5d or 0.0)),
        "yield_spread_change_20d": float(spread_20d if spread_20d is not None else (us_20d or 0.0)),

        # ─── Kronos foundation model (forecasting pipeline) ───
        # NOTE: kronos_regime_encoded uses -1 for "missing/unknown" so the
        # structural veto (== 0, 'efficient') can never fire on absent data.
        "kronos_p50_scaled": float(kronos.get("p50_scaled", 0.0) or 0.0),
        "kronos_uncertainty": float(kronos.get("uncertainty", 0.0) or 0.0),
        "kronos_context_vol": float(kronos.get("context_vol", 0.0) or 0.0),
        "kronos_regime_encoded": KRONOS_REGIME_MAP.get(structure, -1),

        # ─── L3 microstructure (offline Databento pass — placeholders) ───
        "l3_order_book_imbalance": float(l3.get("order_book_imbalance", 0.0) or 0.0),
        "l3_vpin": float(l3.get("vpin", 0.0) or 0.0),

        # ─── Technical (from bars) ───
        "atr_14": technical.get("atr_14", 0.0),
        "rsi_14": technical.get("rsi_14", 50.0),
        "price_vs_ma50": technical.get("price_vs_ma50", 0.0),
        "price_vs_ma200": technical.get("price_vs_ma200", 0.0),
        "body_direction": technical.get("body_direction", 0),
        "body_pct_of_range": technical.get("body_pct_of_range", 0.5),

        # ─── EOD Event Reversal ───
        "eod_event_reversal": signals_summary.get("composite", {}).get("eod_event_reversal", 0),
        "event_surprise_magnitude": signals_summary.get("composite", {}).get("event_surprise_magnitude", 0.0),

        # ─── Time Features ───
        "day_of_week": run_date.weekday(),  # 0=Mon, 4=Fri
        "is_friday": 1 if run_date.weekday() == 4 else 0,

        # ─── Signal Summary ───
        "primary_signal_direction": signals_summary.get("composite", {}).get("direction_encoded", 0),
        "primary_signal_count": signals_summary.get("composite", {}).get("signal_count", 0),
        "composite_strength": signals_summary.get("composite", {}).get("composite_strength", 0.0),
        "tier2_confirmation_count": signals_summary.get("composite", {}).get("tier2_count", 0),
    }

    # Coerce None to 0.0 for model compatibility
    for k, v in vector.items():
        if v is None:
            vector[k] = 0.0

    return vector


def store_feature_vector(run_date: date, instrument: str, vector: dict):
    """Store the feature vector in the database."""
    # Ensure all values are JSON-serializable (Decimal → float)
    clean = {}
    for k, v in vector.items():
        if hasattr(v, 'as_integer_ratio'):  # Decimal or float-like
            clean[k] = float(v)
        elif isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, int):
            clean[k] = v
        else:
            try:
                clean[k] = float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                clean[k] = 0.0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feature_vectors (date, instrument, features)
                VALUES (%s, %s, %s)
                ON CONFLICT (date, instrument) DO UPDATE SET
                    features = EXCLUDED.features,
                    created_at = NOW()
            """, (str(run_date), instrument, json.dumps(clean)))
        conn.commit()
        logger.info(f"Stored feature vector for {instrument} on {run_date}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing feature vector: {e}")
        raise
    finally:
        conn.close()


def _get_macro_data(run_date: date) -> dict:
    """Pull latest macro/yield data from DB."""
    row = fetch_one(
        "SELECT * FROM yield_data WHERE date <= %s ORDER BY date DESC LIMIT 1",
        (str(run_date),),
    )
    return dict(row) if row else {}
