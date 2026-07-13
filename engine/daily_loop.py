#!/usr/bin/env python3
"""
Quant EOD Engine — Daily Loop Orchestrator

Master script: data collection + prediction pipeline.
Schedule cron at or after 5:15 PM America/New_York, Monday–Friday, so the
daily Forex bar is typically complete (OANDA marks complete after the 5 PM ET fix).

Pipeline steps:
  Phase 1 (Data Collection):
    1. Pull OANDA bars (daily + 4H)
    2. Pull FRED yield data
    3. Pull OANDA sentiment/position ratios
    4. Pull swap/financing rates
    5. Pull economic calendar
    6. Run Perplexity AI sentiment analysis
  Phase 2 (Prediction):
    7. HMM regime detection
    8. Generate Tier 1 + Tier 2 signals
    9. Assemble feature vector
   10. Meta-model prediction (XGBoost)
  Finalize:
   11. Assemble daily snapshot
   12. Send Discord notification
   13. Log pipeline run
"""
import sys
import os
import json
import logging
from datetime import datetime, date, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import LOG_DIR, LOG_LEVEL, PRIMARY_INSTRUMENT
from models.database import init_schema, get_connection
from fetchers.oanda_bars import fetch_and_store_all as fetch_bars
from fetchers.fred_yields import fetch_and_store as fetch_yields
# sentiment / swap / Perplexity fetchers removed (intraday CME configuration)
from fetchers.economic_calendar import fetch_and_store as fetch_calendar
from fetchers.discord_notify import send_signal, send_error_alert

# Phase 2 imports
from features.technical import compute_all_features
from features.vector import assemble_feature_vector, store_feature_vector
from models.hmm_regime import RegimeDetector
from signals.tier1 import generate_all_tier1
from signals.tier2 import generate_all_tier2
from signals.composite import compute_composite, store_signals
from models.meta_model import MetaModel

# ─── Logging Setup ────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"daily_{date.today().isoformat()}.log")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_loop")


def log_pipeline_run(run_date: date, started_at: datetime, status: str,
                     steps: dict, errors: dict):
    """Record the pipeline run in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pipeline_runs (run_date, started_at, completed_at, status, steps_completed, errors)
                VALUES (%s, %s, NOW(), %s, %s, %s)
                ON CONFLICT (run_date) DO UPDATE SET
                    completed_at = NOW(),
                    status = EXCLUDED.status,
                    steps_completed = EXCLUDED.steps_completed,
                    errors = EXCLUDED.errors
            """, (
                str(run_date), started_at, status,
                json.dumps(steps), json.dumps(errors),
            ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to log pipeline run: {e}")
    finally:
        conn.close()


def assemble_daily_snapshot(bars_result, yields_result, sentiment_result,
                            swaps_result, calendar_result, ai_result) -> dict:
    """Assemble all collected data into a single daily snapshot JSON."""
    today = date.today()

    snapshot = {
        "type": "daily_snapshot",
        "date": str(today),
        "collection_time": datetime.now(timezone.utc).isoformat(),
        "bars_summary": bars_result,
        "macro": yields_result if isinstance(yields_result, dict) else {},
        "sentiment": sentiment_result if isinstance(sentiment_result, dict) else {},
        "swap_rates": swaps_result if isinstance(swaps_result, dict) else {},
        "calendar": calendar_result if isinstance(calendar_result, dict) else {},
        "ai_sentiment": {
            k: v for k, v in (ai_result or {}).items()
            if k != "raw_response"  # don't duplicate the full response in snapshot
        },
        "is_friday": today.weekday() == 4,
    }

    return snapshot


def store_snapshot(snapshot: dict):
    """Store the assembled snapshot in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO daily_snapshots (date, instrument, snapshot_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (date, instrument) DO UPDATE SET
                    snapshot_data = EXCLUDED.snapshot_data,
                    created_at = NOW()
            """, (
                snapshot["date"], PRIMARY_INSTRUMENT,
                json.dumps(snapshot, default=str),
            ))
        conn.commit()
        logger.info(f"Stored daily snapshot for {snapshot['date']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing snapshot: {e}")
    finally:
        conn.close()


def main():
    """Run the full daily data collection pipeline."""
    started_at = datetime.now(timezone.utc)
    today = date.today()
    steps = {}
    errors = {}

    logger.info("=" * 60)
    logger.info(f"DAILY LOOP STARTED — {today} ({today.strftime('%A')})")
    logger.info("=" * 60)

    # Step 0: Ensure DB schema exists
    try:
        init_schema()
        steps["schema_init"] = True
    except Exception as e:
        logger.error(f"Schema init failed: {e}")
        errors["schema_init"] = str(e)
        # Continue — schema might already exist

    # Step 1: OANDA Bars
    logger.info("─── Step 1: Fetching OANDA bars ───")
    try:
        bars_result = fetch_bars()
        steps["oanda_bars"] = True
        logger.info(f"Bars result: {bars_result}")
    except Exception as e:
        logger.error(f"OANDA bars failed: {e}")
        bars_result = {"error": str(e)}
        errors["oanda_bars"] = str(e)

    # Step 2: FRED Yields
    logger.info("─── Step 2: Fetching FRED yields ───")
    try:
        yields_result = fetch_yields()
        steps["fred_yields"] = True
        logger.info(f"Yields result: {yields_result}")
    except Exception as e:
        logger.error(f"FRED yields failed: {e}")
        yields_result = {"error": str(e)}
        errors["fred_yields"] = str(e)

    # Steps 3, 4 & 6 (sentiment, swap rates, Perplexity AI) REMOVED:
    # intraday CME futures configuration — sentiment analysis and FX carry are
    # out of the feature set and signal stack (see features/vector.py,
    # signals/tier1.py). Kronos foundation-model features replace them.
    sentiment_result = {"removed": True}
    swaps_result = {"removed": True}
    ai_result = {"removed": True}

    # Step 5: Economic Calendar
    logger.info("─── Step 5: Fetching calendar ───")
    try:
        calendar_result = fetch_calendar()
        steps["calendar"] = True
    except Exception as e:
        logger.error(f"Calendar failed: {e}")
        calendar_result = {"error": str(e)}
        errors["calendar"] = str(e)

    # ═══════════════════════════════════════════════════════
    # PHASE 2: PREDICTION ENGINE
    # ═══════════════════════════════════════════════════════

    regime_result = {}
    technical_result = {}
    tier1_signals = []
    tier2_signals = []
    composite_result = {}
    feature_vector = {}
    prediction_result = {}

    # Step 7: Compute Technical Indicators
    logger.info("─── Step 7: Computing technical indicators ───")
    try:
        from models.database import fetch_all as db_fetch_all
        import pandas as pd

        bars_rows = db_fetch_all(
            """SELECT bar_time, open, high, low, close, volume
               FROM bars WHERE instrument = %s AND granularity = 'D' AND complete = TRUE
               ORDER BY bar_time ASC""",
            (PRIMARY_INSTRUMENT,),
        )
        h4_rows = db_fetch_all(
            """SELECT bar_time, open, high, low, close, volume
               FROM bars WHERE instrument = %s AND granularity = 'H4' AND complete = TRUE
               ORDER BY bar_time ASC""",
            (PRIMARY_INSTRUMENT,),
        )
        if bars_rows:
            for r in bars_rows:
                for col in ['open', 'high', 'low', 'close']:
                    r[col] = float(r[col])
                r['volume'] = int(r['volume'])
            bars_df = pd.DataFrame(bars_rows)
            h4_df = None
            if h4_rows:
                for r in h4_rows:
                    for col in ['open', 'high', 'low', 'close']:
                        r[col] = float(r[col])
                    r['volume'] = int(r['volume'])
                h4_df = pd.DataFrame(h4_rows)
            technical_result = compute_all_features(bars_df, h4_df)
            steps["technical"] = True
            logger.info(f"Technical: ATR={technical_result.get('atr_14')}, RSI={technical_result.get('rsi_14')}")
        else:
            logger.warning("No daily bars in DB for technical indicators")
    except Exception as e:
        logger.error(f"Technical indicators failed: {e}")
        errors["technical"] = str(e)

    # Step 8: HMM Regime Detection
    logger.info("─── Step 8: HMM regime detection ───")
    try:
        detector = RegimeDetector()
        # Try to load existing model; fit if not available
        try:
            detector._load_model()
            if detector.model is None:
                raise FileNotFoundError("No model file")
        except Exception:
            logger.info("No HMM model found — fitting on available data...")
            detector.fit(PRIMARY_INSTRUMENT)

        regime_result = detector.predict_regime(PRIMARY_INSTRUMENT)
        detector.store_regime(today, PRIMARY_INSTRUMENT, regime_result)
        steps["regime"] = True
        logger.info(f"Regime: {regime_result.get('state_label')} (conf={regime_result.get('confidence')})")
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        regime_result = {"state_id": 1, "state_label": "high_vol_choppy", "confidence": 0.33, "days_in_regime": 0}
        errors["regime"] = str(e)

    # Step 8b: Load Kronos foundation-model features from the md schema
    # (populated by forecasting/run.py --persist). One point-in-time read,
    # threaded into both the Tier-1 directional signal and the feature vector.
    logger.info("─── Step 8b: Loading Kronos bridge features ───")
    from features.kronos_bridge import load_latest_kronos
    from config.settings import KRONOS_INSTRUMENT
    kronos_features = load_latest_kronos(KRONOS_INSTRUMENT)
    steps["kronos_bridge"] = kronos_features is not None

    # Step 9: Generate Tier 1 + Tier 2 Signals
    logger.info("─── Step 9: Generating signals ───")
    try:
        tier1_signals = generate_all_tier1(
            today, PRIMARY_INSTRUMENT, regime_result.get("state_id", 1),
            technical_result, kronos=kronos_features
        )

        # Determine proposed direction from Tier 1 for Tier 2 confirmation
        composite_result = compute_composite(tier1_signals, [])
        proposed_dir = composite_result["composite_direction"]

        tier2_signals = generate_all_tier2(
            today, PRIMARY_INSTRUMENT, technical_result, proposed_dir
        )

        # Recompute composite with Tier 2
        composite_result = compute_composite(tier1_signals, tier2_signals)

        # Store all signals
        store_signals(today, PRIMARY_INSTRUMENT, tier1_signals, tier2_signals)
        steps["signals"] = True
        logger.info(f"Composite: {composite_result['composite_direction']} (strength={composite_result['composite_strength']})")
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        errors["signals"] = str(e)

    # Step 10: Assemble Feature Vector
    logger.info("─── Step 10: Assembling feature vector ───")
    try:
        feature_vector = assemble_feature_vector(
            today, PRIMARY_INSTRUMENT, technical_result, regime_result,
            composite_result, kronos=kronos_features
        )
        store_feature_vector(today, PRIMARY_INSTRUMENT, feature_vector)
        steps["feature_vector"] = True
        logger.info(f"Feature vector: {len(feature_vector)} features assembled")
    except Exception as e:
        logger.error(f"Feature vector assembly failed: {e}")
        errors["feature_vector"] = str(e)

    # Step 11: Meta-Model Prediction
    logger.info("─── Step 11: Meta-model prediction ───")
    try:
        meta = MetaModel()
        prediction_result = meta.predict(feature_vector)
        meta.store_prediction(
            today, PRIMARY_INSTRUMENT, prediction_result,
            regime_result.get("state_id", 1),
            composite_result.get("composite_strength", 0.0),
        )
        steps["prediction"] = True
        logger.info(
            f"Prediction: {prediction_result['direction']} "
            f"(prob={prediction_result['probability']}, size={prediction_result['size_multiplier']}x)"
        )
    except Exception as e:
        logger.error(f"Meta-model prediction failed: {e}")
        errors["prediction"] = str(e)

    # Step 11b: [NEW] Push regime and predictions to Redis
    logger.info("─── Step 11b: Pushing predictions and regime to Redis ───")
    try:
        import redis
        redis_host = os.environ.get("REDIS_HOST", os.environ.get("DB_HOST", "localhost"))
        r = redis.Redis(host=redis_host, port=6379, db=0)
        
        if "regime_result" in locals() and regime_result:
            r.set(f"{PRIMARY_INSTRUMENT}:regime", json.dumps(regime_result))
            
        if "prediction_result" in locals() and prediction_result:
            r.set(f"{PRIMARY_INSTRUMENT}:metamodel", json.dumps(prediction_result))
            
        if "technical_result" in locals() and technical_result:
            atr_val = technical_result.get("atr_14")
            if atr_val:
                r.set(f"{PRIMARY_INSTRUMENT}:daily_atr", float(atr_val))
                
        logger.info("Successfully pushed daily models and ATR metadata to Redis.")
        steps["redis_push"] = True
    except Exception as e:
        logger.error(f"Failed to push daily data to Redis: {e}")
        errors["redis_push"] = str(e)

    # Step 11c: [NEW] Closed-Loop Event Extraction, Opportunity Measurement, and System Learning
    logger.info("─── Step 11c: Running Closed-Loop Architecture layers ───")
    try:
        # 1. Event Extraction (Layer 3)
        from events.extractor import EventExtractor
        extractor = EventExtractor()
        today_event = extractor.extract_and_store(today, PRIMARY_INSTRUMENT, composite_result, prediction_result)
        
        # 2. Opportunity Measurement (Layer 4) for yesterday (T-1)
        # Find yesterday's event from the database
        from opportunity.measurement import OpportunityMeasurer
        from models.database import fetch_one
        
        # Get the date of the last non-flat event before today
        prev_event = fetch_one(
            """
            SELECT id, date, direction
            FROM events
            WHERE instrument = %s AND date < %s
            ORDER BY date DESC LIMIT 1
            """,
            (PRIMARY_INSTRUMENT, today)
        )
        
        if prev_event:
            measurer = OpportunityMeasurer()
            # Measure how the event played out today (trade date = today)
            measurer.measure_and_store(
                event_id=prev_event["id"],
                event_date=prev_event["date"],
                trade_date=today,
                instrument=PRIMARY_INSTRUMENT,
                direction=prev_event["direction"]
            )
        else:
            logger.info("Closed-Loop: No previous trade event found to measure.")
            
        # 3. System Learning Feedback (Layer 8)
        from learning.feedback import ClosedLoopLearner
        learner = ClosedLoopLearner(lookback_days=20)
        learner.run_feedback_cycle(today, PRIMARY_INSTRUMENT)
        
        logger.info("Closed-Loop: Completed layers 3, 4, and 8 cycle successfully.")
        steps["closed_loop"] = True
    except Exception as e:
        logger.error(f"Closed-Loop: Failed in daily execution: {e}", exc_info=True)
        errors["closed_loop"] = str(e)

    # ═══════════════════════════════════════════════════════
    # FINALIZE
    # ═══════════════════════════════════════════════════════

    # Step 12: Assemble Snapshot
    logger.info("─── Step 12: Assembling daily snapshot ───")
    try:
        snapshot = assemble_daily_snapshot(
            bars_result, yields_result, sentiment_result,
            swaps_result, calendar_result, ai_result,
        )
        # Enrich snapshot with Phase 2 data
        snapshot["regime"] = regime_result
        snapshot["signals"] = {
            "tier1": [{"detector": s["detector"], "direction": s["direction"], "strength": s["strength"]} for s in tier1_signals],
            "tier2": [{"detector": s["detector"], "confirmed": s.get("confirmed", False)} for s in tier2_signals],
            "composite": composite_result,
        }
        snapshot["prediction"] = prediction_result
        snapshot["technical"] = {k: v for k, v in technical_result.items() if not k.startswith("is_")}
        store_snapshot(snapshot)
        steps["snapshot"] = True
    except Exception as e:
        logger.error(f"Snapshot assembly failed: {e}")
        errors["snapshot"] = str(e)

    # Determine overall status
    if not errors:
        status = "success"
    elif len(errors) < 4:
        status = "partial"
    else:
        status = "failed"

    # Log the run
    log_pipeline_run(today, started_at, status, steps, errors)

    # Step 13: Send Discord notification
    logger.info("─── Step 13: Sending Discord notification ───")
    try:
        if 'snapshot' in locals() and snapshot:
            sent = send_signal(snapshot, status)
        else:
            sent = send_error_alert(f"Pipeline {status}: {json.dumps(errors)}")
        if sent:
            steps["discord_notify"] = True
        else:
            logger.warning("Discord notification skipped (no webhook configured or send failed)")
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
        errors["discord_notify"] = str(e)

    logger.info("=" * 60)
    logger.info(f"DAILY LOOP COMPLETED — Status: {status.upper()}")
    logger.info(f"Steps OK: {list(steps.keys())}")
    if errors:
        logger.warning(f"Errors: {list(errors.keys())}")
    logger.info("=" * 60)

    return status


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status in ("success", "partial") else 1)
