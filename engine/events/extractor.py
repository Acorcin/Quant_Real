import logging
import json
from datetime import date
from models.database import get_connection

logger = logging.getLogger(__name__)

class EventExtractor:
    """
    Layer 3: Event Extraction.
    Converts continuous data/signals into discrete trading opportunity events.
    """
    def __init__(self):
        pass

    def extract_and_store(self, run_date: date, instrument: str, composite_result: dict, prediction_result: dict) -> dict | None:
        """
        Extract a discrete trading opportunity event from the composite signal and prediction,
        and write it to the PostgreSQL database.
        
        Args:
            run_date: Date of signal generation (T)
            instrument: Target currency pair
            composite_result: Composite Tier 1/2 voting output
            prediction_result: XGBoost meta-model prediction output
            
        Returns:
            Dict representing the stored event, or None if the direction is flat.
        """
        direction = prediction_result.get("direction", "flat")
        if direction == "flat":
            # If the prediction recommends going flat, check composite as fallback, 
            # but usually prediction direction dictates the final decision.
            direction = composite_result.get("composite_direction", "flat")
            
        if direction == "flat":
            logger.info(f"EventExtractor: No active trading opportunity on {run_date} for {instrument}.")
            return None
            
        magnitude = float(composite_result.get("composite_strength", 0.0))
        confidence = float(prediction_result.get("probability", 0.50))
        
        metadata = {
            "signal_count": composite_result.get("signal_count", 0),
            "tier2_count": composite_result.get("tier2_count", 0),
            "model_version": prediction_result.get("model_version", "unknown"),
            "size_multiplier": prediction_result.get("size_multiplier", 0.0),
            "eod_event_reversal": composite_result.get("eod_event_reversal", 0),
            "event_surprise_magnitude": composite_result.get("event_surprise_magnitude", 0.0)
        }
        
        event_type = "composite_signal"
        
        conn = get_connection()
        event_id = None
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO events (date, instrument, event_type, direction, magnitude, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, instrument, event_type) DO UPDATE SET
                        direction = EXCLUDED.direction,
                        magnitude = EXCLUDED.magnitude,
                        confidence = EXCLUDED.confidence,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW()
                    RETURNING id
                """, (
                    run_date, instrument, event_type, direction, magnitude, confidence, json.dumps(metadata)
                ))
                row = cur.fetchone()
                if row:
                    event_id = row[0]
            conn.commit()
            logger.info(f"EventExtractor: Stored event {event_id} for {run_date} {instrument} ({direction.upper()})")
        except Exception as e:
            conn.rollback()
            logger.error(f"EventExtractor: Failed to save event: {e}")
            raise
        finally:
            conn.close()
            
        return {
            "id": event_id,
            "date": run_date,
            "instrument": instrument,
            "event_type": event_type,
            "direction": direction,
            "magnitude": magnitude,
            "confidence": confidence,
            "metadata": metadata
        }
