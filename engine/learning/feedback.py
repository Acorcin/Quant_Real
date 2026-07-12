import logging
import json
import os
import redis
from datetime import date
from models.database import get_connection, fetch_all

logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", os.environ.get("DB_HOST", "localhost"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

class ClosedLoopLearner:
    """
    Layer 8: System Learning.
    Compares expected opportunity (Layer 4) vs realized execution (Layer 7),
    calculates discrepancies, and updates parameters back into Decision Intelligence.
    """
    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        
    def run_feedback_cycle(self, run_date: date, instrument: str) -> dict:
        """
        Execute the learning feedback cycle on run_date.
        
        Loads historical events, opportunity measurements, and live trades;
        adjusts system parameters; stores them in Redis and PostgreSQL.
        """
        # Defaults
        kelly_fraction = 0.15
        prob_threshold_half = 0.55
        prob_threshold_full = 0.70
        max_spread_pips = 2.0
        velocity_entry_threshold = 1.5e-6
        velocity_exit_threshold = 0.0
        
        # 1. Fetch recent events & opportunity measurements
        opp_rows = fetch_all(
            """
            SELECT e.confidence, e.direction AS sig_dir, o.close_return_pips, o.entry_price AS expected_entry, o.exit_price AS expected_exit
            FROM opportunity_measurements o
            JOIN events e ON o.event_id = e.id
            WHERE o.instrument = %s AND o.date >= %s - INTERVAL '%s days'
            ORDER BY o.date DESC
            """,
            (instrument, run_date, self.lookback_days)
        )
        
        # 2. Fetch recent live trades
        trade_rows = fetch_all(
            """
            SELECT ticket_id, entry_time::date as t_date, direction, entry_price, exit_price, pnl_pips
            FROM live_trades
            WHERE instrument = %s AND entry_time >= %s - INTERVAL '%s days' AND exit_time IS NOT NULL
            ORDER BY entry_time DESC
            """,
            (instrument, run_date, self.lookback_days)
        )
        
        metrics = {
            "sample_size_signals": len(opp_rows),
            "sample_size_trades": len(trade_rows),
            "win_rate_signals": 0.50,
            "avg_confidence": 0.50,
            "calibration_drift": 0.0,
            "avg_entry_slippage_pips": 0.0,
            "avg_exit_slippage_pips": 0.0,
        }
        
        # 3. Calculate Model Calibration (win rate of signals vs. predicted confidence)
        if opp_rows:
            wins = 0
            total_conf = 0.0
            for row in opp_rows:
                ret = float(row["close_return_pips"])
                sig_dir = row["sig_dir"]
                conf = float(row["confidence"])
                
                total_conf += conf
                
                # A signal is a win if the realized return was positive in its direction
                if sig_dir == "long" and ret > 0:
                    wins += 1
                elif sig_dir == "short" and ret > 0: # Note: close_return_pips is already direction-adjusted in measurement
                    wins += 1
                    
            win_rate = wins / len(opp_rows)
            avg_conf = total_conf / len(opp_rows)
            calibration_drift = avg_conf - win_rate
            
            metrics["win_rate_signals"] = round(win_rate, 4)
            metrics["avg_confidence"] = round(avg_conf, 4)
            metrics["calibration_drift"] = round(calibration_drift, 4)
            
            # Adjust decision thresholds based on calibration drift
            # If drift is positive, the model is overconfident. We raise probability gates and lower Kelly fraction.
            if calibration_drift > 0.05:
                # Scale Kelly down
                kelly_fraction = max(0.05, round(0.15 - 0.5 * (calibration_drift - 0.05), 3))
                # Boost probability gates
                prob_threshold_half = min(0.65, round(0.55 + 0.5 * (calibration_drift - 0.05), 3))
                prob_threshold_full = min(0.80, round(0.70 + 0.5 * (calibration_drift - 0.05), 3))
                logger.info(
                    f"ClosedLoopLearner: Calibration drift detected ({calibration_drift:.2f}). "
                    f"Adjusting Sizing: Kelly={kelly_fraction} | Gates={prob_threshold_half}/{prob_threshold_full}"
                )
                
        # 4. Calculate Execution Slippage (actual vs expected prices)
        pip_val = 0.01 if "JPY" in instrument else 0.0001
        
        if trade_rows and opp_rows:
            # Map expected prices by date
            expected_prices = {}
            for row in opp_rows:
                # We need a date index
                # Join by matching trade execution dates
                pass
            
            # Since joining can be sparse, let's do a direct join in database for slippage to be robust
            slippage_rows = fetch_all(
                """
                SELECT abs(t.entry_price - o.entry_price) as entry_diff,
                       abs(t.exit_price - o.exit_price) as exit_diff
                FROM live_trades t
                JOIN opportunity_measurements o ON t.entry_time::date = o.trade_date AND t.instrument = o.instrument
                WHERE t.instrument = %s AND t.entry_time >= %s - INTERVAL '%s days' AND t.exit_time IS NOT NULL
                """,
                (instrument, run_date, self.lookback_days)
            )
            
            if slippage_rows:
                total_entry_slip = 0.0
                total_exit_slip = 0.0
                for row in slippage_rows:
                    total_entry_slip += float(row["entry_diff"]) / pip_val
                    total_exit_slip += float(row["exit_diff"]) / pip_val
                    
                avg_entry_slip = total_entry_slip / len(slippage_rows)
                avg_exit_slip = total_exit_slip / len(slippage_rows)
                
                metrics["avg_entry_slippage_pips"] = round(avg_entry_slip, 2)
                metrics["avg_exit_slippage_pips"] = round(avg_exit_slip, 2)
                
                # Adjust execution parameters based on slippage
                # If slippage is high, tighten max_spread_pips and raise velocity trigger
                if avg_entry_slip > 1.0:
                    max_spread_pips = max(1.0, round(2.0 - 0.5 * (avg_entry_slip - 1.0), 2))
                    # Make entry velocity threshold more selective
                    velocity_entry_threshold = 1.5e-6 * (1.0 + 0.25 * (avg_entry_slip - 1.0))
                    logger.info(
                        f"ClosedLoopLearner: Entry slippage is high ({avg_entry_slip:.2f} pips). "
                        f"Tightening execution constraints: MaxSpread={max_spread_pips} pips | VelocityEntry={velocity_entry_threshold:.8f}"
                    )
                    
        adjusted_params = {
            "kelly_fraction": kelly_fraction,
            "probability_threshold_half": prob_threshold_half,
            "probability_threshold_full": prob_threshold_full,
            "max_spread_pips": max_spread_pips,
            "velocity_entry_threshold": velocity_entry_threshold,
            "velocity_exit_threshold": velocity_exit_threshold,
        }
        
        # 5. Push adjusted parameters to Redis
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
            r.set(f"{instrument}:learning_params", json.dumps(adjusted_params))
            logger.info(f"ClosedLoopLearner: Successfully pushed adapted parameters to Redis.")
        except Exception as e:
            logger.error(f"ClosedLoopLearner: Failed to push to Redis: {e}")
            
        # 6. Log the feedback run in the database
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO learning_runs (date, instrument, metrics, adjusted_parameters)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (date, instrument) DO UPDATE SET
                        metrics = EXCLUDED.metrics,
                        adjusted_parameters = EXCLUDED.adjusted_parameters,
                        created_at = NOW()
                """, (
                    run_date, instrument, json.dumps(metrics), json.dumps(adjusted_params)
                ))
            conn.commit()
            logger.info(f"ClosedLoopLearner: Logged feedback run for {run_date} in PostgreSQL.")
        except Exception as e:
            conn.rollback()
            logger.error(f"ClosedLoopLearner: Failed to log learning run: {e}")
            raise
        finally:
            conn.close()
            
        return {
            "date": run_date,
            "instrument": instrument,
            "metrics": metrics,
            "adjusted_parameters": adjusted_params
        }
