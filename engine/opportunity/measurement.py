import logging
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
from models.database import get_connection, fetch_all, fetch_one

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")

class OpportunityMeasurer:
    """
    Layer 4: Opportunity Measurement (retrospective analysis).
    Quantifies what the market offered (MFE, MAE, optimal return) for each event.
    """
    def __init__(self):
        pass

    def get_pip_value(self, instrument: str) -> float:
        """Return pip multiplier based on instrument name."""
        return 0.01 if "JPY" in instrument else 0.0001

    def measure_and_store(self, event_id: int, event_date: date, trade_date: date, instrument: str, direction: str) -> dict | None:
        """
        Retrospectively analyze the session price path for a trading opportunity event,
        and store the measurements in PostgreSQL.
        
        Args:
            event_id: Database ID of the event
            event_date: Date of signal/event (T)
            trade_date: Execution/trade date (T+1)
            instrument: Currency pair
            direction: Signal direction ('long' or 'short')
            
        Returns:
            Dict representing the stored opportunity measurement, or None if failed.
        """
        pip_val = self.get_pip_value(instrument)
        
        # 1. Fetch H4 candles representing Day T+1 (5 PM ET T to 5 PM ET T+1)
        # OANDA daily session rolls at 5 PM NY time.
        start_dt = datetime.combine(event_date, time(17, 0), tzinfo=NY_TZ)
        end_dt = datetime.combine(trade_date, time(17, 0), tzinfo=NY_TZ)
        
        h4_rows = fetch_all(
            """
            SELECT bar_time, open, high, low, close
            FROM bars
            WHERE instrument = %s
              AND granularity = 'H4'
              AND complete = TRUE
              AND bar_time > %s
              AND bar_time <= %s
            ORDER BY bar_time ASC
            """,
            (instrument, start_dt, end_dt)
        )
        
        entry_price = None
        exit_price = None
        highs = []
        lows = []
        
        if h4_rows:
            # We have H4 candles for the intraday path
            entry_price = float(h4_rows[0]["open"])
            exit_price = float(h4_rows[-1]["close"])
            highs = [float(r["high"]) for r in h4_rows]
            lows = [float(r["low"]) for r in h4_rows]
            logger.info(f"OpportunityMeasurer: Found {len(h4_rows)} H4 bars for trade date {trade_date}")
        else:
            # Fall back to daily bars
            daily_row = fetch_one(
                """
                SELECT open, high, low, close
                FROM bars
                WHERE instrument = %s
                  AND granularity = 'D'
                  AND complete = TRUE
                  AND bar_time::date = %s
                """,
                (instrument, trade_date)
            )
            if daily_row:
                entry_price = float(daily_row["open"])
                exit_price = float(daily_row["close"])
                highs = [float(daily_row["high"])]
                lows = [float(daily_row["low"])]
                logger.info(f"OpportunityMeasurer: H4 bars missing. Fell back to daily bar for {trade_date}")
            else:
                logger.warning(f"OpportunityMeasurer: No daily or H4 bar data available for {trade_date} to measure opportunity.")
                return None
                
        # 2. Calculate excursions
        max_high = max(highs)
        min_low = min(lows)
        
        if direction == "long":
            mfe_pips = (max_high - entry_price) / pip_val
            mae_pips = (min_low - entry_price) / pip_val
            optimal_return_pips = mfe_pips
            close_return_pips = (exit_price - entry_price) / pip_val
        elif direction == "short":
            mfe_pips = (entry_price - min_low) / pip_val
            mae_pips = (entry_price - max_high) / pip_val
            optimal_return_pips = mfe_pips
            close_return_pips = (entry_price - exit_price) / pip_val
        else:
            # Flat
            mfe_pips = 0.0
            mae_pips = 0.0
            optimal_return_pips = 0.0
            close_return_pips = 0.0
            
        # Ensure correct sign constraints
        mfe_pips = max(0.0, mfe_pips)
        mae_pips = min(0.0, mae_pips)
        optimal_return_pips = max(0.0, optimal_return_pips)
        
        # Path ratio: MFE / (MFE + |MAE|)
        abs_mae = abs(mae_pips)
        if (mfe_pips + abs_mae) > 0:
            path_ratio = mfe_pips / (mfe_pips + abs_mae)
        else:
            path_ratio = 0.5
            
        # 3. Store in Postgres
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO opportunity_measurements (
                        event_id, date, trade_date, instrument, direction,
                        entry_price, exit_price, mfe_pips, mae_pips,
                        optimal_return_pips, close_return_pips, path_ratio
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO UPDATE SET
                        entry_price = EXCLUDED.entry_price,
                        exit_price = EXCLUDED.exit_price,
                        mfe_pips = EXCLUDED.mfe_pips,
                        mae_pips = EXCLUDED.mae_pips,
                        optimal_return_pips = EXCLUDED.optimal_return_pips,
                        close_return_pips = EXCLUDED.close_return_pips,
                        path_ratio = EXCLUDED.path_ratio,
                        created_at = NOW()
                """, (
                    event_id, event_date, trade_date, instrument, direction,
                    entry_price, exit_price, mfe_pips, mae_pips,
                    optimal_return_pips, close_return_pips, path_ratio
                ))
            conn.commit()
            logger.info(
                f"OpportunityMeasurer: Stored measurements for event {event_id} on {trade_date}. "
                f"MFE={mfe_pips:.1f} pips | MAE={mae_pips:.1f} pips | Close PnL={close_return_pips:.1f} pips"
            )
        except Exception as e:
            conn.rollback()
            logger.error(f"OpportunityMeasurer: Failed to store measurements: {e}")
            raise
        finally:
            conn.close()
            
        return {
            "event_id": event_id,
            "date": event_date,
            "trade_date": trade_date,
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "mfe_pips": mfe_pips,
            "mae_pips": mae_pips,
            "optimal_return_pips": optimal_return_pips,
            "close_return_pips": close_return_pips,
            "path_ratio": path_ratio
        }
