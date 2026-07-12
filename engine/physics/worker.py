"""
Physics Engine Worker Daemon.

Consumes raw ticks from Redis stream `ticks:{instrument}`,
runs them through the PhysicsEngine (filtering, Kalman, regime clipping),
and publishes conditioned ticks to Redis stream `cond_ticks:{instrument}`.
Persists filter state to Redis for crash resilience.
"""
import os
import json
import time
import signal
import sys
import logging
from datetime import datetime, timezone
import redis
import psycopg2
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, PRIMARY_INSTRUMENT
from physics.engine import PhysicsEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("physics_worker")

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", os.environ.get("DB_HOST", "localhost"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

# State variables
running = True
engine = None
redis_client = None

def handle_signal(signum, frame):
    global running
    logger.info(f"Signal {signum} received. Saving state and shutting down...")
    running = False

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def get_pg_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def parse_oanda_timestamp(ts_str: str) -> float:
    """Parse OANDA string timestamp to Unix epoch float."""
    try:
        clean_str = ts_str.replace("Z", "")
        if "." in clean_str:
            base, frac = clean_str.split(".")
            clean_str = f"{base}.{frac[:6]}"  # truncate to microsecond precision
        dt = datetime.fromisoformat(clean_str)
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()

def get_daily_atr(redis_client, instrument: str) -> float:
    """Get the daily ATR from Redis or Postgres fallback."""
    atr_key = f"{instrument}:daily_atr"
    atr_val = redis_client.get(atr_key)
    if atr_val:
        return float(atr_val)
        
    # Postgres fallback
    logger.info("Daily ATR not found in Redis, querying database...")
    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT features->>'atr_14' AS atr
                FROM feature_vectors
                WHERE instrument = %s
                ORDER BY date DESC LIMIT 1
            """, (instrument,))
            row = cur.fetchone()
            if row and row[0]:
                val = float(row[0])
                redis_client.set(atr_key, val)
                logger.info(f"Loaded ATR={val:.5f} from Postgres")
                conn.close()
                return val
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching ATR from database: {e}")
        
    # Standard EUR/USD ATR fallback
    logger.warning("Using fallback ATR of 0.0075")
    return 0.0075

def get_current_regime(redis_client, instrument: str) -> str:
    """Get current market regime label from Redis or Postgres fallback."""
    regime_key = f"{instrument}:regime"
    regime_data = redis_client.get(regime_key)
    if regime_data:
        try:
            return json.loads(regime_data).get("state_label", "high_vol_choppy")
        except Exception:
            pass
            
    # Postgres fallback
    logger.info("Regime not found in Redis, querying database...")
    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT state_label
                FROM regimes
                WHERE instrument = %s
                ORDER BY date DESC LIMIT 1
            """, (instrument,))
            row = cur.fetchone()
            if row and row[0]:
                val = row[0]
                # set in Redis as expected JSON structure
                redis_client.set(regime_key, json.dumps({"state_label": val}))
                logger.info(f"Loaded regime={val} from Postgres")
                conn.close()
                return val
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching regime from database: {e}")
        
    return "high_vol_choppy"

def run_worker(instrument: str):
    global engine, redis_client, running
    
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    
    # Initialize Physics Engine
    engine = PhysicsEngine()
    
    # Try restoring engine state from Redis
    state_key = f"kalman_state:{instrument}"
    saved_state = redis_client.get(state_key)
    if saved_state:
        try:
            state_dict = json.loads(saved_state)
            engine.set_state(state_dict)
            logger.info("Successfully restored Physics Engine state from Redis.")
        except Exception as e:
            logger.error(f"Failed to restore state: {e}. Starting fresh.")
    else:
        logger.info("No saved state found. Initializing fresh engine.")
        
    # Get last processed stream ID
    id_key = f"last_processed_id:{instrument}"
    last_id = redis_client.get(id_key) or "$"
    logger.info(f"Starting raw tick stream processing from ID: {last_id}")
    
    tick_stream = f"ticks:{instrument}"
    cond_stream = f"cond_ticks:{instrument}"
    
    # Cache regime and ATR, refresh every 60 seconds
    last_meta_refresh = 0.0
    regime = "high_vol_choppy"
    daily_atr = 0.0075
    
    while running:
        # Refresh metadata (regime, ATR) periodically
        now = time.time()
        if now - last_meta_refresh > 60.0:
            regime = get_current_regime(redis_client, instrument)
            daily_atr = get_daily_atr(redis_client, instrument)
            last_meta_refresh = now
            logger.info(f"Metadata refreshed: Regime={regime}, Daily ATR={daily_atr:.5f}")
            
        try:
            # Read from raw ticks stream
            # Block for up to 1000ms
            streams = redis_client.xread({tick_stream: last_id}, count=100, block=1000)
            
            if not streams:
                continue
                
            for stream_name, messages in streams:
                for msg_id, payload in messages:
                    # Extract fields
                    bid = float(payload["bid"])
                    ask = float(payload["ask"])
                    tick_time_str = payload["time"]
                    
                    ts = parse_oanda_timestamp(tick_time_str)
                    
                    # Process tick through Physics Engine
                    cond_data = engine.process_tick(bid, ask, ts, regime, daily_atr)
                    
                    # Prepare conditioned payload
                    cond_payload = {
                        "raw_mid": cond_data["raw_mid"],
                        "filtered_mid": cond_data["filtered_mid"],
                        "is_spike": 1 if cond_data["is_spike"] else 0,
                        "kalman_price": cond_data["kalman_price"],
                        "kalman_velocity": cond_data["kalman_velocity"],
                        "tick_return": cond_data["tick_return"],
                        "normalized_return": cond_data["normalized_return"],
                        "clipped_return": cond_data["clipped_return"],
                        "bid": bid,
                        "ask": ask,
                        "time": tick_time_str,
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Push to conditioned stream
                    redis_client.xadd(cond_stream, cond_payload, maxlen=50000)
                    
                    # Update processed ID
                    last_id = msg_id
                    
            # Persist worker state and ID periodically (every loop iteration where ticks were processed)
            redis_client.set(id_key, last_id)
            redis_client.set(state_key, json.dumps(engine.get_state()))
            
        except redis.ConnectionError:
            logger.error("Lost connection to Redis. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            time.sleep(1)
            
    # Save state one final time on exit
    if engine and redis_client:
        try:
            redis_client.set(state_key, json.dumps(engine.get_state()))
            redis_client.set(id_key, last_id)
            logger.info("Saved state and ID successfully on exit.")
        except Exception as e:
            logger.error(f"Failed to save state on exit: {e}")
            
    logger.info("Worker stopped.")

if __name__ == "__main__":
    run_worker(PRIMARY_INSTRUMENT)
