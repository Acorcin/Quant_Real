"""
Fetcher: OANDA V20 — Real-time pricing stream.

Streams live tick data (bid/ask quotes) from OANDA and pushes them
into the Redis stream `ticks:{instrument}`.
"""
import os
import json
import time
import logging
import requests
import redis
from datetime import datetime, timezone
from config.settings import OANDA_API_TOKEN, OANDA_ACCOUNT_ID, OANDA_BASE_URL, PRIMARY_INSTRUMENT

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("oanda_stream")

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", os.environ.get("DB_HOST", "localhost"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

def stream_ticks(instrument: str):
    """
    Establish pricing stream from OANDA and push ticks to Redis Stream.
    """
    # Build stream URL by replacing api- with stream-
    stream_base = OANDA_BASE_URL.replace("api-", "stream-")
    url = f"{stream_base}/v3/accounts/{OANDA_ACCOUNT_ID}/pricing/stream"
    
    headers = {
        "Authorization": f"Bearer {OANDA_API_TOKEN}",
        "Content-Type": "application/json",
    }
    
    params = {
        "instruments": instrument,
    }
    
    redis_client = get_redis_client()
    redis_key = f"ticks:{instrument}"
    logger.info(f"Connecting to OANDA pricing stream for {instrument}...")
    logger.info(f"Redis target stream: {redis_key} (Host: {REDIS_HOST}:{REDIS_PORT})")
    
    backoff = 1.0
    while True:
        try:
            # Connect with streaming enabled
            response = requests.get(url, headers=headers, params=params, stream=True, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Failed to connect to stream: {response.status_code} - {response.text}")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
                continue
                
            # Reset backoff on successful connection
            backoff = 1.0
            logger.info("Stream connection established. Reading ticks...")
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                data = json.loads(line.decode("utf-8"))
                
                # Check message type
                msg_type = data.get("type")
                if msg_type == "HEARTBEAT":
                    logger.debug(f"Heartbeat received at {data.get('time')}")
                    # Push heartbeat timestamp to Redis to keep track of liveness
                    redis_client.set(f"liveness:{instrument}:streamer", data.get("time"))
                elif msg_type == "PRICE":
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    tick_time = data.get("time")
                    
                    if bids and asks:
                        bid = float(bids[0]["price"])
                        ask = float(asks[0]["price"])
                        
                        tick_payload = {
                            "bid": bid,
                            "ask": ask,
                            "time": tick_time,
                            "received_at": datetime.now(timezone.utc).isoformat()
                        }
                        
                        # Add tick to Redis Stream with maxlen to limit memory growth
                        # maxlen=50000 stores roughly 1-2 days of active ticks for EUR_USD
                        redis_client.xadd(redis_key, tick_payload, maxlen=50000)
                        logger.debug(f"Tick: {instrument} Bid={bid:.5f} Ask={ask:.5f} Time={tick_time}")
                else:
                    logger.warning(f"Unknown message type: {data}")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in pricing stream: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in stream loop: {e}")
            
        # Reconnect logic with exponential backoff
        logger.info(f"Reconnecting in {backoff:.1f} seconds...")
        time.sleep(backoff)
        backoff = min(backoff * 2.0, 60.0)

def run_synthetic_simulation(instrument: str):
    """
    Simulate tick stream with a random walk when OANDA credentials are not present.
    Allows end-to-end system validation locally.
    """
    redis_client = get_redis_client()
    redis_key = f"ticks:{instrument}"
    logger.info(f"Starting synthetic tick generator for {instrument}...")
    logger.info(f"Redis target stream: {redis_key} (Host: {REDIS_HOST}:{REDIS_PORT})")
    
    # Starting price based on instrument
    price = 1.08500 if "EUR" in instrument else 1.25000 if "GBP" in instrument else 155.00
    spread = 0.00015 if "EUR" in instrument or "GBP" in instrument else 0.015
    
    import random
    
    while True:
        try:
            # Random walk step
            change = random.normalvariate(0, 0.00005 if price < 2.0 else 0.005)
            price += change
            
            bid = round(price - spread / 2.0, 5)
            ask = round(price + spread / 2.0, 5)
            tick_time = datetime.now(timezone.utc).isoformat()
            
            tick_payload = {
                "bid": bid,
                "ask": ask,
                "time": tick_time,
                "received_at": datetime.now(timezone.utc).isoformat()
            }
            
            redis_client.xadd(redis_key, tick_payload, maxlen=50000)
            redis_client.set(f"liveness:{instrument}:streamer", tick_time)
            logger.info(f"[SIMULATED] Tick: {instrument} Bid={bid:.5f} Ask={ask:.5f}")
            
            # Sleep 0.2 to 1.5 seconds
            time.sleep(random.uniform(0.2, 1.5))
            
        except Exception as e:
            logger.error(f"Error in synthetic stream: {e}")
            time.sleep(1)

if __name__ == "__main__":
    if not OANDA_API_TOKEN or not OANDA_ACCOUNT_ID:
        logger.warning("OANDA credentials not set in environment. Running in SYNTHETIC TICK SIMULATION mode.")
        run_synthetic_simulation(PRIMARY_INSTRUMENT)
    else:
        stream_ticks(PRIMARY_INSTRUMENT)
