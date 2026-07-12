"""
Live Trading and Execution Engine.

Consumes conditioned ticks from Redis stream `cond_ticks:{instrument}`,
monitors Kalman velocity signals and daily meta-model outputs,
coordinates risk management and position sizing, and executes trades
via the OANDA Order Manager.
"""
import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timezone
import redis
from config.settings import PRIMARY_INSTRUMENT
from execution.order_manager import OrderManager
from execution.risk import RiskManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("live_trader")

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", os.environ.get("DB_HOST", "localhost"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

# Global states
running = True
redis_client = None
order_manager = None
risk_manager = None
active_trade = None

def handle_signal(signum, frame):
    global running
    logger.info(f"Signal {signum} received. Exiting execution loop...")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def load_daily_signals(redis_client, instrument: str) -> tuple[str, float, str, int, str]:
    """
    Load the daily HMM regime and XGBoost prediction from Redis.
    
    Returns:
        Tuple of (direction, probability, regime_label, regime_state, model_version)
    """
    # 1. XGBoost Prediction
    model_data = redis_client.get(f"{instrument}:metamodel")
    direction = "flat"
    probability = 0.50
    model_ver = "default"
    if model_data:
        try:
            m = json.loads(model_data)
            direction = m.get("direction", "flat")
            probability = m.get("probability", 0.50)
            model_ver = m.get("model_version", "xgb_default")
        except Exception as e:
            logger.error(f"Error parsing daily metamodel from Redis: {e}")
            
    # 2. HMM Regime
    regime_data = redis_client.get(f"{instrument}:regime")
    regime_label = "high_vol_choppy"
    regime_state = 1
    if regime_data:
        try:
            r = json.loads(regime_data)
            regime_label = r.get("state_label", "high_vol_choppy")
            regime_state = r.get("state_id", 1)
        except Exception as e:
            logger.error(f"Error parsing daily regime from Redis: {e}")
            
    return direction, probability, regime_label, regime_state, model_ver

def sync_active_trade_from_redis(redis_client, instrument: str) -> dict | None:
    """Load active trade dictionary from Redis."""
    trade_key = f"trade_state:{instrument}"
    data = redis_client.get(trade_key)
    if data:
        try:
            return json.loads(data)
        except Exception:
            pass
    return None

def save_active_trade_to_redis(redis_client, instrument: str, trade_dict: dict):
    """Save active trade dictionary to Redis."""
    trade_key = f"trade_state:{instrument}"
    redis_client.set(trade_key, json.dumps(trade_dict))

def delete_active_trade_from_redis(redis_client, instrument: str):
    """Remove active trade dictionary from Redis."""
    trade_key = f"trade_state:{instrument}"
    redis_client.delete(trade_key)

def run_trader(instrument: str):
    global running, redis_client, order_manager, risk_manager, active_trade
    
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    order_manager = OrderManager()
    risk_manager = RiskManager()
    
    # Sync active trade state from Redis
    active_trade = sync_active_trade_from_redis(redis_client, instrument)
    if active_trade:
        logger.info(f"Synchronized active trade from Redis: {active_trade}")
    else:
        logger.info("No active trade found in state.")
        
    # Get last processed conditioned tick stream ID
    id_key = f"trader_last_processed_id:{instrument}"
    last_id = redis_client.get(id_key) or "$"
    logger.info(f"Starting execution processing from ID: {last_id}")
    
    cond_stream = f"cond_ticks:{instrument}"
    trade_state_key = f"trade_state:{instrument}"
    
    # Micro-signal entry/exit thresholds
    # Velocity is change rate of Kalman price per second.
    # EUR/USD daily price is ~1.08. 1 pip = 0.0001. 
    # Entry threshold: 1.5e-6 is roughly 0.015 pips/sec (representing clear directional movement).
    velocity_entry_threshold = 1.5e-6
    velocity_exit_threshold = 0.0 # Exit when momentum crosses zero
    
    sl_pips = 20.0  # Stop Loss pips
    tp_pips = 40.0  # Take Profit pips
    
    # Metadata cache
    last_meta_refresh = 0.0
    daily_direction = "flat"
    daily_probability = 0.50
    regime_label = "high_vol_choppy"
    regime_state = 1
    model_version = "default"
    
    while running:
        # 1. Periodically refresh daily model outputs and closed-loop parameters (every 60s)
        now = time.time()
        if now - last_meta_refresh > 60.0:
            daily_direction, daily_probability, regime_label, regime_state, model_version = load_daily_signals(redis_client, instrument)
            
            # Load learned parameters from Layer 8
            lp_data = redis_client.get(f"{instrument}:learning_params")
            if lp_data:
                try:
                    lp = json.loads(lp_data)
                    velocity_entry_threshold = float(lp.get("velocity_entry_threshold", 1.5e-6))
                    velocity_exit_threshold = float(lp.get("velocity_exit_threshold", 0.0))
                    
                    # Update risk manager dynamic attributes
                    risk_manager.kelly_fraction = float(lp.get("kelly_fraction", 0.15))
                    risk_manager.max_spread_pips = float(lp.get("max_spread_pips", 2.0))
                    risk_manager.probability_threshold_half = float(lp.get("probability_threshold_half", 0.55))
                    risk_manager.probability_threshold_full = float(lp.get("probability_threshold_full", 0.70))
                    
                    logger.info(
                        f"Closed-Loop: Applied adapted parameters -> "
                        f"VelEntry={velocity_entry_threshold:.8f} | Kelly={risk_manager.kelly_fraction:.3f} | "
                        f"MaxSpread={risk_manager.max_spread_pips:.2f} pips | Gates={risk_manager.probability_threshold_half:.2f}/{risk_manager.probability_threshold_full:.2f}"
                    )
                except Exception as e:
                    logger.error(f"Error parsing learning_params from Redis: {e}")
            
            last_meta_refresh = now
            logger.info(
                f"Daily Models: Dir={daily_direction} ({daily_probability:.2f}) | "
                f"Regime={regime_label} | Model={model_version}"
            )
            
        try:
            # Consume from conditioned stream
            streams = redis_client.xread({cond_stream: last_id}, count=100, block=1000)
            
            if not streams:
                # If there are no ticks, check position status from broker to keep sync
                if active_trade and not active_trade["ticket_id"].startswith("dry_run"):
                    ticket_id = active_trade["ticket_id"]
                    if not order_manager.is_trade_still_open(ticket_id):
                        logger.info(f"Active trade {ticket_id} closed broker-side. Syncing...")
                        exit_price, exit_time, exit_reason = order_manager.get_closed_trade_details(ticket_id)
                        order_manager._log_exit_to_db(ticket_id, exit_time, exit_price, exit_reason)
                        delete_active_trade_from_redis(redis_client, instrument)
                        active_trade = None
                continue
                
            for stream_name, messages in streams:
                for msg_id, payload in messages:
                    # Parse conditioned tick
                    raw_mid = float(payload["raw_mid"])
                    kalman_price = float(payload["kalman_price"])
                    kalman_velocity = float(payload["kalman_velocity"])
                    tick_time_str = payload["time"]
                    
                    # Inside OANDA pricing context, bid-ask spread is crucial
                    # Read the actual bid and ask from the payload if present (passed during simulated streaming)
                    # Otherwise, fall back to the mid approximation (1.4 pip spread)
                    bid = float(payload["bid"]) if "bid" in payload else (raw_mid - 0.00007)
                    ask = float(payload["ask"]) if "ask" in payload else (raw_mid + 0.00007)
                    
                    # ──────────────────────────────────────────────────────────
                    # A. ACTIVE POSITION MONITORING & EXITS
                    # ──────────────────────────────────────────────────────────
                    if active_trade:
                        ticket_id = active_trade["ticket_id"]
                        direction = active_trade["direction"]
                        
                        # 1. Check dry-run SL/TP hits
                        if ticket_id.startswith("dry_run"):
                            sl = active_trade["sl_price"]
                            tp = active_trade["tp_price"]
                            
                            hit_sl = False
                            hit_tp = False
                            
                            if direction == "long":
                                if bid <= sl:
                                    hit_sl = True
                                elif bid >= tp:
                                    hit_tp = True
                            else:
                                if ask >= sl:
                                    hit_sl = True
                                elif ask <= tp:
                                    hit_tp = True
                                    
                            if hit_sl:
                                logger.info(f"Dry-run trade hit SL at {sl:.5f}")
                                order_manager.close_trade(ticket_id, instrument, sl, "stop_loss")
                                delete_active_trade_from_redis(redis_client, instrument)
                                active_trade = None
                                last_id = msg_id
                                continue
                            elif hit_tp:
                                logger.info(f"Dry-run trade hit TP at {tp:.5f}")
                                order_manager.close_trade(ticket_id, instrument, tp, "take_profit")
                                delete_active_trade_from_redis(redis_client, instrument)
                                active_trade = None
                                last_id = msg_id
                                continue
                        else:
                            # Verify if broker-side closed (e.g. hit SL or TP on OANDA)
                            # To avoid hammering OANDA API on every tick (~100/sec), we query only once every 5 seconds or 200 ticks
                            # Let's count ticks or check timestamps
                            if int(msg_id.split("-")[0]) % 100 == 0:
                                if not order_manager.is_trade_still_open(ticket_id):
                                    logger.info(f"OANDA Trade {ticket_id} was closed on broker-side. Syncing exit...")
                                    exit_price, exit_time, exit_reason = order_manager.get_closed_trade_details(ticket_id)
                                    order_manager._log_exit_to_db(ticket_id, exit_time, exit_price, exit_reason)
                                    delete_active_trade_from_redis(redis_client, instrument)
                                    active_trade = None
                                    last_id = msg_id
                                    continue
                                    
                        # 2. Check Signal Reversal (Dynamic Exit)
                        exit_triggered = False
                        exit_reason = "signal_reversal"
                        
                        if direction == "long":
                            # Exit if macro trend is no longer long, or if Kalman velocity shows negative momentum
                            if daily_direction != "long":
                                exit_triggered = True
                                exit_reason = "macro_direction_change"
                            elif kalman_velocity < velocity_exit_threshold:
                                exit_triggered = True
                                exit_reason = "momentum_reversal"
                        elif direction == "short":
                            # Exit if macro trend is no longer short, or if Kalman velocity shows positive momentum
                            if daily_direction != "short":
                                exit_triggered = True
                                exit_reason = "macro_direction_change"
                            elif kalman_velocity > -velocity_exit_threshold:
                                exit_triggered = True
                                exit_reason = "momentum_reversal"
                                
                        if exit_triggered:
                            logger.info(f"Exit signal triggered for trade {ticket_id}: {exit_reason} (Vel: {kalman_velocity:.8f})")
                            exit_price = bid if direction == "long" else ask
                            order_manager.close_trade(ticket_id, instrument, exit_price, exit_reason)
                            delete_active_trade_from_redis(redis_client, instrument)
                            active_trade = None
                            
                    # ──────────────────────────────────────────────────────────
                    # B. NEW TRADE ENTRIES
                    # ──────────────────────────────────────────────────────────
                    else:
                        # 1. Validate macro trend direction (must be long or short, not flat)
                        if daily_direction in ("long", "short"):
                            entry_triggered = False
                            
                            # 2. Check signal entry condition using price velocity
                            if daily_direction == "long" and kalman_velocity > velocity_entry_threshold:
                                entry_triggered = True
                            elif daily_direction == "short" and kalman_velocity < -velocity_entry_threshold:
                                entry_triggered = True
                                
                            if entry_triggered:
                                logger.info(f"Entry signal triggered: Direction={daily_direction} | Velocity={kalman_velocity:.8f}")
                                
                                # 3. Risk Checks (Spread, Daily Drawdown)
                                balance = order_manager.get_account_balance()
                                
                                if not risk_manager.validate_spread(bid, ask):
                                    last_id = msg_id
                                    continue
                                    
                                if not risk_manager.check_daily_drawdown(balance):
                                    last_id = msg_id
                                    # Halt trading
                                    continue
                                    
                                # 4. Sizing Calculation (Kelly + HMM regime)
                                units = risk_manager.calculate_position_size(
                                    probability=daily_probability,
                                    balance=balance,
                                    regime_label=regime_label
                                )
                                
                                if units > 0:
                                    entry_price = ask if daily_direction == "long" else bid
                                    logger.info(f"Executing {daily_direction.upper()} order for {units} units...")
                                    
                                    ticket_id = order_manager.execute_market_order(
                                        instrument=instrument,
                                        direction=daily_direction,
                                        units=units,
                                        current_price=entry_price,
                                        sl_pips=sl_pips,
                                        tp_pips=tp_pips,
                                        regime_state=regime_state,
                                        model_version=model_version
                                    )
                                    
                                    if ticket_id:
                                        # Calculate exact TP/SL levels for tracking
                                        pip_val = 0.0001
                                        if daily_direction == "long":
                                            sl_price = round(entry_price - (sl_pips * pip_val), 5)
                                            tp_price = round(entry_price + (tp_pips * pip_val), 5)
                                        else:
                                            sl_price = round(entry_price + (sl_pips * pip_val), 5)
                                            tp_price = round(entry_price - (tp_pips * pip_val), 5)
                                            
                                        active_trade = {
                                            "ticket_id": ticket_id,
                                            "direction": daily_direction,
                                            "units": units,
                                            "entry_price": entry_price,
                                            "sl_price": sl_price,
                                            "tp_price": tp_price,
                                            "sl_pips": sl_pips,
                                            "tp_pips": tp_pips
                                        }
                                        save_active_trade_to_redis(redis_client, instrument, active_trade)
                                
                    # Record stream position
                    last_id = msg_id
                    
            # Persist processed offset ID
            redis_client.set(id_key, last_id)
            
        except redis.ConnectionError:
            logger.error("Lost connection to Redis. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
            time.sleep(1)
            
    logger.info("Trader stopped.")

if __name__ == "__main__":
    run_trader(PRIMARY_INSTRUMENT)
