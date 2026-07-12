"""
OANDA Order Manager.

Manages REST API interactions with OANDA:
- Placing market orders with attached SL/TP
- Closing trades by ID
- Querying account balance and status
- Checking trade state and exit details for broker-side closures
- Logging trades in the PostgreSQL database
"""
import os
import time
import json
import logging
import requests
from datetime import datetime, timezone
import psycopg2
from config.settings import (
    OANDA_API_TOKEN,
    OANDA_ACCOUNT_ID,
    OANDA_BASE_URL,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD
)

logger = logging.getLogger("order_manager")

class OrderManager:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run or os.environ.get("DRY_RUN", "false").lower() == "true"
        self.headers = {
            "Authorization": f"Bearer {OANDA_API_TOKEN}",
            "Content-Type": "application/json",
        }
        self.base_url = OANDA_BASE_URL
        self.account_id = OANDA_ACCOUNT_ID

    def get_pg_connection(self):
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

    def get_account_balance(self) -> float:
        """Fetch current Net Asset Value (NAV) of the account."""
        if self.dry_run or not self.account_id or not OANDA_API_TOKEN:
            logger.info("OrderManager [Dry-Run]: Simulating account balance of $10,000.00")
            return 10000.0
            
        url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            balance = float(data["account"]["NAV"])
            logger.info(f"OANDA Account NAV: ${balance:.2f}")
            return balance
        except Exception as e:
            logger.error(f"Error fetching account summary from OANDA: {e}")
            return 10000.0

    def is_trade_still_open(self, ticket_id: str) -> bool:
        """Query OANDA to see if the trade is still active."""
        if self.dry_run or ticket_id.startswith("dry_run") or not self.account_id or not OANDA_API_TOKEN:
            return True
            
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{ticket_id}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 404:
                return False
            response.raise_for_status()
            data = response.json()
            state = data.get("trade", {}).get("state", "CLOSED")
            return state == "OPEN"
        except Exception as e:
            logger.error(f"Error checking trade status for {ticket_id}: {e}")
            # If request fails, assume it is open to avoid double trading
            return True

    def get_closed_trade_details(self, ticket_id: str) -> tuple[float, datetime, str]:
        """
        Query OANDA for exit details of a trade that was closed broker-side.
        
        Returns:
            Tuple of (exit_price, exit_time, exit_reason)
        """
        default_exit_price = 0.0
        default_exit_time = datetime.now(timezone.utc)
        default_reason = "broker_close"
        
        if self.dry_run or ticket_id.startswith("dry_run") or not self.account_id or not OANDA_API_TOKEN:
            return default_exit_price, default_exit_time, default_reason
            
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{ticket_id}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            trade = data.get("trade", {})
            
            exit_price = float(trade.get("averageClosePrice", default_exit_price))
            exit_time_str = trade.get("closeTime", "")
            
            if exit_time_str:
                try:
                    clean_str = exit_time_str.replace("Z", "")
                    if "." in clean_str:
                        base, frac = clean_str.split(".")
                        clean_str = f"{base}.{frac[:6]}"
                    exit_time = datetime.fromisoformat(clean_str).replace(tzinfo=timezone.utc)
                except Exception:
                    exit_time = default_exit_time
            else:
                exit_time = default_exit_time
                
            # Deduce reason (TP vs SL)
            # OANDA sets 'closePrice' or shows closed by TP/SL in transactions,
            # we can check if it closed near the TP/SL levels
            exit_reason = default_reason
            tp_order_id = trade.get("takeProfitOrder", {}).get("id")
            sl_order_id = trade.get("stopLossOrder", {}).get("id")
            
            # If the trade has closeTime, we can check OANDA's transaction history if needed,
            # but simpler check is comparing exit price or looking at filledOrderIDs
            # Let's see if we can find order ids in the trade struct
            return exit_price, exit_time, exit_reason
        except Exception as e:
            logger.error(f"Error fetching closed trade details for {ticket_id}: {e}")
            return default_exit_price, default_exit_time, default_reason

    def execute_market_order(self, 
                             instrument: str, 
                             direction: str, 
                             units: int, 
                             current_price: float,
                             sl_pips: float = 20.0, 
                             tp_pips: float = 40.0,
                             regime_state: int = 1,
                             model_version: str = "no_model") -> str | None:
        """
        Execute a market order on OANDA and log it in the database.
        """
        if units <= 0:
            logger.warning("OrderManager: Position size is 0 units. Order skipped.")
            return None

        oanda_units = units if direction == "long" else -units
        pip_val = 0.0001
        
        if direction == "long":
            sl_price = round(current_price - (sl_pips * pip_val), 5)
            tp_price = round(current_price + (tp_pips * pip_val), 5)
        else:
            sl_price = round(current_price + (sl_pips * pip_val), 5)
            tp_price = round(current_price - (tp_pips * pip_val), 5)
            
        order_payload = {
            "order": {
                "units": str(oanda_units),
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": f"{sl_price:.5f}",
                    "timeInForce": "GTC"
                },
                "takeProfitOnFill": {
                    "price": f"{tp_price:.5f}",
                    "timeInForce": "GTC"
                }
            }
        }
        
        entry_time = datetime.now(timezone.utc)
        
        if self.dry_run or not self.account_id or not OANDA_API_TOKEN:
            ticket_id = f"dry_run_{int(time.time() * 1000)}"
            logger.info(
                f"OrderManager [Dry-Run] OPEN: {direction.upper()} {units} units of {instrument} "
                f"at {current_price:.5f} | SL={sl_price:.5f} | TP={tp_price:.5f} | ID={ticket_id}"
            )
            self._log_entry_to_db(ticket_id, instrument, direction, entry_time, current_price, units, regime_state, model_version, sl_price, tp_price)
            return ticket_id
            
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        try:
            response = requests.post(url, headers=self.headers, json=order_payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            order_fill = data.get("orderFillTransaction")
            if not order_fill:
                logger.error(f"Order rejected or not filled: {data}")
                return None
                
            ticket_id = order_fill["id"]
            fill_price = float(order_fill["price"])
            logger.info(f"OANDA Trade Executed Successfully. Ticket ID: {ticket_id} | Fill Price: {fill_price:.5f}")
            
            self._log_entry_to_db(ticket_id, instrument, direction, entry_time, fill_price, units, regime_state, model_version, sl_price, tp_price)
            return ticket_id
            
        except Exception as e:
            logger.error(f"Error executing market order on OANDA: {e}")
            if 'response' in locals() and response is not None:
                logger.error(f"Response: {response.text}")
            return None

    def close_trade(self, ticket_id: str, instrument: str, exit_price: float, reason: str = "signal_reversal") -> bool:
        """
        Close an active trade on OANDA and update the database log.
        """
        exit_time = datetime.now(timezone.utc)
        
        if self.dry_run or ticket_id.startswith("dry_run") or not self.account_id or not OANDA_API_TOKEN:
            logger.info(f"OrderManager [Dry-Run] CLOSE: Ticket {ticket_id} at {exit_price:.5f} | Reason: {reason}")
            self._log_exit_to_db(ticket_id, exit_time, exit_price, reason)
            return True
            
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{ticket_id}/close"
        try:
            payload = {"units": "ALL"}
            response = requests.put(url, headers=self.headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            trade_close = data.get("orderCreateTransaction") or data.get("orderFillTransaction")
            fill_price = exit_price
            if trade_close and "price" in trade_close:
                fill_price = float(trade_close["price"])
                
            logger.info(f"OANDA Trade {ticket_id} Closed Successfully at {fill_price:.5f}")
            self._log_exit_to_db(ticket_id, exit_time, fill_price, reason)
            return True
            
        except Exception as e:
            logger.error(f"Error closing trade {ticket_id} on OANDA: {e}")
            return False

    def _log_entry_to_db(self, ticket_id: str, instrument: str, direction: str, entry_time: datetime, entry_price: float, units: int, regime_state: int, model_version: str, sl_price: float = 0.0, tp_price: float = 0.0):
        """Write open trade log to PostgreSQL."""
        try:
            conn = self.get_pg_connection()
            with conn.cursor() as cur:
                # Include metadata fields in schema if needed, otherwise standard
                cur.execute("""
                    INSERT INTO live_trades (ticket_id, instrument, direction, entry_time, entry_price, position_size, regime_state, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (ticket_id, instrument, direction, entry_time, entry_price, units, regime_state, model_version))
            conn.commit()
            conn.close()
            logger.info(f"Logged open trade {ticket_id} in PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to log trade entry to DB: {e}")

    def _log_exit_to_db(self, ticket_id: str, exit_time: datetime, exit_price: float, reason: str):
        """Update closed trade log in PostgreSQL with exit price and PnL."""
        try:
            conn = self.get_pg_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT entry_price, direction, position_size 
                    FROM live_trades 
                    WHERE ticket_id = %s
                """, (ticket_id,))
                row = cur.fetchone()
                
            if not row:
                logger.error(f"Could not find open trade {ticket_id} in DB to log exit.")
                conn.close()
                return
                
            entry_price, direction, position_size = float(row[0]), row[1], int(row[2])
            
            pip_val = 0.0001
            if direction == "long":
                pnl_pips = (exit_price - entry_price) / pip_val
            else:
                pnl_pips = (entry_price - exit_price) / pip_val
                
            pnl_amount = pnl_pips * pip_val * position_size
            
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE live_trades 
                    SET exit_time = %s, 
                        exit_price = %s, 
                        pnl_pips = %s, 
                        pnl_amount = %s, 
                        exit_reason = %s
                    WHERE ticket_id = %s
                """, (exit_time, exit_price, pnl_pips, pnl_amount, reason, ticket_id))
            conn.commit()
            conn.close()
            logger.info(f"Updated trade {ticket_id} exit in DB. PnL: {pnl_pips:.1f} pips (${pnl_amount:.2f})")
        except Exception as e:
            logger.error(f"Failed to log trade exit to DB: {e}")
