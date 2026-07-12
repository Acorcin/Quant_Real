"""
Risk Management and Position Sizing.

Implements:
- Fractional Kelly Criterion position sizing
- Regime-conditional size limits
- Spread validation (slippage control)
- Daily drawdown limit guards
"""
import logging
from datetime import datetime, timezone
import psycopg2
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, 
                 max_spread_pips: float = 2.0, 
                 daily_drawdown_limit: float = 0.02, 
                 kelly_fraction: float = 0.15,
                 default_rr: float = 1.5):
        """
        Args:
            max_spread_pips: Do not trade if current spread in pips exceeds this.
            daily_drawdown_limit: Daily loss limit as fraction of account equity (e.g. 0.02 = 2%).
            kelly_fraction: Fraction of the standard Kelly size to allocate (e.g. 0.15).
            default_rr: Risk-Reward ratio assumed for Kelly (default 1.5).
        """
        self.max_spread_pips = max_spread_pips
        self.daily_drawdown_limit = daily_drawdown_limit
        self.kelly_fraction = kelly_fraction
        self.default_rr = default_rr
        self.probability_threshold_half = 0.55
        self.probability_threshold_full = 0.70

    def calculate_position_size(self, 
                                probability: float, 
                                balance: float, 
                                regime_label: str = "low_vol", 
                                leverage: float = 10.0) -> int:
        """
        Calculate trade units using fractional Kelly sizing and HMM regime scale factors.
        
        Args:
            probability: XGBoost meta-model success probability.
            balance: Account balance.
            regime_label: Active HMM regime label ('low_vol', 'high_vol_choppy', 'high_vol_crash').
            leverage: Execution leverage (default 10.0).
            
        Returns:
            Units to trade (always integer >= 0).
        """
        if probability < self.probability_threshold_half:  # Meta-model threshold
            logger.info(f"Sizing: Probability {probability:.3f} below threshold {self.probability_threshold_half:.2f}. Flat size.")
            return 0
            
        # Kelly formula: f* = (p * R - (1 - p)) / R
        p = probability
        q = 1.0 - p
        R = self.default_rr
        
        kelly_f = (p * R - q) / R
        
        if kelly_f <= 0:
            logger.info(f"Sizing: Kelly fraction {kelly_f:.3f} is non-positive. Flat size.")
            return 0
            
        # Apply fractional Kelly safety scaling
        allocated_fraction = kelly_f * self.kelly_fraction
        
        # Apply HMM Regime constraints
        if regime_label == "high_vol_choppy":
            # Halve allocation due to whip-saw risk
            allocated_fraction *= 0.5
            logger.info("Sizing: Regime high_vol_choppy detected. Sizing halved.")
        elif regime_label == "high_vol_crash":
            # Hard stop on new trades
            logger.warning("Sizing: Regime high_vol_crash detected. Blocking all trades.")
            return 0
            
        # Cap the maximum allocation fraction at 15% of capital (before leverage) to manage risk
        max_allocation = 0.15
        if allocated_fraction > max_allocation:
            logger.info(f"Sizing: Capping allocation at {max_allocation * 100}% (calculated: {allocated_fraction * 100:.1f}%)")
            allocated_fraction = max_allocation
            
        # Calculate units: Balance * Allocation * Leverage
        units = int(round(balance * allocated_fraction * leverage))
        logger.info(f"Sizing: Allocated={allocated_fraction*100:.2f}% | Leverage={leverage}x | Units={units}")
        return units

    def validate_spread(self, bid: float, ask: float, pip_value: float = 0.0001) -> bool:
        """
        Verify if the current bid-ask spread is within the acceptable limit.
        """
        spread = ask - bid
        spread_pips = spread / pip_value
        
        if spread_pips > self.max_spread_pips:
            logger.warning(f"Risk: Spread is too wide ({spread_pips:.1f} pips). Max allowed: {self.max_spread_pips} pips.")
            return False
        return True

    def check_daily_drawdown(self, balance: float, unrealized_pnl: float = 0.0) -> bool:
        """
        Check if today's cumulative realized + unrealized losses exceed the drawdown limit.
        
        Returns:
            True if within risk limits (trading allowed).
            False if drawdown limit is breached (halt trading).
        """
        realized_pnl = self._get_today_realized_pnl()
        total_pnl = realized_pnl + unrealized_pnl
        
        max_loss = -self.daily_drawdown_limit * balance
        
        if total_pnl < max_loss:
            logger.critical(
                f"Risk: Drawdown limit breached! Total PnL today: ${total_pnl:.2f} "
                f"exceeds limit of ${max_loss:.2f} (2% of balance ${balance:.2f}). HALTING TRADING."
            )
            return False
            
        logger.info(f"Risk: Daily drawdown check OK. Today's PnL: ${total_pnl:.2f} (Limit: ${max_loss:.2f})")
        return True

    def _get_today_realized_pnl(self) -> float:
        """Fetch sum of closed trade PnL for today from PostgreSQL."""
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            with conn.cursor() as cur:
                # Query realized trades closed since start of current calendar day (UTC)
                cur.execute("""
                    SELECT COALESCE(SUM(pnl_amount), 0.0) 
                    FROM live_trades 
                    WHERE exit_time >= CURRENT_DATE 
                      AND exit_time IS NOT NULL
                """)
                row = cur.fetchone()
                pnl = float(row[0]) if row else 0.0
                conn.close()
                return pnl
        except Exception as e:
            logger.error(f"Error fetching daily realized PnL: {e}")
            return 0.0
