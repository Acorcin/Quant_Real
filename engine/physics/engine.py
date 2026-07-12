import numpy as np
from typing import Optional
from physics.kalman import KalmanPriceFilter
from physics.outlier import SpikeFilter

class PhysicsEngine:
    """
    Core mathematical engine for conditioning live currency ticks.
    """
    def __init__(self, R_val: float = 1e-3, Q_accel: float = 1e-5, spike_window: int = 50, spike_sigma: float = 4.0):
        self.spike_filter = SpikeFilter(window=spike_window, sigma=spike_sigma)
        self.kalman_filter = KalmanPriceFilter(R_val=R_val, Q_accel=Q_accel)
        self.prev_smoothed_price: Optional[float] = None
        self.prev_time: Optional[float] = None  # Unix timestamp in seconds
        
    def process_tick(self, bid: float, ask: float, timestamp: float, regime_label: str = "high_vol_choppy", daily_atr: float = 0.0070) -> dict:
        """
        Process a single tick.
        
        Args:
            bid: Bid price.
            ask: Ask price.
            timestamp: Unix timestamp of the tick (seconds).
            regime_label: Semantic HMM regime ('low_vol', 'high_vol_choppy', 'high_vol_crash').
            daily_atr: Daily ATR used to scale returns.
            
        Returns:
            Dict containing:
              - raw_mid: raw mid-price
              - filtered_mid: mid-price after spike rejection
              - is_spike: true if tick was rejected as a spike
              - kalman_price: smoothed price
              - kalman_velocity: estimated price change rate per second
              - tick_return: raw return of the Kalman close
              - normalized_return: return normalized by ATR
              - clipped_return: return clipped to regime-aware thresholds
              - dt: time delta from last tick
        """
        # 1. Calculate Mid Price
        raw_mid = (bid + ask) / 2.0
        
        # 2. Outlier/Spike Rejection
        is_spike, filtered_mid = self.spike_filter.add_and_check(raw_mid)
        
        # 3. Compute elapsed time dt
        if self.prev_time is None:
            dt = 1.0  # default for first tick
        else:
            dt = timestamp - self.prev_time
            if dt <= 0:
                dt = 0.0001  # small positive value to avoid non-positive dt
                
        # 4. Kalman Filtering
        kalman_price, kalman_velocity = self.kalman_filter.update(filtered_mid, dt)
        
        # 5. Compute returns of the Kalman estimate
        tick_return = 0.0
        if self.prev_smoothed_price is not None:
            if self.prev_smoothed_price > 0:
                tick_return = (kalman_price / self.prev_smoothed_price) - 1.0
                
        # 6. Normalize returns using Daily ATR
        atr_val = max(daily_atr, 0.0001)
        normalized_return = tick_return / atr_val
        
        # 7. Regime-Aware Z-Clip
        clip_thresholds = {
            "low_vol":          (-2.5, 2.5),
            "high_vol_choppy":  (-3.5, 3.5),
            "high_vol_crash":   (-5.0, 5.0),
        }
        
        limits = clip_thresholds.get(regime_label, (-3.5, 3.5))
        clipped_return = float(np.clip(normalized_return, limits[0], limits[1]))
        
        # Update states
        self.prev_smoothed_price = kalman_price
        self.prev_time = timestamp
        
        return {
            "raw_mid": raw_mid,
            "filtered_mid": filtered_mid,
            "is_spike": is_spike,
            "kalman_price": kalman_price,
            "kalman_velocity": kalman_velocity,
            "tick_return": tick_return,
            "normalized_return": normalized_return,
            "clipped_return": clipped_return,
            "dt": dt
        }
        
    def get_state(self) -> dict:
        """Get the full serialized state of the engine for database or Redis storage."""
        return {
            "prev_smoothed_price": self.prev_smoothed_price,
            "prev_time": self.prev_time,
            "kalman_x": self.kalman_filter.kf.x.tolist(),
            "kalman_P": self.kalman_filter.get_covariance(),
            "spike_history": self.spike_filter.get_state()
        }
        
    def set_state(self, state: dict):
        """Restore the engine's state from a serialized dict."""
        self.prev_smoothed_price = state.get("prev_smoothed_price")
        self.prev_time = state.get("prev_time")
        
        if "kalman_x" in state and "kalman_P" in state:
            self.kalman_filter.set_state(state["kalman_x"], state["kalman_P"])
            
        if "spike_history" in state:
            self.spike_filter.set_state(state["spike_history"])
