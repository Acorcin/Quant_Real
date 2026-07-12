import collections
import numpy as np

class SpikeFilter:
    """
    Spike/Outlier rejection filter using rolling median and standard deviation.
    Includes a reset counter to prevent getting stuck during genuine sharp market movements.
    """
    def __init__(self, window: int = 50, sigma: float = 4.0, max_consecutive_rejections: int = 5):
        """
        Args:
            window: Rolling window size for statistics.
            sigma: Standard deviation threshold for spike rejection.
            max_consecutive_rejections: Number of consecutive rejections before forcing acceptance.
        """
        self.window = window
        self.sigma = sigma
        self.max_consecutive_rejections = max_consecutive_rejections
        self.history = collections.deque(maxlen=window)
        self.rejection_count = 0
        
    def add_and_check(self, price: float) -> tuple[bool, float]:
        """
        Check if the price is a spike.
        
        Args:
            price: Incoming price tick.
            
        Returns:
            Tuple of (is_spike: bool, filtered_price: float).
        """
        if len(self.history) < 15:
            # Startup phase: accumulate history, assume all valid
            self.history.append(price)
            return False, price
            
        prices = np.array(self.history)
        median = np.median(prices)
        std = np.std(prices)
        
        # Avoid division by zero on flat rates (e.g. weekend close or illiquid periods)
        std = max(std, 1e-6)
        
        z_score = abs(price - median) / std
        
        if z_score > self.sigma:
            self.rejection_count += 1
            if self.rejection_count >= self.max_consecutive_rejections:
                # Force accept the new level to avoid getting stuck on market shifts
                self.rejection_count = 0
                self.history.append(price)
                return False, price
            # Return true (spike detected) and substitute with rolling median
            return True, float(median)
            
        # Reset counter on successful valid tick
        self.rejection_count = 0
        self.history.append(price)
        return False, price
        
    def get_state(self) -> list:
        """Get current history for serialization."""
        return list(self.history)
        
    def set_state(self, history: list):
        """Restore history from state."""
        self.history = collections.deque(history, maxlen=self.window)
        self.rejection_count = 0
        
    def clear(self):
        """Reset filter."""
        self.history.clear()
        self.rejection_count = 0
