import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanPriceFilter:
    """
    2D Kalman Filter for FX Price Smoothing and Velocity Estimation.
    State vector: x = [price, velocity]^T
    Measurement: z = price
    """
    def __init__(self, R_val: float = 1e-3, Q_accel: float = 1e-5):
        """
        Initialize the Kalman Filter.
        
        Args:
            R_val: Measurement noise covariance (larger value smooths more but adds lag).
            Q_accel: Process noise acceleration variance (larger value tracks price shifts faster).
        """
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.zeros(2)
        self.kf.H = np.array([[1.0, 0.0]])
        self.kf.R = np.array([[R_val]])
        self.kf.P = np.eye(2) * 1.0
        self.Q_accel = Q_accel
        self.initialized = False
        
    def update(self, price: float, dt: float) -> tuple[float, float]:
        """
        Predict and update the state with a new price measurement.
        
        Args:
            price: New mid-price tick.
            dt: Time elapsed since the last tick in seconds.
            
        Returns:
            Tuple of (smoothed_price, velocity).
        """
        if not self.initialized:
            self.kf.x = np.array([price, 0.0])
            self.initialized = True
            return price, 0.0
            
        # Ensure dt is positive and bounded
        dt = max(dt, 0.0001)
        
        # Update State Transition Matrix (F) with dt
        self.kf.F = np.array([[1.0, dt],
                              [0.0, 1.0]])
                              
        # Update Process Noise Covariance (Q) based on continuous white noise model
        self.kf.Q = np.array([
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt]
        ]) * self.Q_accel
        
        # Predict step
        self.kf.predict()
        
        # Measurement update step
        self.kf.update(np.array([price]))
        
        return float(self.kf.x[0]), float(self.kf.x[1])
        
    def get_state(self) -> tuple[float, float]:
        """Return current state [price, velocity]."""
        return float(self.kf.x[0]), float(self.kf.x[1])
        
    def get_covariance(self) -> list:
        """Return the current covariance matrix as a nested list for serialization."""
        return self.kf.P.tolist()
        
    def set_state(self, x: list, P: list):
        """Restore state from serialized values."""
        self.kf.x = np.array(x)
        self.kf.P = np.array(P)
        self.initialized = True
