"""
Rung 0: the naive benchmark. Everything else must beat this.

For a *return* series the random-walk-in-price forecast is E[r_{t+1}] = 0. The
predictive distribution is the trailing empirical distribution of returns, which
gives an honest (if unconditional) probabilistic benchmark: it captures the
series' unconditional volatility and fat tails without any conditional signal.

If a sophisticated model cannot beat this, the series is efficient at that horizon
-- a finding, not a failure.
"""

from __future__ import annotations

import numpy as np

from .base import Forecaster
from ..contracts import QUANTILE_LEVELS


class NaiveForecaster(Forecaster):
    name = "naive-rw"
    rung = 0
    stateful = False

    def __init__(self, drift: bool = False, tail_window: int = 250):
        # drift=False -> pure random walk (mean 0). drift=True -> trailing mean.
        self.drift = drift
        self.tail_window = tail_window
        self._mu = 0.0

    def fit(self, train: np.ndarray) -> "NaiveForecaster":
        self._mu = float(np.mean(train)) if self.drift else 0.0
        return self

    def predict(self, context: np.ndarray, horizon: int):
        tail = context[-self.tail_window:] if len(context) > self.tail_window else context
        mu = float(np.mean(tail)) if self.drift else 0.0
        # Empirical quantiles of the trailing return distribution, held constant
        # across the horizon (random walk => same one-step law each step).
        emp = {float(l): np.full(horizon, float(np.quantile(tail, l)) + mu - (
            float(np.mean(tail)) if self.drift else 0.0))
            for l in QUANTILE_LEVELS}
        # center the empirical quantiles on mu while preserving spread
        centered = {}
        med = float(np.quantile(tail, 0.5))
        for l in QUANTILE_LEVELS:
            centered[float(l)] = np.full(horizon, float(np.quantile(tail, l)) - med + mu)
        mean = np.full(horizon, mu)
        return self._result(mean, centered)
