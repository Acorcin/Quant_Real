"""
Rung 2: predictable volatility. GARCH(1,1) with a constant (or AR) mean.

This rung isolates the classic finance result: even when the *direction* of
returns is unpredictable (mean ~ 0, ACF ~ 0), the *variance* is highly
predictable (volatility clustering). If rung 2 beats naive on distributional
score (CRPS / coverage) but directional accuracy stays at chance, the structure
is VOL_ONLY -- you can forecast risk, not direction. That is exactly the signal
the veto gate needs for sizing.

Uses the `arch` package when available; otherwise falls back to an EWMA /
RiskMetrics volatility estimate, which captures the same clustering with a fixed
decay and no fitting.
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import Forecaster


class GARCHForecaster(Forecaster):
    name = "garch"
    rung = 2
    stateful = True

    def __init__(self, ewma_lambda: float = 0.94):
        self.ewma_lambda = ewma_lambda
        self._mu = 0.0
        self._backend = None
        self._fitted = None
        self._uncond_var = None

    def fit(self, train: np.ndarray) -> "GARCHForecaster":
        self._mu = float(np.mean(train))
        self._uncond_var = float(np.var(train, ddof=1))
        try:
            from arch import arch_model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # scale to % returns for numerical stability, unscale on predict
                am = arch_model(train * 100.0, mean="Constant", vol="GARCH",
                                p=1, q=1, dist="t")
                self._fitted = am.fit(disp="off")
            self._backend = "arch"
        except Exception:
            self._backend = "ewma"
            self._fitted = None
        return self

    def _ewma_sigma(self, context: np.ndarray, horizon: int) -> np.ndarray:
        lam = self.ewma_lambda
        var = self._uncond_var if self._uncond_var else float(np.var(context, ddof=1))
        for x in context:
            var = lam * var + (1 - lam) * (x - self._mu) ** 2
        # EWMA one-step var; flat forecast across horizon (RiskMetrics convention)
        return np.sqrt(np.maximum(np.full(horizon, var), 1e-24))

    def predict(self, context: np.ndarray, horizon: int):
        mean = np.full(horizon, self._mu)
        if self._backend == "arch" and self._fitted is not None:
            try:
                fc = self._fitted.forecast(horizon=horizon, reindex=False)
                var = fc.variance.to_numpy().ravel()[:horizon] / (100.0 ** 2)
                sigma = np.sqrt(np.maximum(var, 1e-24))
                # Student-t predictive quantiles (fat tails) using fitted nu
                nu = float(self._fitted.params.get("nu", 8.0))
                q = self._t_quantiles(mean, sigma, nu)
                return self._result(mean, q)
            except Exception:
                pass
        sigma = self._ewma_sigma(context, horizon)
        q = self._gaussian_quantiles(mean, sigma)
        return self._result(mean, q)

    @staticmethod
    def _t_quantiles(mean: np.ndarray, sigma: np.ndarray, nu: float) -> dict:
        from scipy.stats import t as student_t
        from ..contracts import QUANTILE_LEVELS
        nu = max(nu, 2.1)
        # scale so that the t-distribution has the given sigma as its std
        scale = sigma * np.sqrt((nu - 2.0) / nu)
        return {float(l): mean + scale * student_t.ppf(l, df=nu)
                for l in QUANTILE_LEVELS}
