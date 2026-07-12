"""
Rung 1: linear autocorrelation. An AR(p) model on returns, order chosen by AIC.

If this rung beats naive but rung 2/3 add nothing, the predictability is *linear*
-- there is exploitable autocorrelation in the mean (momentum or mean reversion).
Predictive distribution is Gaussian with the fitted residual variance, propagated
across the horizon via the AR recursion.
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import Forecaster, SkipModel


class ARForecaster(Forecaster):
    name = "ar"
    rung = 1
    stateful = True

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag
        self._params = None      # (const, phi[])
        self._sigma2 = None
        self._order = 0

    def fit(self, train: np.ndarray) -> "ARForecaster":
        try:
            from statsmodels.tsa.ar_model import AutoReg
            from statsmodels.tools.sm_exceptions import ValueWarning
        except ImportError as e:  # pragma: no cover
            raise SkipModel("statsmodels not installed") from e

        best_aic, best = np.inf, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(1, self.max_lag + 1):
                if len(train) <= p + 10:
                    break
                try:
                    res = AutoReg(train, lags=p, old_names=False).fit()
                except Exception:
                    continue
                if res.aic < best_aic:
                    best_aic, best = res.aic, res
        if best is None:
            raise SkipModel("AR fit failed for all orders")

        params = np.asarray(best.params, float)
        self._order = len(params) - 1
        self._params = params
        self._sigma2 = float(best.sigma2)
        return self

    def predict(self, context: np.ndarray, horizon: int):
        if self._params is None:
            raise SkipModel("AR not fit")
        const = self._params[0]
        phi = self._params[1:]
        p = self._order
        hist = list(context[-p:]) if p > 0 else []
        # pad if context shorter than order
        while len(hist) < p:
            hist.insert(0, 0.0)

        means = np.empty(horizon)
        # Forecast-error variance grows with horizon: var_h = sigma2 * sum psi_i^2.
        # We accumulate via the recursion on the MA(inf) psi weights implicitly by
        # propagating point forecasts and tracking variance through the AR filter.
        psi = _ar_to_ma(phi, horizon)
        var_h = self._sigma2 * np.cumsum(psi ** 2)
        h_state = list(hist)
        for h in range(horizon):
            yhat = const + (np.dot(phi, h_state[::-1][:p]) if p > 0 else 0.0)
            means[h] = yhat
            h_state.append(yhat)
        sigma = np.sqrt(np.maximum(var_h, 1e-24))
        q = self._gaussian_quantiles(means, sigma)
        return self._result(means, q)


def _ar_to_ma(phi: np.ndarray, horizon: int) -> np.ndarray:
    """Convert AR coefficients to the first `horizon` MA(inf) psi-weights so we can
    accumulate multi-step forecast variance. psi_0 = 1."""
    p = len(phi)
    psi = np.zeros(horizon)
    psi[0] = 1.0
    for j in range(1, horizon):
        s = 0.0
        for i in range(1, min(j, p) + 1):
            s += phi[i - 1] * psi[j - i]
        psi[j] = s
    return psi
