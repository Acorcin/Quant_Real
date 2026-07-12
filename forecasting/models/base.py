"""
Forecaster interface. Every rung implements the same two-method contract so the
backtest engine treats them uniformly and the resulting ForecastResults are
directly comparable (same quantile grid, same target).
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from ..contracts import ForecastResult, QUANTILE_LEVELS


class SkipModel(Exception):
    """Raised by a forecaster that cannot run in the current environment (e.g. a
    foundation model whose weights/deps are unavailable). The backtest catches it
    and continues the ladder rather than aborting the whole run."""


class Forecaster(abc.ABC):
    name: str = "base"
    rung: int = -1
    target: str = "log_return"

    # Whether the model must be refit at each origin (expensive) or can reuse a
    # periodic fit and simply condition on the latest context. Naive/foundation
    # are stateless (refit cheap/none); AR/GARCH set this and honour refit_every.
    stateful: bool = False

    @abc.abstractmethod
    def fit(self, train: np.ndarray) -> "Forecaster":
        """Estimate parameters from training returns. Must not see test data."""

    @abc.abstractmethod
    def predict(self, context: np.ndarray, horizon: int) -> ForecastResult:
        """Produce a probabilistic forecast for the next `horizon` steps given the
        trailing `context` of returns ending at the forecast origin.

        `origin_index` on the returned ForecastResult is filled by the backtest.
        """

    # ---- shared helpers -------------------------------------------------

    @staticmethod
    def _gaussian_quantiles(mean: np.ndarray, sigma: np.ndarray) -> dict:
        """Build the canonical quantile grid from a per-step Gaussian (mean, sigma).

        Many rungs are Gaussian-predictive (AR residuals, GARCH sigma). Foundation
        models override with empirical/sample quantiles instead."""
        from scipy.stats import norm
        mean = np.atleast_1d(np.asarray(mean, float))
        sigma = np.atleast_1d(np.asarray(sigma, float))
        sigma = np.maximum(sigma, 1e-12)
        return {float(l): mean + sigma * norm.ppf(l) for l in QUANTILE_LEVELS}

    def _result(self, mean: np.ndarray, quantiles: dict,
                origin_index: int = -1) -> ForecastResult:
        return ForecastResult(
            model=self.name, rung=self.rung, origin_index=origin_index,
            mean=np.atleast_1d(np.asarray(mean, float)),
            quantiles=quantiles, target=self.target,
        )
