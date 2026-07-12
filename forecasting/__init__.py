"""
Robust probabilistic forecasting pipeline (Chronos / TimesFM + statistical ladder).

Two output planes off one shared forecast core:

  * Statistical plane  -> characterize the data (predictability, stationarity,
                          regimes, information content). Forecasting is used as a
                          *measurement instrument*.
  * Operational plane  -> a point-in-time veto gate (ForecastSignal) consumed by
                          the downstream strategy lab.

Design invariants (enforced throughout, do not violate):

  1. POINT-IN-TIME. Every statistic used at time t is computed only from data
     available at or before t. Scalers, regime labels, thresholds -- all fit on
     the training window and applied forward. Any global fit is a leak.

  2. ONE CODE PATH. Backtest and live emit ForecastSignal from the same functions
     so backtest results transfer to production.

  3. MODEL RETURNS, NOT PRICES. The target is log-returns (stationary). Price
     levels are reconstructed only for reporting.

  4. BEAT THE NAIVE BASELINE. Nothing is trusted until it beats a random walk on
     scaled error AND is non-degenerate. The gate's primary job is to know its
     own failure modes.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import contracts, prep, windows, metrics, characterize, gate, backtest

__all__ = [
    "contracts",
    "prep",
    "windows",
    "metrics",
    "characterize",
    "gate",
    "backtest",
    "__version__",
]
