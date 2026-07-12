"""
Preprocessing core -- the single code path shared by backtest and live.

Philosophy for financial data (differs from generic time-series cleaning):

  * We forecast LOG-RETURNS (stationary), reconstructing price only for reporting.
  * A 3-sigma move is usually *signal*, not noise. We DETECT and FLAG anomalies as
    features; we do NOT delete them. Only true data errors (non-positive price,
    duplicate timestamps, exact-repeat stale ticks) are repaired.
  * No naive forward-fill of returns: carrying a value forward injects look-ahead.
    Calendar gaps (market closed) are legitimate and left alone.
  * Any scaler / anomaly threshold is fit on the TRAINING window only and applied
    forward. A global fit leaks the future into the past.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .data import Series


# ---------------------------------------------------------------------------
# Cleaning true data errors (safe, causal)
# ---------------------------------------------------------------------------

@dataclass
class CleanReport:
    n_input: int
    n_dropped_dupes: int
    n_nonpositive_repaired: int
    n_stale_flagged: int


def clean_prices(series: Series, stale_run: int = 5) -> Tuple[Series, CleanReport]:
    """Repair only *true* errors. Returns a cleaned Series plus an audit report.

    - Duplicate timestamps: keep last (latest correction wins).
    - Non-positive prices: impossible for a traded asset -> replace with previous
      valid price (a genuine data error, not a market move).
    - Stale runs (identical price repeated `stale_run`+ times): flagged, not
      removed -- often a feed freeze the gate should know about.
    """
    idx = series.index
    price = series.price.copy()

    # 1) de-duplicate timestamps, keep last
    df = pd.DataFrame({"price": price}, index=idx)
    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    n_dupes = before - len(df)

    price = df["price"].to_numpy()
    idx = df.index

    # 2) repair non-positive prices causally (carry last valid forward). This is
    # the ONLY forward-fill we allow, and only for impossible values.
    n_repaired = 0
    for i in range(len(price)):
        if price[i] <= 0 or not np.isfinite(price[i]):
            n_repaired += 1
            price[i] = price[i - 1] if i > 0 else np.nan
    if np.isnan(price[0]):
        # first value unusable: find first finite positive and back-fill index
        first = np.argmax((price > 0) & np.isfinite(price))
        price = price[first:]
        idx = idx[first:]

    # 3) flag stale runs (feed freeze) without altering data
    n_stale = 0
    if len(price) >= stale_run:
        same = np.r_[False, price[1:] == price[:-1]]
        run = 0
        for i in range(len(same)):
            run = run + 1 if same[i] else 0
            if run >= stale_run - 1:
                n_stale += 1

    cleaned = Series(series.instrument, idx, price, series.is_synthetic, series.truth)
    return cleaned, CleanReport(len(series.price), n_dupes, n_repaired, n_stale)


# ---------------------------------------------------------------------------
# Target construction: log-returns
# ---------------------------------------------------------------------------

def log_returns(price: np.ndarray) -> np.ndarray:
    """r_t = ln(P_t) - ln(P_{t-1}). Length n-1, aligned to the *later* timestamp."""
    lp = np.log(price)
    return np.diff(lp)


def reconstruct_price(p0: float, returns: np.ndarray) -> np.ndarray:
    """Inverse of log_returns: rebuild the price path from a start level."""
    return p0 * np.exp(np.cumsum(returns))


# ---------------------------------------------------------------------------
# Anomaly detection -> feature flags (never deletion), computed causally
# ---------------------------------------------------------------------------

@dataclass
class AnomalyFlags:
    """Per-observation boolean flags, computed with trailing windows only so they
    are valid point-in-time features. Aligned to the return series."""

    zscore: np.ndarray          # |trailing z| > k
    iqr: np.ndarray             # outside trailing [Q1-1.5IQR, Q3+1.5IQR]
    vol_spike: np.ndarray       # trailing realized vol >> its own median
    any: np.ndarray             # union

    def rate(self) -> float:
        return float(self.any.mean()) if len(self.any) else 0.0


def flag_anomalies(returns: np.ndarray, window: int = 60, z_k: float = 3.0,
                   vol_mult: float = 3.0) -> AnomalyFlags:
    """Trailing (causal) anomaly flags on the RETURN series.

    Every threshold uses only the `window` observations *strictly before* t, so a
    flag at t never peeks at t or later. This is what makes these usable both as
    live features and in an honest backtest.
    """
    n = len(returns)
    z = np.zeros(n, dtype=bool)
    iqr_flag = np.zeros(n, dtype=bool)
    vspike = np.zeros(n, dtype=bool)

    if n == 0:
        return AnomalyFlags(z, iqr_flag, vspike, z)

    s = pd.Series(returns)
    # trailing stats EXCLUDING current point: shift(1) then roll
    roll = s.shift(1).rolling(window, min_periods=max(10, window // 2))
    mu = roll.mean().to_numpy()
    sd = roll.std(ddof=1).to_numpy()
    q1 = roll.quantile(0.25).to_numpy()
    q3 = roll.quantile(0.75).to_numpy()
    # trailing realized vol and its trailing median
    rv = s.shift(1).rolling(window, min_periods=max(10, window // 2)).std(ddof=1)
    rv_med = rv.rolling(window, min_periods=max(10, window // 2)).median().to_numpy()
    rv = rv.to_numpy()

    with np.errstate(invalid="ignore"):
        z = np.abs((returns - mu) / np.where(sd > 0, sd, np.nan)) > z_k
        iqrv = q3 - q1
        lo = q1 - 1.5 * iqrv
        hi = q3 + 1.5 * iqrv
        iqr_flag = (returns < lo) | (returns > hi)
        vspike = rv > (vol_mult * rv_med)

    z = np.nan_to_num(z, nan=False).astype(bool)
    iqr_flag = np.nan_to_num(iqr_flag, nan=False).astype(bool)
    vspike = np.nan_to_num(vspike, nan=False).astype(bool)
    return AnomalyFlags(z, iqr_flag, vspike, z | iqr_flag | vspike)


# ---------------------------------------------------------------------------
# Causal scaler (fit on train window, apply forward)
# ---------------------------------------------------------------------------

@dataclass
class CausalScaler:
    """Standardize returns using ONLY training-window statistics.

    Note: Chronos self-scales internally, so external scaling is optional for it
    and can be harmful (double-scaling). It is provided for the linear/GARCH rungs
    and for entropy measures that assume comparable magnitudes. Fit strictly on
    train; never re-fit on the combined series."""

    mu: float = 0.0
    sd: float = 1.0

    @classmethod
    def fit(cls, train_returns: np.ndarray) -> "CausalScaler":
        mu = float(np.mean(train_returns))
        sd = float(np.std(train_returns, ddof=1))
        return cls(mu, sd if sd > 1e-12 else 1.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.sd + self.mu


@dataclass
class PreparedSeries:
    """Everything downstream needs, derived once, causally."""

    instrument: str
    index: pd.DatetimeIndex     # aligned to returns (len n-1)
    price: np.ndarray           # cleaned price (len n)
    returns: np.ndarray         # log-returns (len n-1)
    flags: AnomalyFlags
    clean_report: CleanReport


def prepare(series: Series, anomaly_window: int = 60) -> PreparedSeries:
    """Full causal preprocessing: clean -> log-returns -> anomaly flags.

    Returns aligned so that returns[i] and flags.any[i] both correspond to
    index[i], the timestamp at which that return is *realized*.
    """
    cleaned, report = clean_prices(series)
    r = log_returns(cleaned.price)
    flags = flag_anomalies(r, window=anomaly_window)
    return PreparedSeries(
        instrument=cleaned.instrument,
        index=cleaned.index[1:],   # returns align to the later timestamp
        price=cleaned.price,
        returns=r,
        flags=flags,
        clean_report=report,
    )
