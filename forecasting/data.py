"""
Data layer: ingestion and a synthetic generator with *known* properties.

The synthetic generator is not a toy -- it is a test oracle. Because the whole
point of Phase 3 is to *measure* structure (predictability, regimes, memory), we
need series whose structure we set by hand so we can check the instrument reports
it correctly. Each generator documents the ground-truth reading the
characterization should recover.

Real loaders (Postgres / Databento / Parquet) are thin and defer schema details
to the caller; they exist so the same prep code path runs on real and synthetic
data without branching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Series:
    """A univariate price/return series aligned to a monotone UTC index.

    We carry *price* as the canonical field and derive log-returns in prep, so
    that reconstruction and reporting always trace back to a single source of
    truth. `is_synthetic` and `truth` let tests assert the instrument's reading."""

    instrument: str
    index: pd.DatetimeIndex
    price: np.ndarray
    is_synthetic: bool = False
    truth: Optional[dict] = None

    def __post_init__(self) -> None:
        self.price = np.asarray(self.price, dtype=float)
        if len(self.index) != len(self.price):
            raise ValueError("index and price length mismatch")
        if not self.index.is_monotonic_increasing:
            raise ValueError("index must be monotone increasing (UTC-sorted)")
        if np.any(~np.isfinite(self.price)):
            raise ValueError("price contains non-finite values; clean before Series")
        if np.any(self.price <= 0):
            raise ValueError("price must be strictly positive for log-returns")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"price": self.price}, index=self.index)


def _bdays(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n, tz="UTC")


# ---------------------------------------------------------------------------
# Synthetic generators (each = a labelled ground truth)
# ---------------------------------------------------------------------------

def gen_random_walk(n: int = 1000, sigma: float = 0.01, seed: int = 0,
                    s0: float = 100.0) -> Series:
    """Efficient market: log-returns are iid Gaussian white noise.

    Ground truth the instrument MUST recover:
      structure_type = EFFICIENT, no rung beats naive, ACF(returns)~0,
      Hurst ~ 0.5, permutation entropy ~ 1.0.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, sigma, size=n)
    price = s0 * np.exp(np.cumsum(r))
    return Series("RW", _bdays(n), price, True,
                  {"structure_type": "efficient", "hurst": 0.5})


def gen_ar1_returns(n: int = 1000, phi: float = 0.3, sigma: float = 0.01,
                    seed: int = 0, s0: float = 100.0) -> Series:
    """Linear structure: returns follow AR(1) with autocorrelation phi.

    Ground truth: structure_type = LINEAR, rung-1 (AR) beats naive,
      ACF(returns) significant at lag 1, Hurst != 0.5 (mean-reverting if phi<0).
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    eps = rng.normal(0.0, sigma, size=n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    price = s0 * np.exp(np.cumsum(r))
    return Series("AR1", _bdays(n), price, True,
                  {"structure_type": "linear", "phi": phi})


def gen_garch(n: int = 1000, omega: float = 1e-6, alpha: float = 0.1,
              beta: float = 0.85, seed: int = 0, s0: float = 100.0) -> Series:
    """Volatility clustering: returns are white noise in the mean but their
    variance follows GARCH(1,1). This is the canonical finance case.

    Ground truth: structure_type = VOL_ONLY, ACF(returns)~0 but
      ACF(squared returns) strongly significant, GARCH rung beats naive on the
      *volatility* forecast while direction stays unpredictable.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    var = np.zeros(n)
    var[0] = omega / max(1e-12, (1 - alpha - beta))
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        r[t] = rng.normal(0.0, np.sqrt(var[t]))
    price = s0 * np.exp(np.cumsum(r))
    return Series("GARCH", _bdays(n), price, True,
                  {"structure_type": "vol_only", "alpha": alpha, "beta": beta})


def gen_regime_switch(n: int = 1200, seed: int = 0, s0: float = 100.0) -> Series:
    """Two-regime series: alternating calm (low vol) and turbulent (high vol,
    fat tails) blocks. Turbulent regime is designed to be *unpredictable* so the
    gate should veto inside it.

    Ground truth: >=1 structural break, two vol regimes, one flagged
      unpredictable.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    labels = np.zeros(n, dtype=int)
    t = 0
    regime = 0
    while t < n:
        block = rng.integers(80, 160)
        for _ in range(block):
            if t >= n:
                break
            if regime == 0:  # calm, slight momentum
                r[t] = 0.2 * r[t - 1] + rng.normal(0, 0.006) if t > 0 else rng.normal(0, 0.006)
            else:            # turbulent, heavy-tailed, memoryless
                r[t] = rng.standard_t(df=3) * 0.02
            labels[t] = regime
            t += 1
        regime = 1 - regime
    price = s0 * np.exp(np.cumsum(r))
    return Series("REGIME", _bdays(n), price, True,
                  {"structure_type": "vol_only", "n_regimes": 2,
                   "true_labels": labels})


# ---------------------------------------------------------------------------
# Real loaders (thin; caller provides connection / path)
# ---------------------------------------------------------------------------

def load_parquet(path: str, instrument: str,
                 price_col: str = "close",
                 ts_col: str = "ts") -> Series:
    """Load a cleaned price series from Parquet (the immutable source of truth).

    Parquet is the interchange format between the one-time DB pull and every
    downstream run, so we never re-hit the database while iterating."""
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(pd.to_datetime(df[ts_col], utc=True))
    order = np.argsort(idx.values)
    return Series(instrument, idx[order], df[price_col].to_numpy()[order])


def load_postgres(dsn: str, query: str, instrument: str,
                  price_col: str = "close", ts_col: str = "ts") -> Series:
    """Pull historical bars via SQLAlchemy. Kept import-local so the package runs
    without a DB driver installed. Use connection pooling in the DSN to survive
    Colab timeouts (e.g. pool_pre_ping=True, pool_recycle=280)."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError("install sqlalchemy + psycopg2 to load from Postgres") from e
    engine = create_engine(dsn, pool_pre_ping=True, pool_recycle=280)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    idx = pd.DatetimeIndex(pd.to_datetime(df[ts_col], utc=True))
    order = np.argsort(idx.values)
    return Series(instrument, idx[order], df[price_col].to_numpy()[order])
