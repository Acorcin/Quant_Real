"""
Phase 3: forecasting as a measurement instrument.

This module answers "what is this series made of" and packages it as a
MarketCharacterization. Four axes:

  1. Predictability  -> R2_oos by horizon (from the backtest) + model-free entropy
  2. Stationarity    -> ADF + KPSS (designed to disagree), rolling profile, breaks
  3. Regimes         -> volatility-clustering labels; error-based unpredictability
  4. Memory / info   -> Hurst, Ljung-Box on returns vs squared returns, MI-by-lag

The ladder classifier turns the rung-by-rung backtest into a single structure_type
verdict -- the headline reading that parameterizes the veto gate.

Every statistic here is descriptive/structural and computed on the *full available
history as of a cutoff*; it is not used inside the point-in-time gate directly.
Instead it sets the gate's thresholds (which regimes/horizons are permitted). The
gate itself recomputes its point-in-time inputs causally in gate.py.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from .contracts import (LadderReading, MarketCharacterization, StructureType)


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------

def adf_test(x: np.ndarray) -> float:
    """Augmented Dickey-Fuller. H0: a unit root (non-stationary). Low p => reject
    => stationary. Returns the p-value."""
    from statsmodels.tsa.stattools import adfuller
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(x, autolag="AIC")[1])


def kpss_test(x: np.ndarray) -> float:
    """KPSS. H0: stationarity (the reverse of ADF). Low p => reject => NON-stationary.
    Using both is deliberate: ADF and KPSS have complementary nulls, and their
    agreement/disagreement is itself informative (e.g. both inconclusive =>
    fractional integration / long memory)."""
    from statsmodels.tsa.stattools import kpss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(kpss(x, regression="c", nlags="auto")[1])


def rolling_stationarity(x: np.ndarray, window: int = 250,
                         step: int = 25) -> np.ndarray:
    """Fraction of rolling windows that test as stationary (ADF p<0.05). Treats
    stationarity as time-varying rather than a single global verdict."""
    flags = []
    for start in range(0, len(x) - window, step):
        seg = x[start:start + window]
        try:
            flags.append(adf_test(seg) < 0.05)
        except Exception:
            continue
    return np.array(flags, dtype=float)


def structural_breaks(x: np.ndarray, min_size: int = 60,
                      penalty: float = 3.0) -> List[int]:
    """Detect variance/mean breaks by binary segmentation on a normalized CUSUM of
    squares. Returns break indices. Deliberately dependency-free (no `ruptures`)
    so it runs anywhere; conservative penalty avoids over-segmenting noise."""
    breaks: List[int] = []

    def _seg(lo: int, hi: int):
        if hi - lo < 2 * min_size:
            return
        seg = x[lo:hi]
        # CUSUM of squared deviations -> point of maximal variance regime change
        c = np.cumsum(seg ** 2)
        c = c / (c[-1] + 1e-12)
        ideal = np.linspace(0, 1, len(seg))
        dev = np.abs(c - ideal)
        k = int(np.argmax(dev))
        stat = dev[k] * np.sqrt(len(seg))
        if stat > penalty and min_size <= k <= (hi - lo) - min_size:
            bp = lo + k
            breaks.append(bp)
            _seg(lo, bp)
            _seg(bp, hi)

    _seg(0, len(x))
    return sorted(breaks)


# ---------------------------------------------------------------------------
# Memory / information content
# ---------------------------------------------------------------------------

def hurst_exponent(x: np.ndarray, max_lag: int = 64) -> float:
    """Hurst via rescaled-range regression on aggregated variance.

    H < 0.5 mean-reverting, ~0.5 random walk, > 0.5 trending/persistent.
    Estimated from the slope of log(std of lag-differences) vs log(lag)."""
    x = np.asarray(x, float)
    lags = np.arange(2, min(max_lag, len(x) // 2))
    if len(lags) < 4:
        return 0.5
    tau = [np.std(x[lag:] - x[:-lag], ddof=1) for lag in lags]
    tau = np.asarray(tau)
    good = tau > 0
    if good.sum() < 4:
        return 0.5
    slope = np.polyfit(np.log(lags[good]), np.log(tau[good]), 1)[0]
    return float(slope)


def ljung_box(x: np.ndarray, lags: int = 20) -> float:
    """Ljung-Box test. H0: no autocorrelation up to `lags` (white noise). Low p =>
    autocorrelation present. Applied to returns (mean structure) and to squared
    returns (volatility structure)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = acorr_ljungbox(x, lags=[lags], return_df=True)
    return float(res["lb_pvalue"].iloc[-1])


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalized permutation entropy in [0,1]. 0 = perfectly predictable ordinal
    structure, 1 = maximally random. A model-free predictability reading that does
    not depend on any forecaster, used to cross-check the ladder."""
    x = np.asarray(x, float)
    n = len(x)
    if n < order * delay + 1:
        return 1.0
    from itertools import permutations
    perms = list(permutations(range(order)))
    counts = {p: 0 for p in perms}
    total = 0
    for i in range(n - delay * (order - 1)):
        window = x[i:i + delay * order:delay]
        pattern = tuple(np.argsort(window))
        counts[pattern] += 1
        total += 1
    probs = np.array([c / total for c in counts.values() if c > 0])
    ent = -np.sum(probs * np.log(probs))
    return float(ent / np.log(len(perms)))


def mutual_information_lag(x: np.ndarray, lag: int = 1, bins: int = 16) -> float:
    """Discretized mutual information between x_t and x_{t-lag}. Captures nonlinear
    dependence that a linear ACF misses (>0 even when correlation is ~0)."""
    a, b = x[lag:], x[:-lag]
    c_ab = np.histogram2d(a, b, bins=bins)[0]
    p_ab = c_ab / c_ab.sum()
    p_a = p_ab.sum(axis=1, keepdims=True)
    p_b = p_ab.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = p_ab * np.log(p_ab / (p_a * p_b))
    return float(np.nansum(mi))


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

def volatility_regimes(returns: np.ndarray, n_regimes: int = 2,
                       vol_window: int = 20) -> Tuple[np.ndarray, Dict[int, str]]:
    """Label each observation by volatility regime.

    Prefers a Gaussian HMM on trailing log-realized-vol (captures persistence of
    regimes) and falls back to quantile bucketing of trailing vol when hmmlearn is
    unavailable. Regime ids are re-sorted so 0 = lowest vol for stable naming."""
    import pandas as pd
    rv = pd.Series(returns).rolling(vol_window, min_periods=5).std(ddof=1)
    rv = rv.bfill().to_numpy()
    logrv = np.log(np.maximum(rv, 1e-12)).reshape(-1, 1)

    labels = None
    try:
        from hmmlearn.hmm import GaussianHMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(n_components=n_regimes, covariance_type="diag",
                              n_iter=100, random_state=0)
            hmm.fit(logrv)
            labels = hmm.predict(logrv)
        order = np.argsort(hmm.means_.ravel())
    except Exception:
        # quantile fallback
        qs = np.quantile(rv, np.linspace(0, 1, n_regimes + 1)[1:-1])
        labels = np.digitize(rv, qs)
        order = np.argsort([np.median(rv[labels == k]) if np.any(labels == k)
                            else np.inf for k in range(n_regimes)])

    remap = {int(old): new for new, old in enumerate(order)}
    labels = np.array([remap.get(int(l), int(l)) for l in labels])
    names = {0: "calm"}
    if n_regimes == 2:
        names = {0: "calm", 1: "turbulent"}
    else:
        names = {k: f"vol_q{k}" for k in range(n_regimes)}
    return labels, names


# ---------------------------------------------------------------------------
# The ladder classifier -> structure_type
# ---------------------------------------------------------------------------

def classify_structure(
    ladder: List[LadderReading],
    *,
    lb_returns_p: float = 1.0,
    lb_sq_returns_p: float = 1.0,
) -> StructureType:
    """Read the ladder + information-content signatures into one structure verdict.

    The ladder's `beats_naive` only measures MEAN point accuracy, so it detects
    LINEAR and NONLINEAR structure. Volatility structure is invisible to a
    mean-error test, so VOL_ONLY is detected from the model-free Ljung-Box
    signature instead: returns look like white noise (no mean structure) but
    squared returns are strongly autocorrelated (volatility clustering).

    Precedence (most specific / highest evidentiary bar first):
      - Foundation beats the best simpler rung on the mean  -> NONLINEAR
      - AR beats naive with a real directional edge         -> LINEAR
      - returns white-noise but squared returns not         -> VOL_ONLY
      - AR beats naive on the mean (weak/no direction)      -> LINEAR
      - none of the above                                   -> EFFICIENT
    """
    by_rung = {r.rung: r for r in ladder}
    naive = by_rung.get(0)
    ar = by_rung.get(1)
    garch = by_rung.get(2)
    found = by_rung.get(3)

    ar_beats = ar is not None and ar.beats_naive
    ar_direction = ar is not None and ar.directional_pvalue < 0.05 and ar.directional_acc > 0.5

    best_simple_r2 = max([r.r2_oos for r in (ar, garch) if r is not None], default=-np.inf)
    found_beats_simple = (found is not None and found.beats_naive
                          and found.r2_oos > best_simple_r2 + 1e-6)

    # Volatility structure is invisible to a mean-error test, so VOL_ONLY needs
    # two independent confirmations, not one borderline test:
    #   (a) model-free signature: returns white but squared returns autocorrelated
    #   (b) the GARCH rung genuinely improves the out-of-sample predictive DENSITY
    #       (lower CRPS than naive) -- i.e. modelling time-varying vol pays off.
    # Requiring both stops a single spurious Ljung-Box hit on an iid series (which
    # happens ~5% of the time at alpha=0.05) from being read as vol structure.
    returns_white = lb_returns_p >= 0.05
    vol_clustering = lb_sq_returns_p < 0.01   # strict: structural claim, guards
                                              # against a spurious 5%-level hit
    crps_improves = (garch is not None and naive is not None
                     and np.isfinite(garch.crps) and np.isfinite(naive.crps)
                     and garch.crps < naive.crps)
    vol_only = returns_white and vol_clustering and crps_improves

    if found_beats_simple:
        return StructureType.NONLINEAR
    if ar_beats and ar_direction:
        return StructureType.LINEAR
    if vol_only:
        return StructureType.VOL_ONLY
    if ar_beats:
        return StructureType.LINEAR
    return StructureType.EFFICIENT


def characterize(
    instrument: str,
    returns: np.ndarray,
    *,
    ladder: List[LadderReading],
    r2_by_horizon: Dict[int, float],
    regime_scores: Optional[Dict[str, float]] = None,
    data_asof: str = "",
    n_regimes: int = 2,
) -> MarketCharacterization:
    """Assemble the full MarketCharacterization from the raw series plus the
    backtest-derived ladder readings and per-horizon R2.

    `regime_scores` optionally maps regime name -> its R2_oos; regimes with R2<=0
    are recorded as unpredictable (the gate will veto inside them)."""
    r = np.asarray(returns, float)

    adf_p = _safe(adf_test, r, default=1.0)
    kpss_p = _safe(kpss_test, r, default=0.0)
    stationary = (adf_p < 0.05) and (kpss_p > 0.05)

    breaks = _safe(structural_breaks, r, default=[])
    labels, names = volatility_regimes(r, n_regimes=n_regimes)
    current = names.get(int(labels[-1]), str(labels[-1]))

    # predictability edge horizon = largest h with positive R2_oos
    edge_h = 0
    for h in sorted(r2_by_horizon):
        if r2_by_horizon[h] > 0:
            edge_h = h

    unpredictable = []
    if regime_scores:
        unpredictable = [name for name, sc in regime_scores.items() if sc <= 0]

    # information-content signatures (computed before classification so the
    # ladder verdict can use the volatility-clustering signal)
    lb_ret = _safe(ljung_box, r, default=1.0)
    lb_sq = _safe(ljung_box, r ** 2, default=1.0)
    # Hurst belongs on the INTEGRATED series (log-price), not returns.
    hurst = _safe(hurst_exponent, np.cumsum(r), default=0.5)

    return MarketCharacterization(
        instrument=instrument,
        data_asof=data_asof,
        structure_type=classify_structure(
            ladder, lb_returns_p=lb_ret, lb_sq_returns_p=lb_sq),
        ladder=ladder,
        r2_by_horizon=r2_by_horizon,
        predictability_edge_horizon=edge_h,
        adf_pvalue=adf_p,
        kpss_pvalue=kpss_p,
        stationary=stationary,
        structural_breaks=breaks,
        regime_labels=labels,
        regime_names=names,
        current_regime=current,
        unpredictable_regimes=unpredictable,
        hurst=hurst,
        ljung_box_returns_pvalue=lb_ret,
        ljung_box_sq_returns_pvalue=lb_sq,
        permutation_entropy=_safe(permutation_entropy, r, default=1.0),
        provenance={"n_obs": str(len(r))},
    )


def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        return default
