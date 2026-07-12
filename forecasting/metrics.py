"""
Point and probabilistic scoring, plus inferential tests.

Two families:

  Point / distributional accuracy
    - MASE            : mean abs error scaled by the in-sample naive MAE (<1 beats
                        naive). Scale-free, comparable across instruments.
    - R2_oos          : Campbell-Thompson out-of-sample R^2 vs naive. Sign is the
                        headline: <0 means "worse than doing nothing".
    - directional_acc : sign hit-rate, with a binomial test vs 0.5 (an edge is not
                        an edge until it clears the noise floor for its sample size).
    - pinball_loss    : quantile (check) loss; the proper score for a single quantile.
    - crps            : continuous ranked probability score, approximated by
                        integrating pinball loss over the quantile grid. The proper
                        score for the whole predictive distribution.

  Calibration / inference
    - pit             : probability integral transform; if the forecast distribution
                        is correct, PIT values are Uniform(0,1). We test that with a
                        chi-square goodness-of-fit -> the backbone of the gate's
                        `calibration_ok`.
    - interval_coverage : empirical hit-rate of a nominal central interval.
    - diebold_mariano : tests H0 "two models have equal expected loss", with the
                        Harvey-Leybourne-Newbold small-sample correction. This is how
                        we say model A is *significantly* better than B (or naive),
                        rather than better by luck.

Everything is written to accept ragged multi-horizon inputs but the common path is
one-step (horizon=1) evaluation across many walk-forward origins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats

from .contracts import QUANTILE_LEVELS


# ---------------------------------------------------------------------------
# Point / scaled accuracy
# ---------------------------------------------------------------------------

def naive_scale(train_target: np.ndarray, season: int = 1) -> float:
    """Denominator for MASE: mean abs error of the in-sample seasonal-naive
    forecast on the TRAINING data. season=1 -> random walk."""
    d = np.abs(np.diff(train_target, n=1) if season == 1
               else train_target[season:] - train_target[:-season])
    scale = float(np.mean(d))
    return scale if scale > 1e-12 else 1e-12


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    """Mean Absolute Scaled Error. <1 => beats the in-sample naive."""
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray,
           y_naive: np.ndarray) -> float:
    """Out-of-sample R^2 vs a naive benchmark (Campbell & Thompson 2008).

        R2_oos = 1 - SSE(model) / SSE(naive)

    Positive iff the model beats naive on squared error. For return series the
    naive forecast is typically 0 (random walk in returns) or the trailing mean.
    """
    sse_m = float(np.sum((y_true - y_pred) ** 2))
    sse_n = float(np.sum((y_true - y_naive) ** 2))
    if sse_n <= 1e-24:
        return 0.0
    return 1.0 - sse_m / sse_n


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray
                         ) -> Tuple[float, float, int]:
    """Sign hit-rate and its binomial p-value against a fair-coin null.

    Returns (accuracy, p_value, n_effective). Zero-realized-return steps are
    dropped (no direction to predict). The p-value guards against declaring a
    52% hit-rate an 'edge' when n is too small to distinguish it from 0.5.
    """
    mask = np.abs(y_true) > 0
    yt = np.sign(y_true[mask])
    yp = np.sign(y_pred[mask])
    n = int(mask.sum())
    if n == 0:
        return 0.5, 1.0, 0
    hits = int(np.sum(yt == yp))
    acc = hits / n
    # two-sided exact binomial test vs p=0.5
    p = float(stats.binomtest(hits, n, 0.5, alternative="two-sided").pvalue)
    return acc, p, n


# ---------------------------------------------------------------------------
# Proper scores for the predictive distribution
# ---------------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, level: float) -> float:
    """Quantile (check) loss at a single level. Proper for that quantile."""
    diff = y_true - q_pred
    return float(np.mean(np.maximum(level * diff, (level - 1.0) * diff)))


def crps_from_quantiles(y_true: np.ndarray,
                        quantiles: Dict[float, np.ndarray],
                        levels: Sequence[float] = QUANTILE_LEVELS) -> float:
    """Approximate CRPS by integrating pinball loss over the quantile grid.

        CRPS ~= 2 * integral_0^1 pinball_tau d(tau)

    A proper score for the full distribution: rewards both sharpness and
    calibration. Trapezoidal integration over the available levels.
    """
    levels = np.array(sorted(levels))
    losses = np.array([pinball_loss(y_true, quantiles[float(l)], float(l))
                       for l in levels])
    return float(2.0 * _trapezoid(losses, levels))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration, version-safe across numpy 1.x/2.x
    (np.trapz was renamed np.trapezoid in numpy 2.0)."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


# ---------------------------------------------------------------------------
# Calibration / probabilistic reliability
# ---------------------------------------------------------------------------

def pit_values(y_true: np.ndarray,
               quantiles: Dict[float, np.ndarray],
               levels: Sequence[float] = QUANTILE_LEVELS) -> np.ndarray:
    """Probability integral transform: for each observation, the forecast CDF
    evaluated at the realized value, approximated by where y_true falls in the
    predicted quantile grid (linear interpolation between grid points).

    If the predictive distribution is correct, these are ~Uniform(0,1)."""
    levels = np.array(sorted(levels))
    qmat = np.stack([quantiles[float(l)] for l in levels], axis=1)  # (N, Q)
    pit = np.empty(len(y_true))
    for i, y in enumerate(y_true):
        qs = qmat[i]
        if y <= qs[0]:
            pit[i] = levels[0] * (y / qs[0]) if qs[0] != 0 else levels[0]
            pit[i] = min(pit[i], levels[0])
        elif y >= qs[-1]:
            pit[i] = levels[-1]
        else:
            pit[i] = float(np.interp(y, qs, levels))
    return np.clip(pit, 0.0, 1.0)


def pit_uniformity_pvalue(pit: np.ndarray, bins: int = 10) -> float:
    """Chi-square goodness-of-fit that PIT ~ Uniform(0,1).

    Low p-value => distribution is miscalibrated (intervals too wide/narrow or
    biased). This drives the gate's `calibration_ok`."""
    if len(pit) < bins * 5:
        # too few points to test reliably; treat as 'cannot reject' rather than
        # fabricate confidence
        return 1.0
    counts, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    expected = np.full(bins, len(pit) / bins)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    return float(stats.chi2.sf(chi2, df=bins - 1))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray,
                      upper: np.ndarray) -> float:
    """Empirical fraction of realized values inside [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# ---------------------------------------------------------------------------
# Inferential model comparison
# ---------------------------------------------------------------------------

def diebold_mariano(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    *, horizon: int = 1, loss: str = "squared"
                    ) -> Tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy between models A and B.

    H0: E[loss(A) - loss(B)] = 0. Returns (DM_stat, two_sided_pvalue).
    Positive stat => A has HIGHER loss (B is better).

    Uses a Newey-West / Bartlett estimate of the long-run variance to account for
    autocorrelation of the loss differential up to horizon-1 lags, then applies
    the Harvey-Leybourne-Newbold (1997) small-sample correction and a Student-t
    reference distribution. This is the honest way to rank the model ladder.
    """
    e_a = y_true - pred_a
    e_b = y_true - pred_b
    if loss == "squared":
        d = e_a ** 2 - e_b ** 2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = len(d)
    if n < 8:
        return 0.0, 1.0
    d_bar = float(np.mean(d))

    # long-run variance with Bartlett kernel, truncation = horizon-1
    gamma0 = float(np.mean((d - d_bar) ** 2))
    lrv = gamma0
    for lag in range(1, horizon):
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        w = 1.0 - lag / horizon
        lrv += 2.0 * w * cov
    if lrv <= 1e-24:
        return 0.0, 1.0

    dm = d_bar / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold small-sample correction
    k = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_corrected = dm * k
    pval = float(2.0 * stats.t.sf(np.abs(dm_corrected), df=n - 1))
    return float(dm_corrected), pval


# ---------------------------------------------------------------------------
# Bundled evaluation for one model over aligned one-step forecasts
# ---------------------------------------------------------------------------

@dataclass
class ScoreCard:
    n: int
    mase: float
    r2_oos: float
    directional_acc: float
    directional_pvalue: float
    crps: float
    pit_pvalue: float
    coverage_90: float
    forecast_std_ratio: float   # std(pred) / std(realized); ~0 => degenerate

    @property
    def degenerate(self) -> bool:
        """A forecast is degenerate if its point predictions barely vary relative
        to reality -- it has collapsed to the unconditional mean and carries no
        conditional signal even if MASE looks fine (Probe 1)."""
        return self.forecast_std_ratio < 0.05

    @property
    def calibrated(self) -> bool:
        return self.pit_pvalue >= 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n, "mase": self.mase, "r2_oos": self.r2_oos,
            "directional_acc": self.directional_acc,
            "directional_pvalue": self.directional_pvalue,
            "crps": self.crps, "pit_pvalue": self.pit_pvalue,
            "coverage_90": self.coverage_90,
            "forecast_std_ratio": self.forecast_std_ratio,
        }


def score(y_true: np.ndarray, mean_pred: np.ndarray,
          quantiles: Dict[float, np.ndarray], *,
          naive_pred: np.ndarray, naive_mase_scale: float) -> ScoreCard:
    """Compute the full scorecard for one model's aligned one-step forecasts."""
    y_true = np.asarray(y_true, float)
    mean_pred = np.asarray(mean_pred, float)
    da, dp, _ = directional_accuracy(y_true, mean_pred)
    lo, hi = quantiles[0.05], quantiles[0.95]
    pit = pit_values(y_true, quantiles)
    std_true = float(np.std(y_true)) or 1e-12
    return ScoreCard(
        n=len(y_true),
        mase=mase(y_true, mean_pred, naive_mase_scale),
        r2_oos=r2_oos(y_true, mean_pred, naive_pred),
        directional_acc=da,
        directional_pvalue=dp,
        crps=crps_from_quantiles(y_true, quantiles),
        pit_pvalue=pit_uniformity_pvalue(pit),
        coverage_90=interval_coverage(y_true, lo, hi),
        forecast_std_ratio=float(np.std(mean_pred)) / std_true,
    )
