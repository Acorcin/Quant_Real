"""
Typed contracts shared across both output planes.

These are the *interfaces* the strategy lab and the research layer depend on.
Everything upstream exists to populate them. Keeping them as frozen dataclasses
(with explicit validation) means a malformed forecast fails loudly at the
boundary instead of silently poisoning a backtest or a live decision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Forecast primitive (produced once by the core, consumed by both planes)
# ---------------------------------------------------------------------------

# Canonical quantile grid. Symmetric around the median so interval coverage is
# trivial to read off (e.g. 0.05/0.95 -> nominal 90% interval). Kept fixed so
# every model in the ladder emits comparable distributions.
QUANTILE_LEVELS: Tuple[float, ...] = (
    0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99,
)


@dataclass(frozen=True)
class ForecastResult:
    """A single probabilistic forecast for one origin time.

    All arrays are indexed by forecast step h = 1..horizon. The forecast is a
    *distribution* per step, represented by a mean and a set of quantiles. The
    veto gate collapses this to a decision; the statistical plane keeps all of it.

    Attributes
    ----------
    model : str
        Producing model name (e.g. "chronos-bolt-base").
    rung : int
        Position on the model-complexity ladder (0=naive .. 3=foundation).
    origin_index : int
        Integer position of the forecast origin in the source series.
    mean : np.ndarray, shape (horizon,)
        Point forecast per step (predictive mean, in target units = log-returns).
    quantiles : dict[float, np.ndarray]
        Map level -> array of shape (horizon,). Must contain QUANTILE_LEVELS and
        be monotone non-decreasing in the level for every step.
    target : str
        What is being forecast, e.g. "log_return" or "log_price".
    """

    model: str
    rung: int
    origin_index: int
    mean: np.ndarray
    quantiles: Dict[float, np.ndarray]
    target: str = "log_return"

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        object.__setattr__(self, "mean", mean)
        h = mean.shape[0]
        if h == 0:
            raise ValueError("ForecastResult.mean must be non-empty")
        clean: Dict[float, np.ndarray] = {}
        for lvl in QUANTILE_LEVELS:
            if lvl not in self.quantiles:
                raise ValueError(f"{self.model}: missing quantile level {lvl}")
            arr = np.asarray(self.quantiles[lvl], dtype=float)
            if arr.shape != (h,):
                raise ValueError(
                    f"{self.model}: quantile {lvl} has shape {arr.shape}, expected {(h,)}"
                )
            clean[lvl] = arr
        object.__setattr__(self, "quantiles", clean)
        # Enforce monotonicity of the quantile function per step. A crossed
        # quantile (q_0.9 < q_0.5) is a model bug that silently corrupts CRPS and
        # coverage, so we sort it in place and record nothing was assumed.
        self._enforce_monotone()

    def _enforce_monotone(self) -> None:
        levels = np.array(QUANTILE_LEVELS)
        stacked = np.stack([self.quantiles[l] for l in levels], axis=0)  # (Q, H)
        stacked = np.sort(stacked, axis=0)  # non-crossing per step
        for i, l in enumerate(levels):
            object.__setattr__(
                self, "quantiles", {**self.quantiles, float(l): stacked[i]}
            )

    @property
    def horizon(self) -> int:
        return int(self.mean.shape[0])

    def median(self) -> np.ndarray:
        return self.quantiles[0.50]

    def interval(self, coverage: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) arrays for a nominal central `coverage` band."""
        alpha = (1.0 - coverage) / 2.0
        lo = _closest_level(alpha)
        hi = _closest_level(1.0 - alpha)
        return self.quantiles[lo], self.quantiles[hi]

    def predictive_std(self) -> np.ndarray:
        """Gaussian-equivalent sigma implied by the central 68% (0.16..0.84)
        interval, approximated from the nearest available quantiles. Used as the
        gate's volatility forecast when a parametric sigma is not exposed."""
        lo, hi = self.quantiles[_closest_level(0.16)], self.quantiles[_closest_level(0.84)]
        return np.maximum((hi - lo) / 2.0, 1e-12)


def _closest_level(target: float) -> float:
    return float(min(QUANTILE_LEVELS, key=lambda l: abs(l - target)))


# ---------------------------------------------------------------------------
# Operational plane: the veto gate object handed to the strategy lab
# ---------------------------------------------------------------------------

class GateState(str, Enum):
    GO = "GO"                # trade freely; no quality flag fired
    GO_REDUCED = "GO_REDUCED"  # tradeable but shrink size (soft flags)
    VETO = "VETO"            # do not trade; a hard quality flag fired


class VetoReason(str, Enum):
    # Hard vetoes -> VETO
    DEGENERATE_FORECAST = "degenerate_forecast"          # no conditional signal
    MISCALIBRATED = "miscalibrated"                       # interval coverage broken
    UNPREDICTABLE_REGIME = "unpredictable_regime"         # regime measured as efficient
    DATA_QUALITY = "data_quality"                         # upstream bad data
    NONSTATIONARY_BREAK = "nonstationary_break"           # recent structural break
    HORIZON_BEYOND_EDGE = "horizon_beyond_edge"           # past predictability decay
    # Soft vetoes -> GO_REDUCED
    LOW_CONVICTION = "low_conviction"                     # models disagree
    WEAK_DIRECTION = "weak_direction"                     # directional edge insignificant
    ELEVATED_VOL = "elevated_vol"                         # vol forecast high


_HARD_REASONS = frozenset({
    VetoReason.DEGENERATE_FORECAST,
    VetoReason.MISCALIBRATED,
    VetoReason.UNPREDICTABLE_REGIME,
    VetoReason.DATA_QUALITY,
    VetoReason.NONSTATIONARY_BREAK,
    VetoReason.HORIZON_BEYOND_EDGE,
})


@dataclass(frozen=True)
class ForecastSignal:
    """Point-in-time decision object. Emitted per (instrument, timestamp).

    `gate` is a hard switch for the strategy lab. `signals` are continuous inputs
    to the lab's own position sizing. `diagnostics`/`provenance` are for audit and
    must never be required to reconstruct the gate (the gate is a pure function of
    the underlying statistics, recorded here for traceability)."""

    instrument: str
    origin_index: int
    gate: GateState
    veto_reasons: List[VetoReason]
    # signals -> for sizing
    direction: float          # in [-1, 1], sign * strength of directional edge
    conviction: float         # in [0, 1], from model agreement
    vol_forecast: float       # sigma-hat of next-step return
    # diagnostics -> insight only
    regime: str
    calibration_ok: bool
    forecast_degenerate: bool
    r2_oos: float             # out-of-sample R^2 vs naive at this horizon/regime
    # provenance
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-1.0 <= self.direction <= 1.0):
            raise ValueError(f"direction {self.direction} out of [-1,1]")
        if not (0.0 <= self.conviction <= 1.0):
            raise ValueError(f"conviction {self.conviction} out of [0,1]")
        if self.vol_forecast < 0:
            raise ValueError("vol_forecast must be >= 0")
        # Cross-check that the declared gate matches the reasons. This catches a
        # composer bug where the state and the reasons drift apart.
        expected = _gate_from_reasons(self.veto_reasons)
        if expected != self.gate:
            raise ValueError(
                f"gate {self.gate} inconsistent with reasons -> expected {expected}"
            )

    def to_json(self) -> str:
        d = asdict(self)
        d["gate"] = self.gate.value
        d["veto_reasons"] = [r.value for r in self.veto_reasons]
        return json.dumps(d, sort_keys=True)


def _gate_from_reasons(reasons: Sequence[VetoReason]) -> GateState:
    if any(r in _HARD_REASONS for r in reasons):
        return GateState.VETO
    if len(reasons) > 0:
        return GateState.GO_REDUCED
    return GateState.GO


def make_signal(
    *,
    instrument: str,
    origin_index: int,
    reasons: Sequence[VetoReason],
    direction: float,
    conviction: float,
    vol_forecast: float,
    regime: str,
    calibration_ok: bool,
    forecast_degenerate: bool,
    r2_oos: float,
    provenance: Optional[Dict[str, str]] = None,
) -> ForecastSignal:
    """Single constructor so the gate state is *always* derived from the reasons,
    never set independently. This is the invariant the __post_init__ check
    guards."""
    reasons = list(dict.fromkeys(reasons))  # de-dup, preserve order
    return ForecastSignal(
        instrument=instrument,
        origin_index=origin_index,
        gate=_gate_from_reasons(reasons),
        veto_reasons=reasons,
        direction=float(np.clip(direction, -1.0, 1.0)),
        conviction=float(np.clip(conviction, 0.0, 1.0)),
        vol_forecast=float(max(vol_forecast, 0.0)),
        regime=regime,
        calibration_ok=bool(calibration_ok),
        forecast_degenerate=bool(forecast_degenerate),
        r2_oos=float(r2_oos),
        provenance=dict(provenance or {}),
    )


# ---------------------------------------------------------------------------
# Statistical plane: the durable structural map of the data
# ---------------------------------------------------------------------------

class StructureType(str, Enum):
    EFFICIENT = "efficient"      # nothing beats naive -> unpredictable
    LINEAR = "linear"            # linear AR beats naive
    VOL_ONLY = "vol_only"        # volatility predictable, direction not
    NONLINEAR = "nonlinear"      # foundation model beats linear + vol


@dataclass
class LadderReading:
    """One rung of the model ladder: how much this model beats naive, and whether
    that improvement is statistically significant (Diebold-Mariano vs naive)."""

    rung: int
    model: str
    mase: float                  # <1 beats naive on scaled abs error
    r2_oos: float                # >0 beats naive on squared error
    dm_stat_vs_naive: float      # Diebold-Mariano statistic
    dm_pvalue_vs_naive: float    # two-sided p-value
    directional_acc: float       # fraction of correct sign predictions
    directional_pvalue: float    # binomial test vs 0.5
    crps: float = float("nan")   # mean CRPS of the predictive distribution

    @property
    def beats_naive(self) -> bool:
        """Significant improvement: better R2_oos AND DM rejects equal accuracy."""
        return self.r2_oos > 0.0 and self.dm_pvalue_vs_naive < 0.05


@dataclass
class MarketCharacterization:
    """The Phase-3 deliverable: a versioned structural map of one instrument.

    This is 'forecasting as a measurement instrument' -- the answer to *what is
    this series made of*, which in turn parameterizes the veto gate."""

    instrument: str
    data_asof: str
    structure_type: StructureType
    ladder: List[LadderReading]
    # predictability
    r2_by_horizon: Dict[int, float]              # horizon -> R2_oos
    predictability_edge_horizon: int             # last horizon with R2_oos>0
    # stationarity
    adf_pvalue: float
    kpss_pvalue: float
    stationary: bool
    structural_breaks: List[int]                 # indices of detected breaks
    # regimes
    regime_labels: np.ndarray                    # per-observation regime id
    regime_names: Dict[int, str]
    current_regime: str
    unpredictable_regimes: List[str]             # regimes measured as efficient
    # memory / information content
    hurst: float
    ljung_box_returns_pvalue: float              # H0: returns are white noise
    ljung_box_sq_returns_pvalue: float           # H0: squared returns white noise
    permutation_entropy: float                   # 0=perfectly predictable, 1=random
    provenance: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Market characterization: {self.instrument} (asof {self.data_asof})",
            f"  structure type      : {self.structure_type.value}",
            f"  stationary          : {self.stationary} "
            f"(ADF p={self.adf_pvalue:.3g}, KPSS p={self.kpss_pvalue:.3g})",
            f"  predictable horizon : up to {self.predictability_edge_horizon} step(s)",
            f"  hurst exponent      : {self.hurst:.3f} "
            f"({'mean-reverting' if self.hurst < 0.45 else 'trending' if self.hurst > 0.55 else 'random'})",
            f"  permutation entropy : {self.permutation_entropy:.3f}",
            f"  current regime      : {self.current_regime}",
            f"  unpredictable       : {', '.join(self.unpredictable_regimes) or 'none'}",
            f"  structural breaks   : {len(self.structural_breaks)}",
            "  model ladder:",
        ]
        for r in self.ladder:
            mark = "beats" if r.beats_naive else "  ~  "
            lines.append(
                f"    [{r.rung}] {r.model:<20} MASE={r.mase:.3f} "
                f"R2oos={r.r2_oos:+.4f} DMp={r.dm_pvalue_vs_naive:.3g} "
                f"dir={r.directional_acc:.3f} {mark} naive"
            )
        return "\n".join(lines)
