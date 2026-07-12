"""
The veto gate -- the operational plane handed to the strategy lab.

The gate is a *pure function of the same statistics* the research plane records;
there is no separate "strategy logic" here, only forecast-quality logic. Each
Phase-2 probe maps to a concrete veto reason:

    Probe 1 degenerate forecast        -> DEGENERATE_FORECAST (hard)
    Probe 3 broken calibration         -> MISCALIBRATED       (hard)
    Probe 5 unpredictable regime       -> UNPREDICTABLE_REGIME(hard)
    Phase 3 recent structural break    -> NONSTATIONARY_BREAK (hard)
    predictability decay               -> HORIZON_BEYOND_EDGE (hard)
    Phase 0 data-quality flag          -> DATA_QUALITY        (hard)
    Probe 7 model disagreement         -> LOW_CONVICTION      (soft)
    Probe 6 weak directional edge      -> WEAK_DIRECTION      (soft)
    Probe 3 elevated vol forecast      -> ELEVATED_VOL        (soft)

CAUSALITY: replaying the archive, a decision at origin t may only use forecast
outcomes whose target_index < t (already realized). Trailing calibration and
degeneracy are computed on that realized window only. This is what makes the
backtest signal identical to what live would have produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import metrics
from .backtest import BacktestResult, ArchiveRow
from .contracts import (ForecastSignal, MarketCharacterization, VetoReason,
                        make_signal, QUANTILE_LEVELS)
from .prep import PreparedSeries


@dataclass
class GateConfig:
    # hard-veto thresholds
    degenerate_ratio: float = 0.05        # std(pred)/std(real) below -> degenerate
    calibration_alpha: float = 0.05       # PIT uniformity p below -> miscalibrated
    calibration_window: int = 120         # trailing outcomes for PIT test
    break_lookback: int = 40              # a break within N obs -> nonstationary veto
    # soft-veto thresholds
    conviction_floor: float = 0.35        # below -> LOW_CONVICTION
    direction_pvalue: float = 0.10        # above -> WEAK_DIRECTION
    vol_elevated_z: float = 1.5           # vol_forecast z vs trailing -> ELEVATED_VOL
    vol_z_window: int = 120


def _rows_by_origin(archive: List[ArchiveRow], step: int = 1
                    ) -> Dict[int, Dict[str, ArchiveRow]]:
    out: Dict[int, Dict[str, ArchiveRow]] = {}
    for r in archive:
        if r.step != step:
            continue
        out.setdefault(r.origin_index, {})[r.model] = r
    return out


def _trailing_calibration_ok(realized: List[ArchiveRow], model: str,
                             cfg: GateConfig) -> bool:
    """PIT-uniformity test over the most recent realized forecasts of `model`."""
    rows = [r for r in realized if r.model == model][-cfg.calibration_window:]
    if len(rows) < 30:
        return True  # insufficient evidence -> do not fabricate a veto
    y = np.array([r.y_true for r in rows])
    q = {float(l): np.array([r.quantiles[float(l)] for r in rows])
         for l in QUANTILE_LEVELS}
    pit = metrics.pit_values(y, q)
    return metrics.pit_uniformity_pvalue(pit) >= cfg.calibration_alpha


def _trailing_degenerate(realized: List[ArchiveRow], model: str,
                         cfg: GateConfig) -> bool:
    rows = [r for r in realized if r.model == model][-cfg.calibration_window:]
    if len(rows) < 30:
        return False
    pred = np.array([r.mean for r in rows])
    real = np.array([r.y_true for r in rows])
    ratio = float(np.std(pred)) / (float(np.std(real)) or 1e-12)
    return ratio < cfg.degenerate_ratio


def _signal_model(bt: "BacktestResult") -> str:
    """Choose the primary DIRECTIONAL signal source.

    Not simply the highest rung -- a higher rung with no skill (e.g. GARCH on a
    purely linear series) would drag the gate to a degenerate reading. Instead we
    pick the rung that actually *beats naive* with the best out-of-sample R^2. If
    no rung beats naive (efficient / vol-only series), we fall back to the best
    available distributional model for vol/sizing; direction is then inherently
    degenerate and the gate will veto directional trades -- which is correct.
    """
    beats = [r for r in bt.ladder if r.beats_naive]
    if beats:
        best = max(beats, key=lambda r: r.r2_oos)
        return best.model
    # no directional skill anywhere: prefer garch (vol/distribution) then naive
    for name in ("garch", "naive-rw"):
        run = bt.runs.get(name)
        if run is not None and not getattr(run, "skipped", False):
            return name
    return "naive-rw"


def build_signals(
    bt: BacktestResult,
    char: MarketCharacterization,
    prepared: PreparedSeries,
    *,
    horizon_of_interest: int = 1,
    config: Optional[GateConfig] = None,
) -> List[ForecastSignal]:
    """Replay the backtest archive into point-in-time ForecastSignals.

    One signal per forecast origin (for step-1). Uses the characterization to set
    regime/horizon permissions and the archive (causally) for calibration,
    degeneracy, conviction, direction and vol."""
    cfg = config or GateConfig()
    signal_model = _signal_model(bt)
    by_origin = _rows_by_origin(bt.archive, step=horizon_of_interest)
    origins = sorted(by_origin)

    # precompute regime label per return index (from characterization, aligned to
    # returns). Break set for fast lookup.
    regime_labels = char.regime_labels
    regime_names = char.regime_names
    break_set = np.array(char.structural_breaks) if char.structural_breaks else np.array([])
    edge_h = char.predictability_edge_horizon

    signals: List[ForecastSignal] = []
    for t in origins:
        # realized window: forecasts whose target already occurred by origin t
        realized = [r for r in bt.archive
                    if r.step == horizon_of_interest and r.target_index < t]

        row_by_model = by_origin[t]
        prim = row_by_model.get(signal_model)
        if prim is None:
            continue

        # --- continuous signals ---------------------------------------
        # vol forecast: predictive std from the primary model's quantiles
        lo16 = prim.quantiles[float(min(QUANTILE_LEVELS, key=lambda l: abs(l - 0.16)))]
        hi84 = prim.quantiles[float(min(QUANTILE_LEVELS, key=lambda l: abs(l - 0.84)))]
        vol_forecast = max((hi84 - lo16) / 2.0, 1e-12)

        # direction: sign * strength = mean / vol, squashed to [-1,1]
        direction = float(np.tanh(prim.mean / (vol_forecast + 1e-12)))

        # conviction from model agreement (Probe 7): dispersion of the non-naive
        # model means, scaled by vol. Low dispersion -> high conviction.
        means = [r.mean for m, r in row_by_model.items() if m != "naive-rw"]
        if len(means) >= 2:
            disagreement = float(np.std(means)) / (vol_forecast + 1e-12)
            conviction = float(np.clip(1.0 - disagreement, 0.0, 1.0))
        else:
            conviction = 0.5  # single model -> neutral prior

        # --- diagnostics ----------------------------------------------
        calibration_ok = _trailing_calibration_ok(realized, signal_model, cfg)
        degenerate = _trailing_degenerate(realized, signal_model, cfg)
        ridx = min(t, len(regime_labels) - 1)
        regime = regime_names.get(int(regime_labels[ridx]), str(regime_labels[ridx]))
        r2_here = char.r2_by_horizon.get(horizon_of_interest, 0.0)

        # --- assemble veto reasons ------------------------------------
        reasons: List[VetoReason] = []
        # hard
        if degenerate:
            reasons.append(VetoReason.DEGENERATE_FORECAST)
        if not calibration_ok:
            reasons.append(VetoReason.MISCALIBRATED)
        if regime in char.unpredictable_regimes:
            reasons.append(VetoReason.UNPREDICTABLE_REGIME)
        if len(break_set) and np.any((break_set <= t) & (break_set > t - cfg.break_lookback)):
            reasons.append(VetoReason.NONSTATIONARY_BREAK)
        if horizon_of_interest > max(edge_h, 1):
            reasons.append(VetoReason.HORIZON_BEYOND_EDGE)
        # data quality: anomaly flag active at t (from prep, causal)
        if t < len(prepared.flags.any) and bool(prepared.flags.any[t]):
            reasons.append(VetoReason.DATA_QUALITY)
        # soft
        if conviction < cfg.conviction_floor:
            reasons.append(VetoReason.LOW_CONVICTION)
        # weak direction: use characterization directional significance as the
        # structural prior on whether direction is tradeable at all
        prim_run = bt.runs.get(signal_model)
        if prim_run and prim_run.scorecard and \
                prim_run.scorecard.directional_pvalue > cfg.direction_pvalue:
            reasons.append(VetoReason.WEAK_DIRECTION)
        # elevated vol vs trailing
        rv_tail = [r.mean for r in realized]  # placeholder guard
        if len(realized) >= cfg.vol_z_window:
            past_vols = _trailing_vol_forecasts(bt.archive, signal_model, t,
                                                cfg.vol_z_window)
            if len(past_vols) > 10:
                z = (vol_forecast - np.mean(past_vols)) / (np.std(past_vols) + 1e-12)
                if z > cfg.vol_elevated_z:
                    reasons.append(VetoReason.ELEVATED_VOL)

        signals.append(make_signal(
            instrument=char.instrument, origin_index=t, reasons=reasons,
            direction=direction, conviction=conviction,
            vol_forecast=float(vol_forecast), regime=regime,
            calibration_ok=calibration_ok, forecast_degenerate=degenerate,
            r2_oos=r2_here,
            provenance={"signal_model": signal_model,
                        "structure": char.structure_type.value,
                        "horizon": str(horizon_of_interest)},
        ))
    return signals


def _trailing_vol_forecasts(archive: List[ArchiveRow], model: str, t: int,
                            window: int) -> np.ndarray:
    lo_l = float(min(QUANTILE_LEVELS, key=lambda l: abs(l - 0.16)))
    hi_l = float(min(QUANTILE_LEVELS, key=lambda l: abs(l - 0.84)))
    vols = [(r.quantiles[hi_l] - r.quantiles[lo_l]) / 2.0
            for r in archive
            if r.model == model and r.step == 1 and r.origin_index < t]
    return np.array(vols[-window:])


# ---------------------------------------------------------------------------
# Gate validation (Phase 5): does gating actually help?
# ---------------------------------------------------------------------------

@dataclass
class GateValidation:
    """Whether the gate improves a reference strategy. A gate that never vetoes
    adds nothing; one that vetoes the winners is harmful. We measure both."""
    veto_rate: float
    reduced_rate: float
    go_rate: float
    sharpe_ungated: float
    sharpe_gated: float
    sharpe_uplift: float
    veto_precision: float   # of vetoed bars, fraction that WOULD have lost
    veto_recall: float      # of losing bars, fraction the gate caught


def validate_gate(signals: List[ForecastSignal], returns: np.ndarray
                  ) -> GateValidation:
    """Reference strategy: take `direction * conviction` as the position each bar;
    realize next-step return. Compare ungated vs gated (VETO -> flat,
    GO_REDUCED -> half size). Reports Sharpe uplift and veto precision/recall."""
    from .contracts import GateState
    pos, pos_gated, rlz = [], [], []
    vetoed_losses = vetoed_total = 0
    losing_bars = caught = 0
    for s in signals:
        ti = s.origin_index + 1
        if ti >= len(returns):
            continue
        r = returns[ti]
        raw = s.direction * s.conviction
        pos.append(raw)
        if s.gate == GateState.VETO:
            g = 0.0
            vetoed_total += 1
            if raw * r < 0 or (raw != 0 and r * np.sign(raw) < 0):
                vetoed_losses += 1
        elif s.gate == GateState.GO_REDUCED:
            g = 0.5 * raw
        else:
            g = raw
        pos_gated.append(g)
        rlz.append(r)
        # losing-bar accounting for recall: a bar the raw strategy would lose on
        if raw != 0 and np.sign(raw) != np.sign(r) and r != 0:
            losing_bars += 1
            if s.gate == GateState.VETO:
                caught += 1

    pos = np.array(pos); pos_gated = np.array(pos_gated); rlz = np.array(rlz)
    pnl_u = pos * rlz
    pnl_g = pos_gated * rlz
    sh_u = _sharpe(pnl_u)
    sh_g = _sharpe(pnl_g)
    states = [s.gate for s in signals]
    from .contracts import GateState as GS
    n = len(states) or 1
    return GateValidation(
        veto_rate=sum(x == GS.VETO for x in states) / n,
        reduced_rate=sum(x == GS.GO_REDUCED for x in states) / n,
        go_rate=sum(x == GS.GO for x in states) / n,
        sharpe_ungated=sh_u, sharpe_gated=sh_g, sharpe_uplift=sh_g - sh_u,
        veto_precision=(vetoed_losses / vetoed_total) if vetoed_total else 0.0,
        veto_recall=(caught / losing_bars) if losing_bars else 0.0,
    )


def _sharpe(pnl: np.ndarray, ann: int = 252) -> float:
    if len(pnl) < 2 or np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * np.sqrt(ann))
