"""
Walk-forward backtest engine -- the shared harness both planes read from.

For each model in the ladder we roll through walk-forward splits, fit on the
training window (refitting stateful models periodically for speed), forecast the
horizon from the origin context, and record the full predictive distribution plus
the realized value. The result is:

  * a FORECAST ARCHIVE (tidy, per origin per model) -- the durable record the
    statistical plane queries and the gate replays. This is what makes the system
    'for statistical purposes': predictions + outcomes are never thrown away.
  * per-model SCORECARDS (metrics.ScoreCard).
  * DM-tested LADDER READINGS vs naive (contracts.LadderReading).

Leakage controls: training data ends at `origin` (= train_stop-1); the embargo gap
sits between origin and the first test point; the MASE scale and every model
parameter are computed from train only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import metrics
from .contracts import ForecastResult, LadderReading, QUANTILE_LEVELS
from .models.base import Forecaster, SkipModel
from .windows import Split, walk_forward


@dataclass
class ArchiveRow:
    """One realized forecast: everything needed to score, calibrate, or replay."""
    model: str
    rung: int
    origin_index: int
    step: int                 # 1..horizon
    target_index: int
    y_true: float
    mean: float
    quantiles: Dict[float, float]
    latency_ms: float


@dataclass
class ModelRun:
    model: str
    rung: int
    scorecard: metrics.ScoreCard
    per_step_r2: Dict[int, float]
    mean_latency_ms: float
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class BacktestResult:
    instrument: str
    horizon: int
    archive: List[ArchiveRow] = field(default_factory=list)
    runs: Dict[str, ModelRun] = field(default_factory=dict)
    ladder: List[LadderReading] = field(default_factory=list)
    splits: int = 0

    def archive_frame(self):
        import pandas as pd
        rows = []
        for r in self.archive:
            base = {"model": r.model, "rung": r.rung, "origin": r.origin_index,
                    "step": r.step, "target_index": r.target_index,
                    "y_true": r.y_true, "mean": r.mean, "latency_ms": r.latency_ms}
            for l, v in r.quantiles.items():
                base[f"q{l}"] = v
            rows.append(base)
        return pd.DataFrame(rows)

    def r2_by_horizon(self, model: str) -> Dict[int, float]:
        return self.runs[model].per_step_r2 if model in self.runs else {}


def _predictions_for_step(archive: List[ArchiveRow], model: str, step: int):
    rows = [r for r in archive if r.model == model and r.step == step]
    rows.sort(key=lambda r: r.origin_index)
    y = np.array([r.y_true for r in rows])
    mean = np.array([r.mean for r in rows])
    q = {float(l): np.array([r.quantiles[float(l)] for r in rows])
         for l in QUANTILE_LEVELS}
    return y, mean, q


def run_backtest(
    returns: np.ndarray,
    ladder: Sequence[Forecaster],
    *,
    instrument: str = "SERIES",
    horizon: int = 1,
    min_train: int = 252,
    step: int = 1,
    refit_every: int = 21,
    mode: str = "expanding",
    window: Optional[int] = None,
    verbose: bool = False,
) -> BacktestResult:
    """Roll the full ladder through walk-forward validation.

    `refit_every` controls how often stateful models (AR, GARCH) re-estimate
    parameters; between refits they condition on fresh context but reuse fitted
    coefficients. Stateless models (naive, foundation) ignore it. This is the
    accuracy/compute trade-off; set 1 for strict per-origin refitting.
    """
    returns = np.asarray(returns, float)
    n = len(returns)
    splits: List[Split] = list(walk_forward(
        n, horizon=horizon, min_train=min_train, step=step,
        mode=mode, window=window))
    result = BacktestResult(instrument=instrument, horizon=horizon,
                            splits=len(splits))
    if not splits:
        raise ValueError("no walk-forward splits; series too short for min_train")

    # naive scale for MASE, computed once on the first training window (a stable,
    # leak-free denominator; using the very first train avoids look-ahead).
    first_train = returns[splits[0].train_slice]
    mase_scale = metrics.naive_scale(first_train, season=1)

    for model in ladder:
        latencies: List[float] = []
        last_fit_at = -10 ** 9
        skipped = False
        try:
            for si, sp in enumerate(splits):
                train = returns[sp.train_slice]
                context = returns[:sp.origin + 1]
                need_refit = (model.stateful and (sp.origin - last_fit_at) >= refit_every) \
                    or (not model.stateful) or last_fit_at < 0
                if need_refit:
                    model.fit(train)
                    last_fit_at = sp.origin
                t0 = time.perf_counter()
                fc: ForecastResult = model.predict(context, horizon)
                latency = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency)
                for h in range(horizon):
                    ti = sp.test_start + h
                    result.archive.append(ArchiveRow(
                        model=model.name, rung=model.rung,
                        origin_index=sp.origin, step=h + 1, target_index=ti,
                        y_true=float(returns[ti]), mean=float(fc.mean[h]),
                        quantiles={float(l): float(fc.quantiles[float(l)][h])
                                   for l in QUANTILE_LEVELS},
                        latency_ms=latency,
                    ))
        except SkipModel as e:
            skipped = True
            result.runs[model.name] = ModelRun(
                model=model.name, rung=model.rung, scorecard=None,  # type: ignore
                per_step_r2={}, mean_latency_ms=0.0, skipped=True,
                skip_reason=str(e))
            if verbose:
                print(f"[skip] {model.name}: {e}")
            continue

        # score step-1 (headline) and record per-step R2
        y1, m1, q1 = _predictions_for_step(result.archive, model.name, step=1)
        naive_pred1 = np.zeros_like(y1)  # random walk in returns -> 0
        sc = metrics.score(y1, m1, q1, naive_pred=naive_pred1,
                           naive_mase_scale=mase_scale)
        per_step_r2 = {}
        for h in range(1, horizon + 1):
            yh, mh, _ = _predictions_for_step(result.archive, model.name, step=h)
            per_step_r2[h] = metrics.r2_oos(yh, mh, np.zeros_like(yh))
        result.runs[model.name] = ModelRun(
            model=model.name, rung=model.rung, scorecard=sc,
            per_step_r2=per_step_r2,
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0)
        if verbose:
            print(f"[ok]   {model.name:<12} MASE={sc.mase:.3f} "
                  f"R2oos={sc.r2_oos:+.4f} dir={sc.directional_acc:.3f} "
                  f"deg={sc.degenerate} cal={sc.calibrated} "
                  f"{sc.mean_latency_ms if hasattr(sc,'mean_latency_ms') else ''}")

    result.ladder = _build_ladder(result, mase_scale)
    return result


def _build_ladder(result: BacktestResult, mase_scale: float) -> List[LadderReading]:
    """DM-test every non-naive rung against naive and assemble LadderReadings."""
    if "naive-rw" not in result.runs:
        return []
    yN, mN, _ = _predictions_for_step(result.archive, "naive-rw", step=1)
    readings: List[LadderReading] = []
    for name, run in sorted(result.runs.items(), key=lambda kv: kv[1].rung):
        if run.skipped or run.scorecard is None:
            continue
        y, m, _ = _predictions_for_step(result.archive, name, step=1)
        if name == "naive-rw":
            dm_stat, dm_p = 0.0, 1.0
        else:
            # align lengths (all step-1 arrays share origins for a given horizon)
            k = min(len(y), len(yN))
            dm_stat, dm_p = metrics.diebold_mariano(
                y[:k], m[:k], mN[:k], horizon=result.horizon, loss="squared")
            # DM sign convention: positive => first arg (naive baseline pred mN?) ...
            # here A=model pred m, B=naive pred mN, positive stat => model worse.
        sc = run.scorecard
        readings.append(LadderReading(
            rung=run.rung, model=name, mase=sc.mase, r2_oos=sc.r2_oos,
            dm_stat_vs_naive=dm_stat, dm_pvalue_vs_naive=dm_p,
            directional_acc=sc.directional_acc,
            directional_pvalue=sc.directional_pvalue,
            crps=sc.crps))
    return readings
