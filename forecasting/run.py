"""
End-to-end orchestration + CLI demo.

Runs the whole pipeline on either a synthetic oracle series (default) or a Parquet
file, and prints:
  * the model-ladder backtest (which rungs beat naive, significantly)
  * the MarketCharacterization (structure type, stationarity, regimes, memory)
  * the veto-gate validation (does gating help a reference strategy?)

Usage:
    python -m forecasting.run --synthetic garch --horizon 1
    python -m forecasting.run --parquet data/AAPL.parquet --instrument AAPL
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from . import data as datamod
from . import prep as prepmod
from . import backtest as btmod
from . import characterize as charmod
from . import gate as gatemod
from .models import default_ladder


SYNTH = {
    "rw": datamod.gen_random_walk,
    "ar1": datamod.gen_ar1_returns,
    "garch": datamod.gen_garch,
    "regime": datamod.gen_regime_switch,
}


def _regime_r2(bt: btmod.BacktestResult, char_labels, signal_model: str
               ) -> Dict[str, float]:
    """R2_oos of the primary model *within* each regime -> flags unpredictable
    regimes for the gate."""
    from .contracts import StructureType  # noqa
    out: Dict[str, float] = {}
    rows = [r for r in bt.archive if r.model == signal_model and r.step == 1]
    if not rows:
        return out
    import numpy as np
    # map regime id (at target index) -> lists
    buckets: Dict[int, list] = {}
    for r in rows:
        idx = min(r.target_index, len(char_labels) - 1)
        buckets.setdefault(int(char_labels[idx]), []).append(r)
    for rid, rs in buckets.items():
        y = np.array([r.y_true for r in rs])
        m = np.array([r.mean for r in rs])
        from .metrics import r2_oos
        out[str(rid)] = r2_oos(y, m, np.zeros_like(y))
    return out


def run_pipeline(series: datamod.Series, *, horizon: int = 1,
                 min_train: int = 252, include_foundation: bool = True,
                 step: int = 1, verbose: bool = True, real=None,
                 features_out: str = "", **foundation_kwargs) -> dict:
    prepared = prepmod.prepare(series)
    r = prepared.returns
    if verbose:
        cr = prepared.clean_report
        print(f"\n=== {series.instrument} : {len(r)} returns "
              f"(cleaned {cr.n_dropped_dupes} dupes, {cr.n_nonpositive_repaired} "
              f"bad prices; anomaly rate {prepared.flags.rate():.2%}) ===")

    ladder = default_ladder(include_foundation=include_foundation,
                            **foundation_kwargs)
    if real is not None:
        # real-data mode: hand the foundation rung the true candles + scaler
        from .models import FoundationForecaster
        for m in ladder:
            if isinstance(m, FoundationForecaster):
                m.attach_klines(real)
    bt = btmod.run_backtest(r, ladder, instrument=series.instrument,
                            horizon=horizon, min_train=min_train, step=step,
                            verbose=verbose)

    # per-horizon R2 from the best available rung for the char report
    signal_model = gatemod._signal_model(bt)
    r2_by_h = bt.r2_by_horizon(signal_model)

    # regime scores (need labels first via a quick characterize pass on labels)
    labels, names = charmod.volatility_regimes(r, n_regimes=2)
    regime_scores_raw = _regime_r2(bt, labels, signal_model)
    regime_scores = {names.get(int(k), k): v for k, v in regime_scores_raw.items()}

    char = charmod.characterize(
        series.instrument, r, ladder=bt.ladder, r2_by_horizon=r2_by_h,
        regime_scores=regime_scores,
        data_asof=str(series.index[-1].date()))

    signals = gatemod.build_signals(bt, char, prepared,
                                    horizon_of_interest=min(horizon, 1) or 1)
    gv = gatemod.validate_gate(signals, r)

    if verbose:
        print("\n" + char.summary())
        print(f"\n--- Veto gate ({len(signals)} decisions) ---")
        print(f"  gate mix   : GO {gv.go_rate:.0%} | REDUCED {gv.reduced_rate:.0%} "
              f"| VETO {gv.veto_rate:.0%}")
        print(f"  sharpe     : ungated {gv.sharpe_ungated:+.2f} -> "
              f"gated {gv.sharpe_gated:+.2f} (uplift {gv.sharpe_uplift:+.2f})")
        print(f"  veto prec/recall : {gv.veto_precision:.2f} / {gv.veto_recall:.2f}")
        if series.truth:
            print(f"\n  [oracle] ground-truth structure: "
                  f"{series.truth.get('structure_type')}  "
                  f"-> measured: {char.structure_type.value}")

    features = None
    if real is not None:
        # bifurcated outputs: stationary features for the model plane, absolute
        # price levels for the operator plane
        from . import databento_adapter as dba
        features = dba.xgboost_features(bt, char, signals, real)
        if features_out:
            features.to_parquet(features_out, index=False)
            if verbose:
                print(f"\n[features] {len(features)} rows -> {features_out}")
        if verbose:
            print(dba.manual_trading_view(bt, char, real))

    return {"prepared": prepared, "backtest": bt, "characterization": char,
            "signals": signals, "gate_validation": gv, "features": features}


def main():
    ap = argparse.ArgumentParser(description="Probabilistic forecasting pipeline")
    ap.add_argument("--synthetic", choices=list(SYNTH), default="garch")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--instrument", default=None)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--min-train", type=int, default=252)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--no-foundation", action="store_true")
    ap.add_argument("--step", type=int, default=1,
                    help="forecast every k-th origin (thins the backtest; "
                         "useful when the foundation rung is slow on CPU)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "kronos", "chronos", "timesfm"],
                    help="foundation backend; auto = kronos > chronos > timesfm")
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--num-paths", type=int, default=32,
                    help="sampled paths per origin (kronos quantile resolution)")
    ap.add_argument("--kronos-model", default="NeoQuasar/Kronos-small")
    ap.add_argument("--kronos-tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--real-data", default=None, metavar="TRADES_PARQUET",
                    help="Databento trades-schema parquet (ts/price/size); "
                         "aggregates volume bars and runs the real-data path")
    ap.add_argument("--bar-size", type=int, default=250,
                    help="contracts per volume bar (real-data mode)")
    ap.add_argument("--vol-window", type=int, default=256,
                    help="trailing bars for the sigma scaler (real-data mode)")
    ap.add_argument("--features-out", default="",
                    help="write per-origin XGBoost features parquet here "
                         "(default: <real-data stem>_features.parquet)")
    ap.add_argument("--persist", action="store_true",
                    help="land the run in the md Postgres schema (bars, "
                         "conditioned series, forecast archive, gate signals, "
                         "features) under one run_id; needs psycopg2 and "
                         "QUANT_DB_URL/DATABASE_URL (defaults to the engine's "
                         "docker-compose Postgres)")
    args = ap.parse_args()

    real = None
    if args.real_data:
        from . import databento_adapter as dba
        real = dba.prepare_real(args.real_data, bar_size=args.bar_size,
                                vol_window=args.vol_window,
                                instrument=args.instrument or "")
        series = real.to_series()
        if not args.features_out:
            stem = args.real_data.rsplit(".", 1)[0]
            args.features_out = f"{stem}_features.parquet"
        print(f"[real] {series.instrument}: {len(real.scaled)} volume bars "
              f"of ~{args.bar_size} contracts "
              f"({real.bar_ts[0]:%Y-%m-%d} -> {real.bar_ts[-1]:%Y-%m-%d})")
    elif args.parquet:
        series = datamod.load_parquet(args.parquet,
                                      args.instrument or "SERIES")
    else:
        series = SYNTH[args.synthetic](n=args.n)

    result = run_pipeline(series, horizon=args.horizon, min_train=args.min_train,
                          include_foundation=not args.no_foundation,
                          step=args.step, real=real,
                          features_out=args.features_out,
                          backend=args.backend,
                          context_length=args.context_length,
                          num_paths=args.num_paths,
                          kronos_model=args.kronos_model,
                          kronos_tokenizer=args.kronos_tokenizer)

    if args.persist:
        if real is None:
            print("[persist] skipped: --persist only applies to --real-data runs")
        else:
            from . import persist as persistmod
            run_id = persistmod.persist_run(
                result, real, source_file=args.real_data,
                config={k: v for k, v in vars(args).items()
                        if k not in ("persist",)})
            print(f"[persist] run {run_id} landed in md schema "
                  f"({persistmod.resolve_dsn().rsplit('@', 1)[-1]})")


if __name__ == "__main__":
    main()
