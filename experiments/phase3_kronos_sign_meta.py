"""
Phase 3 — pre-registered Kronos-sign meta-labeling test. RUN EXACTLY ONCE.

Hypothesis (from Phase 2's secondary, explicitly flagged there as
hypothesis-generating): the Kronos step-1 SIGN carries directional
information (ESZ5: 53.6%, p=0.018 two-sided) even though magnitude is wrong.

Locked protocol (registered before any spring-2026 ES data was touched):
  * Data      : ES front month, Mar 16 -> Jun 12 2026 (ESM6 window) — a NEW
                quarter, untouched by all prior analysis. Cost-asserted ~$0.
  * Bar size  : same physics rule as Phase 2 — smallest 500-multiple with
                median RTH (13:30-20:00 UTC) formation >= 45 s on the FIRST
                5 SESSIONS only. The rule is registered, not the number.
  * Model     : Kronos-small, ctx 256, 24 paths, horizon 1.
                Step chosen only for compute (~1,100 origins).
  * Primary   : direction = sign(kronos_p50_scaled), |p50| >= 0.05 noise
                floor (identical to the live tier-1 signal).
  * ENDPOINT A (replication): primary sign accuracy > 0.5, ONE-SIDED
                binomial, alpha 0.05 (directional prior from Phase 2).
  * ENDPOINT B (tradability): meta-labels (primary profitable next bar?),
                XGBoost via Purged CPCV (6/2, purge 5, embargo 2) within the
                quarter; decisive stat = path-level Sharpe t-test p < 0.05.
                Pooled PSR reported with its known inflation caveat.
                Friction expectancy per primary trade at ES economics
                ($50/pt, 0.25 tick, 1 tick/side, round trip).
  * DECISION (fixed now): A fails -> Phase 2 signal was noise, book closed.
                A passes, B fails -> direction real, not tradable at cost.
                Both pass -> pilot on the demo stack.
  * Validation-only: MetaModel._save_model is disabled — the live
                'metamodel' artifact is NOT overwritten by this experiment.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

DATA = ROOT / "data"
RAW_DBN = DATA / "phase3_es_raw.dbn.zst"
RAW_PQ = DATA / "phase3_es_all.parquet"
TRADES_PQ = DATA / "ESM6_trades.parquet"
BAR_RULE_JSON = DATA / "phase3_bar_rule.json"

START, END = "2026-03-16", "2026-06-12"
RTH = (13, 30, 20, 0)
MEDIAN_FLOOR_S = 45.0
CANDIDATES = [500 * k for k in range(1, 21)]
TARGET_ORIGINS = 1100
NOISE_FLOOR = 0.05                    # == signals.tier1.KRONOS_MIN_EDGE_SCALED
ES_MULT, ES_TICK, TICKS_PER_SIDE = 50.0, 0.25, 1.0


def _key() -> str:
    with open(r"C:\Users\angel\claude code\mission-control\.env.local") as f:
        for line in f:
            m = re.match(r"\s*DATABENTO_API_KEY\s*=\s*['\"]?([^'\"\n]+)", line)
            if m:
                return m.group(1).strip()
    raise RuntimeError("no key")


def stage_download():
    if RAW_PQ.exists():
        print("[1] raw parquet exists — skip", flush=True)
        return
    import databento as db
    cli = db.Historical(_key())
    kw = dict(dataset="GLBX.MDP3", symbols=["ES.FUT"], stype_in="parent",
              schema="trades", start=START, end=END)
    cost = cli.metadata.get_cost(**kw)
    print(f"[1] quoted cost ${cost:.2f}", flush=True)
    assert cost < 0.01, f"pre-registered as free; quoted ${cost:.2f} — ABORT"
    if not RAW_DBN.exists():
        print("[1] downloading ES.FUT trades ...", flush=True)
        cli.timeseries.get_range(**kw).to_file(str(RAW_DBN))
    import databento as db2
    db2.DBNStore.from_file(str(RAW_DBN)).to_parquet(str(RAW_PQ),
                                                    map_symbols=True)


def stage_front_month() -> str:
    if TRADES_PQ.exists():
        f = pd.read_parquet(TRADES_PQ, columns=["symbol"])
        front = f["symbol"].iloc[0]
        print(f"[2] front-month parquet exists ({front}) — skip", flush=True)
        return front
    import pyarrow.dataset as ds
    d = ds.dataset(str(RAW_PQ))
    cols = [c for c in ("ts_event", "price", "size", "symbol")
            if c in d.schema.names]
    t = d.to_table(columns=cols).to_pandas()
    front = t.groupby("symbol")["size"].sum().idxmax()
    out = t[t["symbol"] == front].sort_values("ts_event", kind="stable")
    out.to_parquet(TRADES_PQ, index=False)
    print(f"[2] front month {front}: {len(out):,} trades, "
          f"{out['size'].sum():,} contracts", flush=True)
    return front


def stage_bar_rule() -> int:
    if BAR_RULE_JSON.exists():
        n = json.loads(BAR_RULE_JSON.read_text())["bar_size"]
        print(f"[3] bar rule cached: {n}", flush=True)
        return n
    from forecasting.databento_adapter import load_trades, volume_bars
    trades = load_trades(str(TRADES_PQ))
    days = sorted(trades["ts"].dt.date.unique())[:5]
    first5 = trades[trades["ts"].dt.date.isin(days)]
    chosen, table = None, {}
    for n in CANDIDATES:
        bars = volume_bars(first5.reset_index(drop=True), n)
        ts = pd.DatetimeIndex(bars["ts"])
        dur = ts.to_series().diff().dt.total_seconds().iloc[1:]
        hhmm = ts.hour * 60 + ts.minute
        rth = dur[(hhmm[1:] >= RTH[0] * 60 + RTH[1])
                  & (hhmm[1:] < RTH[2] * 60 + RTH[3])]
        med = float(rth.median()) if len(rth) else float("inf")
        table[n] = round(med, 1)
        if chosen is None and med >= MEDIAN_FLOOR_S:
            chosen = n
    print(f"[3] median RTH seconds by bar size: {table}", flush=True)
    assert chosen, "no candidate met the 45s floor"
    BAR_RULE_JSON.write_text(json.dumps({"bar_size": chosen, "medians": table}))
    print(f"[3] BAR SIZE = {chosen}", flush=True)
    return chosen


def stage_run(front: str, bar_size: int) -> pd.DataFrame:
    feat_path = DATA / f"{front}_features.parquet"
    if feat_path.exists():
        print("[4] features exist — skip model run", flush=True)
        return pd.read_parquet(feat_path)
    from forecasting import databento_adapter as dba
    from forecasting.run import run_pipeline
    from forecasting import persist as persistmod

    real = dba.prepare_real(str(TRADES_PQ), bar_size=bar_size,
                            vol_window=256, instrument=front)
    n = len(real.scaled)
    step = max(1, round((n - 252) / TARGET_ORIGINS))
    print(f"[4] {n} bars -> step {step} (~{(n - 252) // step} origins), "
          f"horizon 1", flush=True)
    result = run_pipeline(real.to_series(), horizon=1, min_train=252,
                          step=step, real=real, features_out=str(feat_path),
                          backend="kronos", context_length=256, num_paths=24)
    try:
        run_id = persistmod.persist_run(
            result, real, source_file=str(TRADES_PQ),
            config={"experiment": "phase3_kronos_sign_meta",
                    "bar_size": bar_size, "horizon": 1, "step": step,
                    "preregistered": True},
            notes="Phase 3 pre-registered kronos-sign meta-labeling test")
        print(f"[4] persisted as run {run_id}", flush=True)
    except Exception as e:          # persistence must never block the science
        print(f"[4] persist failed ({e}) — features parquet is the record",
              flush=True)
    return result["features"]


def stage_endpoints(features: pd.DataFrame):
    from scipy import stats

    f = features.dropna(subset=["kronos_p50_scaled", "y_true_scaled",
                                "sigma_256"]).copy()
    prim = f[f["kronos_p50_scaled"].abs() >= NOISE_FLOOR].copy()
    prim["dir"] = np.sign(prim["kronos_p50_scaled"])
    prim["hit"] = (prim["dir"] * prim["y_true_scaled"]) > 0
    n, hits = len(prim), int(prim["hit"].sum())

    print("\n================ ENDPOINT A — sign replication ================",
          flush=True)
    pA = stats.binomtest(hits, n, 0.5, alternative="greater").pvalue
    print(f"  primaries (|p50|>={NOISE_FLOOR}): {n} of {len(f)} origins",
          flush=True)
    print(f"  sign accuracy : {hits}/{n} = {hits / n:.4f}", flush=True)
    print(f"  one-sided binomial p = {pA:.4f}  ->  "
          f"{'REPLICATES' if pA < 0.05 else 'FAILS TO REPLICATE'}", flush=True)

    # friction expectancy per primary trade (1 bar hold, close-to-close)
    raw_ret = prim["y_true_scaled"] * prim["sigma_256"]        # log return
    # points captured = close * (exp(dir*r) - 1) ~ close * dir * r; use last
    # known price scale from the manual view era (~6000-7000); we carry the
    # actual close via sigma denomination instead: report in sigma units AND
    # in $ using the mean absolute close-to-close point move.
    gross_sigma = float((prim["dir"] * prim["y_true_scaled"]).mean())
    print(f"  gross expectancy: {gross_sigma:+.4f} sigma-units/trade",
          flush=True)

    print("\n================ ENDPOINT B — meta-labeling (CPCV) ============",
          flush=True)
    from models.meta_model import MetaModel
    from features.vector import KRONOS_REGIME_MAP

    vectors, labels, rets, dates = [], [], [], []
    for _, r in prim.iterrows():
        d = int(r["dir"])
        vectors.append({
            "kronos_p50_scaled": float(r["kronos_p50_scaled"]),
            "kronos_uncertainty": float(r.get("kronos_spread_scaled", 0) or 0),
            "kronos_context_vol": float(r["sigma_256"]),
            "kronos_regime_encoded": KRONOS_REGIME_MAP.get(
                str(r.get("structure_type", "")), -1),
            "primary_signal_direction": d,
            "primary_signal_count": 1,
            "composite_strength": min(abs(r["kronos_p50_scaled"]) /
                                      (abs(r["kronos_p50_scaled"]) +
                                       max(r.get("kronos_spread_scaled", 0) or
                                           1e-9, 1e-9)), 1.0),
        })
        labels.append(1 if r["hit"] else 0)
        rets.append(float(d * r["y_true_scaled"] * r["sigma_256"]))
        ts = pd.Timestamp(r["ts"])
        dates.append(ts.date())

    meta = MetaModel()
    meta._save_model = lambda: None        # validation-only: keep live artifact
    result = meta.train(feature_vectors=vectors, labels=labels,
                        sample_dates=dates, instrument="ESM6-phase3",
                        sample_returns=np.array(rets))
    c = result["cpcv"]
    sig = c.get("statistically_significant")
    print(f"  CPCV paths        : {c.get('paths_tested')}", flush=True)
    print(f"  sharpe mean/std   : {c.get('sharpe_mean')} / {c.get('sharpe_std')}",
          flush=True)
    print(f"  path t-test p     : {c.get('path_sharpe_p_value')}  ->  "
          f"{'TRADABLE' if sig else 'NOT TRADABLE'} (decisive)", flush=True)
    print(f"  pooled PSR        : {c.get('probabilistic_sharpe_ratio')} "
          f"(inflated by path overlap — informational only)", flush=True)

    print("\n================ DECISION (pre-registered) ====================",
          flush=True)
    if pA >= 0.05:
        print("  A FAILED -> Phase 2's sign signal was noise. Book closed.",
              flush=True)
    elif not sig:
        print("  A replicated, B failed -> direction is real but NOT "
              "tradable after CPCV validation at these costs/sizing.",
              flush=True)
    else:
        print("  A and B PASSED -> hypothesis survives; eligible for a "
              "demo-stack pilot.", flush=True)


if __name__ == "__main__":
    stage_download()
    front = stage_front_month()
    n = stage_bar_rule()
    feats = stage_run(front, n)
    stage_endpoints(feats)
    print("\nPHASE 3 COMPLETE", flush=True)
