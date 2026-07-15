"""
Phase 2 — pre-registered ES execution-reality test. RUN EXACTLY ONCE.

Locked protocol (registered before any ES data was touched):
  * Instrument : ES front month, trades schema, Sep 15 -> Dec 12 2025 (ESZ5
                 front window; different quarter than the M6E baseline).
  * Bar size   : smallest multiple of 500 contracts whose MEDIAN RTH
                 (13:30-20:00 UTC) bar formation time is >= 45 s, measured on
                 the FIRST 5 SESSIONS ONLY — a tape-physics rule fixed a
                 priori (our CPU inference is ~4-6 s; bars must form slower
                 than the decision loop), never a backtest-tuned number.
  * Model      : Kronos-small, context 256, 24 paths — frozen defaults.
  * Horizon    : 4 bars. Step chosen ONLY for compute (~1,000-1,200 origins).
  * Output     : step-1 ladder (MASE/R2oos/DMp), per-step R2 decay to h=4,
                 step-4 DM vs naive from the SAME archive (no second run),
                 structure_type, gate stats. Persisted to md under one run_id.

Stages checkpoint to disk so a crash resumes without re-downloading;
none of the checkpoints feed parameter choices.
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

DATA = ROOT / "data"
RAW_DBN = DATA / "phase2_es_raw.dbn.zst"
RAW_PQ = DATA / "phase2_es_all.parquet"
TRADES_PQ = DATA / "ESZ5_trades.parquet"
BAR_RULE_JSON = DATA / "phase2_bar_rule.json"

START, END = "2025-09-15", "2025-12-12"
RTH = (13, 30, 20, 0)                    # 13:30 -> 20:00 UTC
MEDIAN_FLOOR_S = 45.0
CANDIDATES = [500 * k for k in range(1, 21)]
TARGET_ORIGINS = 1100


def _key() -> str:
    with open(r"C:\Users\angel\claude code\mission-control\.env.local") as f:
        for line in f:
            m = re.match(r"\s*DATABENTO_API_KEY\s*=\s*['\"]?([^'\"\n]+)", line)
            if m:
                return m.group(1).strip()
    raise RuntimeError("no key")


def stage_download():
    if RAW_PQ.exists():
        print("[1] raw parquet exists — skip download", flush=True)
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
        store = cli.timeseries.get_range(**kw)
        store.to_file(str(RAW_DBN))
    import databento as db2
    store = db2.DBNStore.from_file(str(RAW_DBN))
    print("[1] converting DBN -> parquet ...", flush=True)
    store.to_parquet(str(RAW_PQ), map_symbols=True)


def stage_front_month():
    if TRADES_PQ.exists():
        print("[2] front-month parquet exists — skip", flush=True)
        return
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


def stage_bar_rule() -> int:
    """Pre-registered physics rule, first 5 sessions only."""
    if BAR_RULE_JSON.exists():
        n = json.loads(BAR_RULE_JSON.read_text())["bar_size"]
        print(f"[3] bar rule cached: {n}", flush=True)
        return n
    from forecasting.databento_adapter import load_trades, volume_bars
    trades = load_trades(str(TRADES_PQ))
    days = sorted(trades["ts"].dt.date.unique())[:5]
    first5 = trades[trades["ts"].dt.date.isin(days)]
    print(f"[3] first 5 sessions: {days[0]} -> {days[-1]}, "
          f"{len(first5):,} trades", flush=True)

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
    BAR_RULE_JSON.write_text(json.dumps(
        {"bar_size": chosen, "rule": ">=45s median RTH, first 5 sessions",
         "medians": table}))
    print(f"[3] BAR SIZE = {chosen} contracts", flush=True)
    return chosen


def stage_run(bar_size: int):
    from forecasting import databento_adapter as dba
    from forecasting.run import run_pipeline
    from forecasting import persist as persistmod
    from forecasting.metrics import diebold_mariano

    real = dba.prepare_real(str(TRADES_PQ), bar_size=bar_size,
                            vol_window=256, instrument="ESZ5")
    n = len(real.scaled)
    step = max(1, round((n - 252) / TARGET_ORIGINS))
    print(f"[4] {n} bars -> step {step} "
          f"(~{(n - 252) // step} origins), horizon 4", flush=True)

    result = run_pipeline(
        real.to_series(), horizon=4, min_train=252, step=step, real=real,
        features_out=str(DATA / "ESZ5_features.parquet"),
        backend="kronos", context_length=256, num_paths=24)

    bt = result["backtest"]

    # step-4 DM vs naive from the SAME archive — no second run
    def step_preds(model, s):
        rows = sorted((r for r in bt.archive
                       if r.model == model and r.step == s),
                      key=lambda r: r.origin_index)
        return (np.array([r.y_true for r in rows]),
                np.array([r.mean for r in rows]))
    print("\n[5] per-step diagnostics (same archive):", flush=True)
    models = sorted({r.model for r in bt.archive})
    y4n, m4n = step_preds("naive-rw", 4)
    for mname in models:
        if mname == "naive-rw":
            continue
        y4, m4 = step_preds(mname, 4)
        k = min(len(y4), len(y4n))
        if k > 30:
            dm, p = diebold_mariano(y4[:k], m4[:k], m4n[:k], horizon=4)
            print(f"    step-4 DM {mname} vs naive: stat={dm:+.3f} p={p:.4g}",
                  flush=True)
        r2 = result["backtest"].r2_by_horizon(mname)
        print(f"    {mname} R2oos by step: "
              f"{ {h: round(v, 4) for h, v in r2.items()} }", flush=True)

    run_id = persistmod.persist_run(
        result, real, source_file=str(TRADES_PQ),
        config={"experiment": "phase2_es_execution_reality",
                "bar_size": bar_size, "horizon": 4, "step": step,
                "preregistered": True},
        notes="Phase 2 pre-registered ES execution-reality test (single run)")
    print(f"\n[6] persisted as run {run_id}", flush=True)


if __name__ == "__main__":
    stage_download()
    stage_front_month()
    n = stage_bar_rule()
    stage_run(n)
    print("\nPHASE 2 COMPLETE", flush=True)
