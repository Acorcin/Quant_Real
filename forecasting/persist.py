"""
Persistence layer: land a pipeline run in the `md` Postgres schema.

Fixes the four per-run losses of the file-and-memory design:
  * the forecast ARCHIVE (in-memory list today) becomes a durable record
  * volume bars + conditioned series get a canonical, run-independent store
  * gate decisions / structure verdicts become auditable rows, not console logs
  * run config + input-file hash land in md.runs (no more shell-history forensics)

Design: thin psycopg2, no ORM. Everything under one run_id; bars/conditioned
are keyed by (instrument, bar_size[, vol_window]) and upserted idempotently.
Opt-in: nothing here is imported unless --persist is passed to run.py, so the
file-based path keeps working without psycopg2 installed.

DSN resolution: $QUANT_DB_URL, then $DATABASE_URL, then the engine's
docker-compose default (postgres:postgres@localhost:5432/quant_eod).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from typing import Optional

import numpy as np

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/quant_eod"


def resolve_dsn(dsn: str = "") -> str:
    return (dsn or os.environ.get("QUANT_DB_URL")
            or os.environ.get("DATABASE_URL") or DEFAULT_DSN)


def _connect(dsn: str):
    import psycopg2
    return psycopg2.connect(dsn)


def apply_schema(dsn: str = "") -> None:
    """Apply every forecasting/sql/*.sql migration, sorted (all idempotent)."""
    from pathlib import Path
    conn = _connect(resolve_dsn(dsn))
    try:
        with conn.cursor() as cur:
            for f in sorted((Path(__file__).parent / "sql").glob("*.sql")):
                cur.execute(f.read_text())
        conn.commit()
    finally:
        conn.close()


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(__file__),
        ).stdout.strip() or None
    except Exception:
        return None


def _file_hash(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def persist_run(result: dict, real, *, source_file: str = "",
                config: Optional[dict] = None, dsn: str = "",
                notes: str = "") -> str:
    """Land one completed real-data pipeline run. Returns the run_id.

    Args:
        result: the dict returned by run.run_pipeline (backtest,
            characterization, signals, features).
        real:   the databento_adapter.RealSeries the run consumed.
    """
    dsn = resolve_dsn(dsn)
    apply_schema(dsn)

    bt = result["backtest"]
    char = result["characterization"]
    signals = result["signals"]
    features = result.get("features")

    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            # ── run root ────────────────────────────────────────────────
            cur.execute(
                """INSERT INTO md.runs (git_sha, instrument, config,
                                        source_file, source_hash, notes)
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING run_id""",
                (_git_sha(), bt.instrument, json.dumps(config or {}),
                 source_file, _file_hash(source_file) if source_file else None,
                 notes),
            )
            run_id = str(cur.fetchone()[0])

            _upsert_bars(cur, real)
            _upsert_conditioned(cur, real)
            _copy_archive(cur, run_id, bt)
            _insert_characterization(cur, run_id, char)
            _insert_gate_signals(cur, run_id, signals, real)
            if features is not None:
                _insert_features(cur, run_id, features)
        conn.commit()
        return run_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── section writers ─────────────────────────────────────────────────────────

def _f(x) -> str:
    """Format any scalar (incl. numpy) as a plain COPY-safe number."""
    return repr(float(x))


def _upsert_bars(cur, real) -> None:
    """Canonical bars: idempotent by (instrument, bar_size, bar_seq)."""
    kl = real.klines
    rows = io.StringIO()
    for i in range(len(kl)):
        ts = real.bar_ts[i].isoformat()
        r = kl.iloc[i]
        rows.write(f"{real.instrument}\t{real.bar_size}\t{i}\t{ts}\t"
                   f"{_f(r['open'])}\t{_f(r['high'])}\t{_f(r['low'])}\t"
                   f"{_f(r['close'])}\t{int(r['volume'])}\n")
    rows.seek(0)
    cur.execute("""CREATE TEMP TABLE _bars
                   (LIKE md.volume_bars INCLUDING DEFAULTS) ON COMMIT DROP""")
    cur.copy_expert("COPY _bars FROM STDIN", rows)
    cur.execute("""INSERT INTO md.volume_bars SELECT * FROM _bars
                   ON CONFLICT (instrument, bar_size, bar_seq) DO NOTHING""")


def _upsert_conditioned(cur, real) -> None:
    rows = io.StringIO()
    for i in range(len(real.scaled)):
        rows.write(f"{real.instrument}\t{real.bar_size}\t{real.vol_window}\t"
                   f"{i}\t{real.bar_ts[i].isoformat()}\t"
                   f"{_f(real.raw_returns[i])}\t{_f(real.sigma[i])}\t"
                   f"{_f(real.scaled[i])}\n")
    rows.seek(0)
    cur.execute("""CREATE TEMP TABLE _cond
                   (LIKE md.conditioned_series INCLUDING DEFAULTS)
                   ON COMMIT DROP""")
    cur.copy_expert("COPY _cond FROM STDIN", rows)
    cur.execute("""INSERT INTO md.conditioned_series SELECT * FROM _cond
                   ON CONFLICT (instrument, bar_size, vol_window, seq)
                   DO NOTHING""")


def _copy_archive(cur, run_id: str, bt) -> None:
    """Bulk-COPY the full forecast archive (the biggest table by far)."""
    rows = io.StringIO()
    for r in bt.archive:
        q = json.dumps({str(k): float(v) for k, v in r.quantiles.items()})
        rows.write(f"{run_id}\t{r.model}\t{r.rung}\t{r.origin_index}\t"
                   f"{r.step}\t{r.target_index}\t{_f(r.y_true)}\t{_f(r.mean)}\t"
                   f"{q}\t{_f(r.latency_ms)}\n")
    rows.seek(0)
    cur.copy_expert("COPY md.forecast_archive FROM STDIN", rows)


def _insert_characterization(cur, run_id: str, char) -> None:
    ladder = [{
        "rung": l.rung, "model": l.model, "mase": l.mase,
        "r2_oos": None if not np.isfinite(l.r2_oos) else l.r2_oos,
        "dm_pvalue": l.dm_pvalue_vs_naive,
        "directional_acc": l.directional_acc,
    } for l in char.ladder]
    cur.execute(
        """INSERT INTO md.characterizations
           (run_id, instrument, data_asof, structure, adf_p, kpss_p, hurst,
            perm_entropy, edge_horizon, ladder, regimes, breaks)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (run_id, char.instrument, char.data_asof or None,
         char.structure_type.value, char.adf_pvalue, char.kpss_pvalue,
         char.hurst, char.permutation_entropy,
         char.predictability_edge_horizon,
         json.dumps(ladder),
         json.dumps({"names": {str(k): v for k, v in char.regime_names.items()},
                     "current": char.current_regime,
                     "unpredictable": char.unpredictable_regimes}),
         json.dumps([int(b) for b in char.structural_breaks])),
    )


def _insert_gate_signals(cur, run_id: str, signals, real) -> None:
    from psycopg2.extras import execute_values
    vals = []
    for s in signals:
        t = s.origin_index
        ts = real.bar_ts[min(t, len(real.bar_ts) - 1)].isoformat()
        vals.append((run_id, t, ts, s.gate.value,
                     [r.value for r in s.veto_reasons],
                     s.direction, s.conviction, s.vol_forecast, s.regime))
    execute_values(cur, """INSERT INTO md.gate_signals
        (run_id, origin_seq, ts, state, veto_reasons, direction, conviction,
         vol_forecast, regime) VALUES %s""", vals)


def _insert_features(cur, run_id: str, features) -> None:
    """Land the bifurcated XGBoost feature rows (bridge table)."""
    from psycopg2.extras import execute_values
    core = {"origin_index", "ts", "sigma_256", "structure_type", "gate_state",
            "veto_reasons", "y_true_scaled", "kronos_p50_scaled",
            "kronos_spread_scaled"}
    vals = []
    for _, row in features.iterrows():
        extra = {k: (None if (isinstance(v, float) and not np.isfinite(v))
                     else v)
                 for k, v in row.items() if k not in core}
        vals.append((
            run_id, int(row["origin_index"]),
            row["ts"].isoformat() if hasattr(row["ts"], "isoformat") else None,
            float(row["sigma_256"]),
            float(row["y_true_scaled"]),
            float(row.get("kronos_p50_scaled", float("nan")))
            if np.isfinite(row.get("kronos_p50_scaled", float("nan"))) else None,
            float(row.get("kronos_spread_scaled", float("nan")))
            if np.isfinite(row.get("kronos_spread_scaled", float("nan"))) else None,
            row["structure_type"], row["gate_state"],
            (row.get("veto_reasons") or "").split("|") if row.get("veto_reasons")
            else [],
            json.dumps(extra, default=str),
        ))
    execute_values(cur, """INSERT INTO md.kronos_features
        (run_id, origin_seq, ts, sigma_256, y_true_scaled, kronos_p50_scaled,
         kronos_spread_scaled, structure, gate_state, veto_reasons, extra)
        VALUES %s""", vals)
