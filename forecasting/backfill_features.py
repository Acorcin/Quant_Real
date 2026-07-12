"""
One-time backfill: import an existing *_features.parquet (a completed run's
bifurcated output) into md.kronos_features under its own md.runs row.

Exists for exactly one reason: the forecast archive of a run that predates the
md schema is gone (it lived in process memory), but its features parquet
survives. This lands what survives, marked as a backfill in md.runs.notes.

    python -m forecasting.backfill_features data/M6EH6_trades_features.parquet \
        --instrument M6EH6 --source data/M6EH6_trades.parquet
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import persist as persistmod


def backfill(features_path: str, instrument: str, source_file: str = "",
             notes: str = "backfill from features parquet (archive not "
                          "recoverable — predates md schema)") -> str:
    features = pd.read_parquet(features_path)
    dsn = persistmod.resolve_dsn()
    persistmod.apply_schema(dsn)

    conn = persistmod._connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO md.runs (git_sha, instrument, config,
                                        source_file, source_hash, notes)
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING run_id""",
                (persistmod._git_sha(), instrument,
                 json.dumps({"backfill_from": features_path}),
                 source_file,
                 persistmod._file_hash(source_file) if source_file else None,
                 notes),
            )
            run_id = str(cur.fetchone()[0])
            persistmod._insert_features(cur, run_id, features)
        conn.commit()
        return run_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("features_parquet")
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--source", default="", help="original trades parquet")
    args = ap.parse_args()
    run_id = backfill(args.features_parquet, args.instrument, args.source)
    print(f"[backfill] {args.features_parquet} -> md.kronos_features "
          f"(run {run_id})")


if __name__ == "__main__":
    main()
