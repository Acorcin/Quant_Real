"""
One-shot stack bootstrap (runs before the feed services):

  1. Apply the engine schema files (idempotent CREATE IF NOT EXISTS).
  2. Apply the forecasting md migrations (idempotent).
  3. If the target instrument has no bars yet (fresh volume) and a trades
     parquet is available under /app/data, seed md with a fast stat-only
     pipeline run so live_loop has the history it needs.

Exits 0 on success; compose gates the feeds on that.
"""
import os
import subprocess
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/engine")

INSTRUMENT = os.environ.get("QR_INSTRUMENT", "M6EU6")
BAR_SIZE = os.environ.get("QR_BAR_SIZE", "250")
SEED_PARQUET = f"/app/data/{INSTRUMENT}_trades.parquet"


def main() -> int:
    print("[bootstrap] applying engine schemas...", flush=True)
    from models.database import init_schema
    init_schema()

    print("[bootstrap] applying md migrations...", flush=True)
    from forecasting import persist
    persist.apply_schema()

    import psycopg2
    conn = psycopg2.connect(persist.resolve_dsn())
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM md.volume_bars WHERE instrument=%s",
                    (INSTRUMENT,))
        n = cur.fetchone()[0]
    conn.close()
    print(f"[bootstrap] {INSTRUMENT}: {n} bars in md", flush=True)

    if n < 300:
        if not os.path.exists(SEED_PARQUET):
            print(f"[bootstrap] WARNING: no bars and no seed parquet at "
                  f"{SEED_PARQUET} — live_loop will refuse to start until "
                  f"md is seeded (run a --real-data --persist pipeline once).",
                  flush=True)
            return 0    # schemas are in place; don't block the other feeds
        print(f"[bootstrap] seeding from {SEED_PARQUET} (stat rungs only)...",
              flush=True)
        r = subprocess.run([sys.executable, "-m", "forecasting.run",
                            "--real-data", SEED_PARQUET,
                            "--instrument", INSTRUMENT,
                            "--bar-size", BAR_SIZE,
                            "--no-foundation", "--step", "4", "--persist"],
                           cwd="/app")
        if r.returncode != 0:
            print("[bootstrap] seed run failed", flush=True)
            return 1
    print("[bootstrap] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
