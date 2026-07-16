"""
Physics feed: per-tick conditioning of Databento CME trades -> md.cond_ticks.

The engine's PhysicsEngine (rolling-median spike rejection -> 2D Kalman
price/velocity filter -> scale-normalized, regime-clipped returns) predates
the CME transition: it consumed OANDA FX bid/ask ticks from Redis. This worker
is its new body — same math, new plumbing:

    input   OANDA Redis stream      ->  free Databento Historical polling
                                        (watermark, cost-guard, adaptive
                                         windows — same as the other feeds)
    scale   FX daily ATR            ->  daily scale derived from the md
                                        conditioned series (sigma_256 x
                                        sqrt(bars/day) x close)
    regime  FX HMM regimes table    ->  md.characterizations current regime
                                        (calm/turbulent)
    state   Redis checkpoint        ->  md.physics_state (JSONB + watermark)
    output  Redis cond_ticks stream ->  md.cond_ticks (+ md.v_physics_latest)

Downstream: cleaned per-tick prices/velocity for dashboards and, later, an
opt-in cleaned-price input to the volume-bar builder.

    python -m forecasting.physics_feed --symbol M6E.FUT --instrument M6EU6 --once
    python -m forecasting.physics_feed --symbol M6E.FUT --instrument M6EU6 --loop 60
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .l3_puller import _api_key, AVAILABILITY_LAG, COST_ABORT_USD, DATASET
from . import persist as persistmod

# the physics package lives in the engine half of the monorepo
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

logger = logging.getLogger("physics_feed")

MAX_WINDOW = timedelta(minutes=30)
CATCHUP_WINDOW = timedelta(hours=4)
CATCHUP_AFTER = timedelta(hours=2)


class PhysicsFeed:
    def __init__(self, symbol: str, instrument: str, *, dsn: str = ""):
        self.symbol = symbol          # Databento parent, e.g. M6E.FUT
        self.instrument = instrument  # tradable/md series, e.g. M6EU6
        self.dsn = dsn
        self._client = None
        from physics.engine import PhysicsEngine
        self.engine = PhysicsEngine()
        self._state_loaded = False
        self._meta_at = 0.0
        self._regime = "turbulent"
        self._scale = 0.005

    def _cli(self):
        if self._client is None:
            import databento as db
            self._client = db.Historical(_api_key())
        return self._client

    def _conn(self):
        import psycopg2
        persistmod.apply_schema(self.dsn)
        return psycopg2.connect(persistmod.resolve_dsn(self.dsn))

    # ── md-sourced metadata (regime + daily scale), refreshed hourly ────────
    def _refresh_meta(self, conn) -> None:
        if time.time() - self._meta_at < 3600:
            return
        with conn.cursor() as cur:
            cur.execute("""SELECT c.regimes->>'current'
                           FROM md.characterizations c
                           JOIN md.runs r USING (run_id)
                           WHERE c.instrument = %s
                           ORDER BY r.started_at DESC LIMIT 1""",
                        (self.instrument,))
            row = cur.fetchone()
            if row and row[0]:
                self._regime = row[0]
            # daily scale = sigma x sqrt(bars in last 24h of data) x close
            cur.execute("""
                WITH last_bar AS (
                    SELECT max(ts_close) AS mx FROM md.volume_bars
                    WHERE instrument = %s),
                bpd AS (
                    SELECT count(*) AS n FROM md.volume_bars b, last_bar
                    WHERE b.instrument = %s
                      AND b.ts_close > last_bar.mx - interval '24 hours'),
                cs AS (
                    SELECT sigma FROM md.conditioned_series
                    WHERE instrument = %s ORDER BY seq DESC LIMIT 1),
                px AS (
                    SELECT close FROM md.volume_bars
                    WHERE instrument = %s ORDER BY bar_seq DESC LIMIT 1)
                SELECT cs.sigma, bpd.n, px.close FROM cs, bpd, px""",
                        (self.instrument,) * 4)
            row = cur.fetchone()
            if row and all(v is not None for v in row):
                sigma, n, close = float(row[0]), max(int(row[1]), 1), float(row[2])
                self._scale = max(sigma * (n ** 0.5) * close, 1e-6)
        self._meta_at = time.time()
        logger.info("meta: regime=%s daily_scale=%.5f", self._regime, self._scale)

    # ── state / watermark (replaces Redis checkpointing) ────────────────────
    def _load_state(self, conn) -> Optional[datetime]:
        with conn.cursor() as cur:
            cur.execute("""SELECT state, watermark FROM md.physics_state
                           WHERE instrument = %s""", (self.instrument,))
            row = cur.fetchone()
        if not row:
            return None
        if not self._state_loaded:
            try:
                self.engine.set_state(row[0])
                self._state_loaded = True
                logger.info("restored PhysicsEngine state (watermark %s)", row[1])
            except Exception as e:
                logger.warning("state restore failed (%s); starting fresh", e)
        return row[1]

    def _save_state(self, cur, watermark) -> None:
        cur.execute("""
            INSERT INTO md.physics_state
                (instrument, state, watermark, regime, daily_scale, updated_at)
            VALUES (%s, %s, %s, %s, %s, now())
            ON CONFLICT (instrument) DO UPDATE SET
                state = EXCLUDED.state, watermark = EXCLUDED.watermark,
                regime = EXCLUDED.regime, daily_scale = EXCLUDED.daily_scale,
                updated_at = now()""",
                    (self.instrument, json.dumps(self.engine.get_state()),
                     watermark, self._regime, self._scale))

    # ── one poll cycle ───────────────────────────────────────────────────────
    def cycle(self) -> dict:
        conn = self._conn()
        try:
            self._refresh_meta(conn)
            now = datetime.now(timezone.utc)
            end = now - AVAILABILITY_LAG
            wm = self._load_state(conn)
            if wm is None:
                wm = end - MAX_WINDOW          # first run: start one window back
            cap = CATCHUP_WINDOW if (now - wm) > CATCHUP_AFTER else MAX_WINDOW
            if end - wm > cap:
                end = wm + cap
            if wm >= end:
                return {"status": "idle"}

            trades = self._pull(wm, end)
            if trades is None:
                return {"status": "empty"}

            n_ticks, n_spikes = self._process_and_write(conn, trades, end)
            logger.info("%s: %s -> %s | %d ticks, %d spikes",
                        self.instrument, wm.strftime("%H:%M"),
                        end.strftime("%H:%M"), n_ticks, n_spikes)
            return {"status": "ok", "ticks": n_ticks, "spikes": n_spikes}
        finally:
            conn.close()

    def _pull(self, start, end) -> Optional[pd.DataFrame]:
        cli = self._cli()
        kw = dict(dataset=DATASET, symbols=[self.symbol], stype_in="parent",
                  schema="trades")
        quoted = cli.metadata.get_cost(start=start, end=end, **kw)
        if quoted > COST_ABORT_USD:
            logger.error("ABORT: pull would cost $%.2f", quoted)
            return None
        try:
            store = cli.timeseries.get_range(start=start, end=end, **kw)
        except Exception as e:
            if "unavailable_range" in str(e):
                return None
            raise
        df = store.to_df().reset_index()
        if df.empty:
            return df
        df = df[df["symbol"] == self.instrument]
        return df.sort_values("ts_event", kind="stable")

    def _process_and_write(self, conn, trades: pd.DataFrame, watermark) -> tuple:
        rows = io.StringIO()
        n = spikes = 0
        for _, t in trades.iterrows():
            ts = pd.Timestamp(t["ts_event"])
            ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
            out = self.engine.process_trade(
                float(t["price"]), ts.timestamp(),
                regime_label=self._regime, daily_scale=self._scale)
            n += 1
            spikes += int(out["is_spike"])
            rows.write("\t".join([
                self.instrument, ts.isoformat(), repr(float(t["price"])),
                str(int(t["size"])), repr(float(out["filtered_mid"])),
                "t" if out["is_spike"] else "f",
                repr(float(out["kalman_price"])),
                repr(float(out["kalman_velocity"])),
                repr(float(out["tick_return"])),
                repr(float(out["normalized_return"])),
                repr(float(out["clipped_return"])),
            ]) + "\n")
        rows.seek(0)
        with conn.cursor() as cur:
            if n:
                cur.copy_expert("""
                    COPY md.cond_ticks (instrument, ts, price, size,
                        filtered_price, is_spike, kalman_price,
                        kalman_velocity, tick_return, normalized_return,
                        clipped_return) FROM STDIN""", rows)
            self._save_state(cur, watermark)
        conn.commit()
        return n, spikes


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="M6E.FUT")
    ap.add_argument("--instrument", default="M6EU6")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS")
    args = ap.parse_args()

    if args.loop:
        from .singleton import acquire, AlreadyRunning
        try:
            acquire(f"physics_feed_{args.instrument}")
        except AlreadyRunning as e:
            logger.error(str(e))
            raise SystemExit(1)

    feed = PhysicsFeed(args.symbol, args.instrument)
    if args.loop:
        while True:
            try:
                r = feed.cycle()
                logger.info("cycle: %s", r)
            except Exception:
                logger.exception("cycle failed; retrying next interval")
            time.sleep(args.loop)
    else:
        print(feed.cycle())


if __name__ == "__main__":
    main()
