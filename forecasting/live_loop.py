"""
Live forecast loop: keeps md fed with CURRENT Kronos forecasts.

The pipeline's batch mode measures a series and dies; this loop keeps the
instrument pointed at the market:

  poll free Historical trades (cost-guarded, ~10min lag)
    -> extend canonical volume bars (md.volume_bars, same accumulate-until-N
       rule as the batch adapter, forming bar carried in memory)
    -> per completed bar: trailing sigma -> conditioned row
    -> ONE Kronos inference on the latest context (real K-lines)
    -> fresh md.kronos_features row under a persistent 'live' run

The engine's bridge (load_latest_kronos) then serves a forecast that is
minutes old, not months. structure_type rides along from the newest FULL
characterization run for the instrument — the slowly-varying verdict is
refreshed by periodic batch runs, not per bar (same stance as the batch
feature exporter). If no characterization exists yet, rows carry NULL
structure and the engine's veto treats them as unknown (-1) — which the
structural veto deliberately does NOT veto, so run a batch --persist first.

    python -m forecasting.live_loop --symbol M6E.FUT --instrument M6EU6 --once
    python -m forecasting.live_loop --symbol M6E.FUT --instrument M6EU6 --loop 120
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from .l3_puller import _api_key, AVAILABILITY_LAG, COST_ABORT_USD, DATASET
from . import persist as persistmod

logger = logging.getLogger("live_loop")

MAX_WINDOW = timedelta(minutes=30)        # near-real-time: same-day get_range
                                          # 504s beyond ~30 min
CATCHUP_WINDOW = timedelta(hours=4)       # backlog is historical/fully
                                          # processed — pull big chunks
CATCHUP_AFTER = timedelta(hours=2)        # "old enough to be historical"


class LiveForecaster:
    def __init__(self, symbol: str, instrument: str, *, bar_size: int = 250,
                 vol_window: int = 256, context_length: int = 256,
                 num_paths: int = 24, dsn: str = ""):
        self.symbol = symbol            # Databento parent, e.g. M6E.FUT
        self.instrument = instrument    # tradable/md series, e.g. M6EU6
        self.bar_size = bar_size
        self.vol_window = vol_window
        self.dsn = dsn
        self._client = None
        self._fc = None
        self._run_id: Optional[str] = None
        self._structure: Optional[str] = None
        # forming-bar state (rebuilt from md watermark on restart)
        self._forming = {"vol": 0, "open": None, "high": -np.inf,
                         "low": np.inf, "close": None, "ts": None}
        self._kwargs = dict(context_length=context_length, num_paths=num_paths)

    # -- lazy singletons ------------------------------------------------------

    def _cli(self):
        if self._client is None:
            import databento as db
            self._client = db.Historical(_api_key())
        return self._client

    def _forecaster(self):
        if self._fc is None:
            from .models import FoundationForecaster
            self._fc = FoundationForecaster(backend="kronos", **self._kwargs)
            self._fc.fit(np.zeros(8))       # resolve backend + load weights now
            logger.info("kronos backend resident")
        return self._fc

    def _conn(self):
        import psycopg2
        persistmod.apply_schema(self.dsn)
        return psycopg2.connect(persistmod.resolve_dsn(self.dsn))

    # -- db state -------------------------------------------------------------

    def _ensure_live_run(self, conn) -> str:
        if self._run_id:
            return self._run_id
        with conn.cursor() as cur:
            cur.execute("""SELECT run_id FROM md.runs
                           WHERE instrument = %s AND notes = 'live loop'
                           ORDER BY started_at DESC LIMIT 1""",
                        (self.instrument,))
            row = cur.fetchone()
            if row:
                self._run_id = str(row[0])
            else:
                import json
                cur.execute(
                    """INSERT INTO md.runs (instrument, config, notes)
                       VALUES (%s, %s, 'live loop') RETURNING run_id""",
                    (self.instrument,
                     json.dumps({"symbol": self.symbol,
                                 "bar_size": self.bar_size,
                                 "vol_window": self.vol_window,
                                 **self._kwargs})))
                self._run_id = str(cur.fetchone()[0])
                conn.commit()
        return self._run_id

    def _load_structure(self, conn) -> Optional[str]:
        with conn.cursor() as cur:
            cur.execute("""SELECT c.structure::text FROM md.characterizations c
                           JOIN md.runs r USING (run_id)
                           WHERE c.instrument = %s
                           ORDER BY r.started_at DESC LIMIT 1""",
                        (self.instrument,))
            row = cur.fetchone()
        return row[0] if row else None

    def _bars_tail(self, conn, n: int):
        """Last n bars + the trades watermark implied by the last bar."""
        with conn.cursor() as cur:
            cur.execute("""SELECT bar_seq, ts_close, open, high, low, close, volume
                           FROM md.volume_bars
                           WHERE instrument = %s AND bar_size = %s
                           ORDER BY bar_seq DESC LIMIT %s""",
                        (self.instrument, self.bar_size, n))
            rows = cur.fetchall()[::-1]
        if not rows:
            raise RuntimeError(
                f"no bars for {self.instrument} (bar_size {self.bar_size}) — "
                "seed md first: forecasting.run --real-data ... --persist")
        cols = ["bar_seq", "ts_close", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(rows, columns=cols)

    # -- one cycle ------------------------------------------------------------

    def cycle(self) -> dict:
        conn = self._conn()
        try:
            run_id = self._ensure_live_run(conn)
            if self._structure is None:
                self._structure = self._load_structure(conn)
                logger.info("structure verdict for %s: %s",
                            self.instrument, self._structure or "none yet")

            bars = self._bars_tail(conn, max(self.vol_window + 8, 300))
            wm = bars["ts_close"].iloc[-1]
            if wm.tzinfo is None:
                wm = wm.tz_localize("UTC")

            now = datetime.now(timezone.utc)
            end = now - AVAILABILITY_LAG
            # adaptive chunk: big for historical backlog, small near real-time
            cap = (CATCHUP_WINDOW if (now - wm) > CATCHUP_AFTER else MAX_WINDOW)
            if end - wm > cap:
                end = wm + cap
            if wm >= end:
                return {"status": "idle", "reason": "no new window"}

            trades = self._pull_trades(wm, end)
            if trades is None:
                return {"status": "empty"}
            new_bars = self._extend_bars(trades)
            if not new_bars:
                return {"status": "ok", "new_bars": 0,
                        "forming_vol": self._forming["vol"]}

            made = self._commit_bars(conn, bars, new_bars, run_id)
            return {"status": "ok", "new_bars": len(new_bars),
                    "forecasts": made,
                    "forming_vol": self._forming["vol"]}
        finally:
            conn.close()

    def _pull_trades(self, start, end) -> Optional[pd.DataFrame]:
        cli = self._cli()
        kw = dict(dataset=DATASET, symbols=[self.symbol], stype_in="parent",
                  schema="trades")
        quoted = cli.metadata.get_cost(start=start, end=end, **kw)
        if quoted > COST_ABORT_USD:
            logger.error("ABORT: trades pull would cost $%.2f", quoted)
            return None
        try:
            store = cli.timeseries.get_range(start=start, end=end, **kw)
        except Exception as e:
            if "unavailable_range" in str(e):
                return None                 # boundary; next cycle
            raise
        df = store.to_df().reset_index()
        if df.empty:
            return df
        df = df[df["symbol"] == self.instrument]
        return df.sort_values("ts_event", kind="stable")

    def _extend_bars(self, trades: pd.DataFrame) -> List[dict]:
        """Continue the accumulate-until-N rule across polls via forming state."""
        out = []
        f = self._forming
        for _, t in trades.iterrows():
            px, sz, ts = float(t["price"]), int(t["size"]), t["ts_event"]
            if f["open"] is None:
                f.update(open=px, high=px, low=px)
            f["high"] = max(f["high"], px)
            f["low"] = min(f["low"], px)
            f["close"], f["ts"] = px, ts
            f["vol"] += sz
            if f["vol"] >= self.bar_size:
                out.append(dict(f))
                self._forming = {"vol": 0, "open": None, "high": -np.inf,
                                 "low": np.inf, "close": None, "ts": None}
                f = self._forming
        return out

    def _commit_bars(self, conn, hist: pd.DataFrame, new_bars: List[dict],
                     run_id: str) -> int:
        """Insert bars + conditioned rows, then one Kronos forecast per bar."""
        made = 0
        closes = hist["close"].to_numpy(dtype=float).tolist()
        seq = int(hist["bar_seq"].iloc[-1])
        # trailing raw returns for sigma (recomputed from closes as we extend)
        rets = list(np.diff(np.log(hist["close"].to_numpy(dtype=float))))

        kl = hist[["open", "high", "low", "close", "volume"]].copy()
        ts_list = list(pd.DatetimeIndex(hist["ts_close"]))

        with conn.cursor() as cur:
            for b in new_bars:
                seq += 1
                ts = pd.Timestamp(b["ts"]).tz_convert("UTC") \
                    if pd.Timestamp(b["ts"]).tzinfo else \
                    pd.Timestamp(b["ts"]).tz_localize("UTC")
                cur.execute(
                    """INSERT INTO md.volume_bars
                       (instrument, bar_size, bar_seq, ts_close, open, high,
                        low, close, volume)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT DO NOTHING""",
                    (self.instrument, self.bar_size, seq, ts, b["open"],
                     b["high"], b["low"], b["close"], b["vol"]))

                r = float(np.log(b["close"] / closes[-1]))
                closes.append(b["close"])
                sigma_win = rets[-self.vol_window:]
                sigma = float(np.std(sigma_win, ddof=1)) if len(sigma_win) > 32 \
                    else float("nan")
                rets.append(r)
                if not np.isfinite(sigma) or sigma <= 0:
                    continue
                scaled = r / sigma
                cur.execute(
                    """INSERT INTO md.conditioned_series
                       (instrument, bar_size, vol_window, seq, ts, raw_return,
                        sigma, scaled_return)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT DO NOTHING""",
                    (self.instrument, self.bar_size, self.vol_window, seq, ts,
                     r, sigma, scaled))

                kl.loc[len(kl)] = [b["open"], b["high"], b["low"], b["close"],
                                   b["vol"]]
                ts_list.append(ts)

                p50, spread = self._forecast(kl, ts_list, rets, sigma)
                cur.execute(
                    """INSERT INTO md.kronos_features
                       (run_id, origin_seq, ts, sigma_256, y_true_scaled,
                        kronos_p50_scaled, kronos_spread_scaled, structure)
                       VALUES (%s,%s,%s,%s,NULL,%s,%s,%s)
                       ON CONFLICT (run_id, origin_seq) DO UPDATE SET
                           kronos_p50_scaled = EXCLUDED.kronos_p50_scaled,
                           kronos_spread_scaled = EXCLUDED.kronos_spread_scaled""",
                    (run_id, seq, ts, sigma, p50, spread, self._structure))
                made += 1
                logger.info("%s bar %d @ %s close=%.5f -> p50=%+.4f spread=%.4f",
                            self.instrument, seq, ts, b["close"], p50, spread)
        conn.commit()
        return made

    def _forecast(self, kl: pd.DataFrame, ts_list, rets, sigma_next: float):
        """One Kronos inference on the live context; returns scaled p50/spread."""
        fc = self._forecaster()
        L = min(600, len(rets))             # klines row i <-> scaled return i

        class _Shim:                        # what attach_klines expects
            klines = kl.tail(L).reset_index(drop=True)
            bar_ts = pd.DatetimeIndex(ts_list[-L:])
            sigma = np.append(np.zeros(L - 1), sigma_next)
        fc.attach_klines(_Shim)

        ctx = np.array(rets[-L:]) / max(sigma_next, 1e-12)
        res = fc.predict(ctx, horizon=1)
        q = res.quantiles
        p50 = float(q[0.5][0])
        spread = float(q[0.9][0] - q[0.1][0])
        return p50, spread


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="M6E.FUT")
    ap.add_argument("--instrument", default="M6EU6",
                    help="tradable front-month contract (md series)")
    ap.add_argument("--bar-size", type=int, default=250)
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if args.loop:
        from .singleton import acquire, AlreadyRunning
        try:
            acquire(f"live_loop_{args.instrument}")
        except AlreadyRunning as e:
            logger.error(str(e))
            raise SystemExit(1)

    lf = LiveForecaster(args.symbol, args.instrument, bar_size=args.bar_size)
    if args.loop:
        while True:
            try:
                r = lf.cycle()
                logger.info("cycle: %s", r)
            except Exception:
                logger.exception("cycle failed; retrying next interval")
            time.sleep(args.loop)
    else:
        print(lf.cycle())


if __name__ == "__main__":
    main()
