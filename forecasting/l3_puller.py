"""
Free periodic L3 (MBO) puller: incremental Databento Historical pulls
throughout the trading day -> order-book state + VPIN in the md schema.

Why polling and not Live: on this account, same-day Historical slices quote
$0.00 (intraday replay is inside the included window) while Live streams meter
at $2.16/GB for MBO. The pipeline's volume bars form over 30s+ anyway, so a
minutes-scale replay lag costs nothing strategically. Every pull is preceded
by a get_cost quote and ABORTS if it isn't ~free — the guard that keeps
"periodically all day" from quietly becoming a bill.

Mechanics per poll:
  * window = [watermark, now - AVAILABILITY_LAG); the first window of a UTC
    day starts at midnight, whose synthetic snapshot seeds the book. The
    watermark comes from md.l3_polls, so restarts resume correctly — but a
    restarted process has an empty in-memory book, so it re-anchors from
    midnight (still free) instead of trusting the watermark.
  * MBO replay: A(dd)/C(ancel)/M(odify)/F(ill) maintain per-order state,
    R(eset/clear) empties the book, T(rade) events feed VPIN buckets via
    aggressor side. Book state is sampled every SAMPLE_EVERY of event time
    into md.l3_book_state (L1 + depth-weighted top-10 imbalance).
  * raw DBN increment is kept in the lake: data/lake/<sym>/mbo/.

Downstream: md.v_l3_latest exposes l3_order_book_imbalance / l3_vpin — the
meta-model's placeholder columns.

    python -m forecasting.l3_puller --symbol M6E.FUT --once
    python -m forecasting.l3_puller --symbol ES.FUT --loop 300
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("l3_puller")

AVAILABILITY_LAG = timedelta(minutes=20)   # historical availability buffer:
                                           # the account's replay window ends at
                                           # ~now-15min, so stay clear of it
SAMPLE_EVERY = timedelta(seconds=1)        # book-state sampling (event time)
COST_ABORT_USD = 0.01                      # hard guard: pulls must be ~free
MAX_WINDOW = timedelta(minutes=30)         # per-poll cap: same-day get_range
                                           # 504s beyond ~30min windows; the
                                           # loop catches up chunk by chunk
DATASET = "GLBX.MDP3"
LAKE = Path(__file__).resolve().parents[1] / "data" / "lake"

_KEY_FILE = r"C:\Users\angel\claude code\mission-control\.env.local"


def _api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY")
    if key:
        return key
    try:
        with open(_KEY_FILE) as f:
            for line in f:
                m = re.match(r"\s*DATABENTO_API_KEY\s*=\s*['\"]?([^'\"\n]+)", line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    raise RuntimeError("DATABENTO_API_KEY not set and no key file found")


# ---------------------------------------------------------------------------
# Order-book engine (per-order, aggregated to price levels on demand)
# ---------------------------------------------------------------------------

@dataclass
class Book:
    """MBO book: order_id -> (side, price, size). Rebuilt each UTC day from
    the midnight snapshot; cleared on R(eset) events."""

    orders: Dict[int, Tuple[str, float, int]] = field(default_factory=dict)

    def apply(self, action: str, side: str, order_id: int,
              price: float, size: int) -> None:
        if action == "A":
            if side in ("A", "B"):
                self.orders[order_id] = (side, price, size)
        elif action == "C":
            self.orders.pop(order_id, None)
        elif action == "M":
            old = self.orders.get(order_id)
            s = side if side in ("A", "B") else (old[0] if old else None)
            if s:
                self.orders[order_id] = (s, price, size)
        elif action == "F":
            old = self.orders.get(order_id)
            if old:
                left = old[2] - size
                if left > 0:
                    self.orders[order_id] = (old[0], old[1], left)
                else:
                    self.orders.pop(order_id, None)
        elif action == "R":
            self.orders.clear()

    def levels(self, depth: int = 10):
        """(bids, asks) as price->size dicts limited to `depth` best levels."""
        bids: Dict[float, int] = {}
        asks: Dict[float, int] = {}
        for side, px, sz in self.orders.values():
            (bids if side == "B" else asks)[px] = \
                (bids if side == "B" else asks).get(px, 0) + sz
        best_b = dict(sorted(bids.items(), reverse=True)[:depth])
        best_a = dict(sorted(asks.items())[:depth])
        return best_b, best_a

    def snapshot(self):
        """L1 + depth-imbalance sample, or None while the book is one-sided."""
        bids, asks = self.levels(10)
        if not bids or not asks:
            return None
        bid_px, ask_px = max(bids), min(asks)
        bid_sz, ask_sz = bids[bid_px], asks[ask_px]
        d_b, d_a = sum(bids.values()), sum(asks.values())
        return {
            "bid_px": bid_px, "ask_px": ask_px,
            "bid_sz": bid_sz, "ask_sz": ask_sz,
            "imbalance_l1": (bid_sz - ask_sz) / max(bid_sz + ask_sz, 1),
            "imbalance_d10": (d_b - d_a) / max(d_b + d_a, 1),
            "n_bid": sum(1 for s, _, _ in self.orders.values() if s == "B"),
            "n_ask": sum(1 for s, _, _ in self.orders.values() if s == "A"),
        }


@dataclass
class VpinState:
    """Volume bucketing with aggressor-side classification (Easley et al.)."""

    bucket_vol: int
    seq: int = 0
    buy: int = 0
    sell: int = 0
    fill: int = 0

    def add_trade(self, side: str, size: int, ts) -> List[dict]:
        """Aggressor side: 'B' = buyer-initiated, 'A' = seller-initiated.
        Returns any buckets closed by this trade."""
        closed = []
        remaining = size
        while remaining > 0:
            take = min(remaining, self.bucket_vol - self.fill)
            if side == "B":
                self.buy += take
            elif side == "A":
                self.sell += take
            else:                       # unknown aggressor: split evenly
                self.buy += take / 2
                self.sell += take / 2
            self.fill += take
            remaining -= take
            if self.fill >= self.bucket_vol:
                closed.append({"seq": self.seq, "ts": ts,
                               "buy": int(self.buy), "sell": int(self.sell),
                               "vol": self.bucket_vol})
                self.seq += 1
                self.buy = self.sell = self.fill = 0
        return closed


# ---------------------------------------------------------------------------
# The poller
# ---------------------------------------------------------------------------

class L3Poller:
    """schema='mbp-1' (default): the live loop. L1 book state rides on every
    record (no reconstruction needed) and is available ~10min behind realtime.
    schema='mbo': deep backfill with full-depth imbalance — Databento's MBO
    processing for GLBX runs HOURS behind, so it cannot see the current
    session; use it to enrich history, not to trade."""

    def __init__(self, symbol: str, *, schema: str = "mbp-1",
                 vpin_bucket: int = 500, dsn: str = "", keep_raw: bool = True):
        if schema not in ("mbp-1", "mbo"):
            raise ValueError("schema must be mbp-1 or mbo")
        self.symbol = symbol
        self.schema = schema
        self.dsn = dsn
        self.keep_raw = keep_raw
        self.book = Book()
        self.vpin = VpinState(bucket_vol=vpin_bucket)
        self.book_day: Optional[datetime.date] = None  # day the book is valid for
        self._client = None

    # -- infrastructure -----------------------------------------------------

    def _cli(self):
        if self._client is None:
            import databento as db
            self._client = db.Historical(_api_key())
        return self._client

    def _conn(self):
        import psycopg2
        from . import persist
        persist.apply_schema(self.dsn)
        return psycopg2.connect(persist.resolve_dsn(self.dsn))

    def _watermark(self, conn) -> Optional[datetime]:
        with conn.cursor() as cur:
            cur.execute("""SELECT max(ts_end) FROM md.l3_polls
                           WHERE instrument = %s AND schema = %s
                             AND status = 'ok'""",
                        (self.symbol, self.schema))
            row = cur.fetchone()
        return row[0] if row and row[0] else None

    def _resume_vpin(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute("""SELECT coalesce(max(bucket_seq), -1)
                           FROM md.l3_vpin_buckets WHERE instrument = %s""",
                        (self.symbol,))
            self.vpin.seq = cur.fetchone()[0] + 1

    # -- one poll -----------------------------------------------------------

    def poll(self) -> dict:
        now = datetime.now(timezone.utc)
        end = now - AVAILABILITY_LAG
        midnight = end.replace(hour=0, minute=0, second=0, microsecond=0)

        conn = self._conn()
        try:
            self._resume_vpin(conn)
            wm = self._watermark(conn)

            # window start: mbp-1 carries L1 state on every record, so we
            # simply resume at the watermark. mbo must rebuild its book from
            # the midnight snapshot after a restart or day roll (still free,
            # just re-reads the day) — but trades before the watermark are
            # excluded from VPIN so re-reads never double-count buckets.
            vpin_after = None
            if self.schema == "mbp-1":
                start = wm or midnight
            elif (self.book_day != end.date()) or wm is None:
                start = midnight
                self.book = Book()
                vpin_after = wm
            else:
                start = wm
            if start >= end:
                return {"status": "empty", "reason": "window not yet available"}
            if end - start > MAX_WINDOW:
                end = start + MAX_WINDOW    # chunk; next poll continues

            cli = self._cli()
            kw = dict(dataset=DATASET, symbols=[self.symbol],
                      stype_in="parent", schema=self.schema)

            # Fetch with two failure modes handled:
            #  * 422 unavailable_range — the free replay boundary is only
            #    knowable from the error message; clamp `end` to it.
            #  * 5xx gateway timeouts — same-day slices are flaky; retry with
            #    backoff, then shrink the window (next poll resumes anyway).
            quoted = store = None
            for shrink in range(3):
                for attempt in range(3):
                    try:
                        if quoted is None:
                            quoted = cli.metadata.get_cost(
                                start=start, end=end, **kw)
                            if quoted > COST_ABORT_USD:
                                logger.error("ABORT: pull would cost $%.2f "
                                             "(window %s -> %s)",
                                             quoted, start, end)
                                return {"status": "aborted_cost",
                                        "quoted": quoted}
                        store = cli.timeseries.get_range(
                            start=start, end=end, **kw)
                        break
                    except Exception as e:
                        msg = str(e)
                        m = re.search(r"before (\S+?)Z", msg)
                        if "unavailable_range" in msg and m:
                            end = datetime.fromisoformat(
                                m.group(1)[:26]).replace(
                                tzinfo=timezone.utc) - timedelta(seconds=1)
                            quoted = None
                            if start >= end:
                                return {"status": "empty",
                                        "reason": "availability boundary"}
                            logger.info("clamped end to server boundary: %s",
                                        end)
                        elif "504" in msg or "50" in msg[:60]:
                            logger.warning("gateway 5xx (attempt %d), "
                                           "backing off", attempt + 1)
                            time.sleep(3 * (attempt + 1))
                        else:
                            raise
                if store is not None:
                    break
                # shrink the window and try again; next poll picks up the rest
                end = start + (end - start) / 2
                quoted = None
                if end - start < timedelta(minutes=2):
                    break
                logger.info("shrunk window to %s -> %s", start, end)
            if store is None:
                return {"status": "error", "reason": "gateway kept timing out"}

            raw_file = ""
            if self.keep_raw:
                d = LAKE / self.symbol.replace(".", "_") / self.schema
                d.mkdir(parents=True, exist_ok=True)
                raw_file = str(d / f"{start:%Y%m%d_%H%M%S}_{end:%H%M%S}.dbn.zst")
                store.to_file(raw_file)

            n_events, n_trades, samples, buckets = self._replay(store, vpin_after)
            self.book_day = end.date()

            self._write(conn, start, end, n_events, n_trades, quoted,
                        raw_file, samples, buckets)
            logger.info("%s: %s -> %s | %d events, %d trades, "
                        "%d samples, %d vpin buckets",
                        self.symbol, start, end, n_events, n_trades,
                        len(samples), len(buckets))
            return {"status": "ok", "events": n_events, "trades": n_trades,
                    "samples": len(samples), "buckets": len(buckets)}
        finally:
            conn.close()

    def _replay(self, store, vpin_after: Optional[datetime] = None):
        n_events = n_trades = 0
        samples: List[tuple] = []
        buckets: List[dict] = []
        next_sample = None
        mbp1 = self.schema == "mbp-1"
        for rec in store:
            n_events += 1
            ts = datetime.fromtimestamp(rec.ts_event / 1e9, tz=timezone.utc)
            action = chr(rec.action) if isinstance(rec.action, int) else rec.action
            side = chr(rec.side) if isinstance(rec.side, int) else rec.side
            if action == "T":
                n_trades += 1
                if vpin_after is None or ts > vpin_after:
                    buckets.extend(self.vpin.add_trade(side, rec.size, ts))
            elif not mbp1:
                self.book.apply(action, side, rec.order_id,
                                rec.price / 1e9, rec.size)
            if next_sample is None or ts >= next_sample:
                snap = (self._l1_snapshot(rec) if mbp1
                        else self.book.snapshot())
                if snap:
                    samples.append((ts, snap))
                    next_sample = ts + SAMPLE_EVERY
        return n_events, n_trades, samples, buckets

    @staticmethod
    def _l1_snapshot(rec):
        """MBP-1 records carry the top of book on every event — no
        reconstruction, just read it. Depth imbalance needs MBO (backfill)."""
        try:
            lvl = rec.levels[0]
        except (AttributeError, IndexError):
            return None
        bid_sz, ask_sz = int(lvl.bid_sz), int(lvl.ask_sz)
        if bid_sz <= 0 or ask_sz <= 0:
            return None
        return {
            "bid_px": lvl.bid_px / 1e9, "ask_px": lvl.ask_px / 1e9,
            "bid_sz": bid_sz, "ask_sz": ask_sz,
            "imbalance_l1": (bid_sz - ask_sz) / (bid_sz + ask_sz),
            "imbalance_d10": None,
            "n_bid": int(getattr(lvl, "bid_ct", 0)),
            "n_ask": int(getattr(lvl, "ask_ct", 0)),
        }

    def _write(self, conn, start, end, n_events, n_trades, quoted,
               raw_file, samples, buckets) -> None:
        from psycopg2.extras import execute_values
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO md.l3_polls (instrument, schema, ts_start,
                       ts_end, n_events, n_trades, quoted_cost, raw_file)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (self.symbol, self.schema, start, end, n_events, n_trades,
                 quoted, raw_file))
            if samples:
                execute_values(cur, """
                    INSERT INTO md.l3_book_state
                        (instrument, ts, bid_px, ask_px, bid_sz, ask_sz,
                         imbalance_l1, imbalance_d10, n_bid_orders, n_ask_orders)
                    VALUES %s
                    ON CONFLICT (instrument, ts) DO UPDATE SET
                        imbalance_l1 = EXCLUDED.imbalance_l1,
                        imbalance_d10 = EXCLUDED.imbalance_d10""",
                    [(self.symbol, ts, s["bid_px"], s["ask_px"], s["bid_sz"],
                      s["ask_sz"], s["imbalance_l1"], s["imbalance_d10"],
                      s["n_bid"], s["n_ask"]) for ts, s in samples])
            if buckets:
                execute_values(cur, """
                    INSERT INTO md.l3_vpin_buckets
                        (instrument, bucket_seq, ts_close, bucket_vol,
                         buy_vol, sell_vol)
                    VALUES %s ON CONFLICT DO NOTHING""",
                    [(self.symbol, b["seq"], b["ts"], b["vol"],
                      b["buy"], b["sell"]) for b in buckets])
        conn.commit()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="M6E.FUT",
                    help="parent symbol (M6E.FUT, ES.FUT)")
    ap.add_argument("--schema", default="mbp-1", choices=["mbp-1", "mbo"],
                    help="mbp-1: live loop (~10min lag). mbo: deep backfill "
                         "with full-depth imbalance (hours behind)")
    ap.add_argument("--vpin-bucket", type=int, default=500,
                    help="contracts per VPIN bucket (ES~500, M6E~100)")
    ap.add_argument("--once", action="store_true", help="single poll and exit")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                    help="poll forever at this interval")
    ap.add_argument("--no-raw", action="store_true",
                    help="don't keep raw DBN increments in the lake")
    args = ap.parse_args()

    p = L3Poller(args.symbol, schema=args.schema,
                 vpin_bucket=args.vpin_bucket, keep_raw=not args.no_raw)
    if args.loop:
        while True:
            try:
                p.poll()
            except Exception:
                logger.exception("poll failed; retrying next cycle")
            time.sleep(args.loop)
    else:
        r = p.poll()
        print(r)


if __name__ == "__main__":
    main()
