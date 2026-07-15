#!/usr/bin/env python3
"""
Demo decision loop: md forecasts -> meta-model -> Tradovate DEMO orders.

Each cycle:
  1. Bridge features: latest live Kronos forecast (md.kronos_features via
     load_latest_kronos) + latest L3 microstructure (md.v_l3_latest).
     Stale forecasts (older than --max-age) stand the loop down — never
     trade on an old opinion.
  2. Tier-1 kronos_directional_signal proposes a direction; the feature
     vector is assembled (technical/macro default to 0 in this phase — the
     live columns are kronos_* + l3_*); MetaModel.predict applies the
     structural veto and the 0.55/0.70 sizing gates.
  3. Target position = direction * size_mult * --base-qty, diffed against
     the actual Tradovate net position -> market order for the difference.

Safety rails (all hard):
  * TradovateClient refuses orders unless env is demo (ALLOW_LIVE=False).
  * --max-pos clamp on absolute position; --max-orders per session.
  * kill file (data/KILL next to the repo root): flatten and exit.
  * CME maintenance window (21:00-22:05 UTC) stands down.

MECHANICS-TEST MODE (--mechanics-test): the M6E series measures 'efficient',
so the structural veto correctly forces flat and a faithful demo would never
trade. This flag trades the tier-1 direction at 1 lot ANYWAY, purely to
exercise order routing end-to-end. Every such order is logged as
MECHANICS_TEST; it is not a strategy.

    DB_PORT=5433 python demo_loop.py --once
    DB_PORT=5433 python demo_loop.py --loop 60 --mechanics-test
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo_loop")

KILL_FILE = Path(__file__).resolve().parents[1] / "data" / "KILL"
DECISION_LOG = Path(__file__).resolve().parent / "logs" / "demo_decisions.jsonl"


def in_maintenance(now: datetime) -> bool:
    """CME daily maintenance 21:00-22:05 UTC (16:00 CT close -> 17:05 reopen)."""
    hhmm = now.hour * 100 + now.minute
    return 2100 <= hhmm <= 2205


class DemoLoop:
    def __init__(self, args):
        self.args = args
        self.orders_placed = 0
        from execution.tradovate_client import TradovateClient
        from models.meta_model import MetaModel
        self.tv = TradovateClient()
        self.meta = MetaModel()
        self.contract = self.tv.contract_find(args.contract)
        logger.info("demo loop: %s (contract id %s), base qty %d, max pos %d%s",
                    args.contract, self.contract["id"], args.base_qty,
                    args.max_pos,
                    " [MECHANICS-TEST MODE]" if args.mechanics_test else "")

    # -- position helpers -----------------------------------------------------

    def net_position(self) -> int:
        for p in self.tv.positions():
            if p.get("contractId") == self.contract["id"]:
                return int(p.get("netPos", 0))
        return 0

    def order_to(self, target: int, current: int, reason: str):
        delta = target - current
        if delta == 0:
            return
        if self.orders_placed >= self.args.max_orders:
            logger.error("max orders (%d) reached — standing down",
                         self.args.max_orders)
            return
        action = "Buy" if delta > 0 else "Sell"
        qty = min(abs(delta), self.args.max_pos * 2)
        result = self.tv.place_order(action=action, symbol=self.args.contract,
                                     qty=qty)
        self.orders_placed += 1
        logger.info("ORDER: %s %d %s (%s -> %s) [%s] id=%s",
                    action, qty, self.args.contract, current, target, reason,
                    result.get("orderId"))

    # -- one decision cycle ---------------------------------------------------

    def cycle(self) -> dict:
        now = datetime.now(timezone.utc)
        if KILL_FILE.exists():
            cur = self.net_position()
            if cur != 0:
                logger.warning("KILL file present — flattening %+d", cur)
                self.order_to(0, cur, "kill-switch")
            return {"status": "killed"}
        if in_maintenance(now):
            return {"status": "maintenance"}

        from features.kronos_bridge import load_latest_kronos, load_latest_l3
        from config.settings import KRONOS_INSTRUMENT, L3_INSTRUMENT
        import features.vector as fv
        from signals.tier1 import kronos_directional_signal

        kronos = load_latest_kronos(KRONOS_INSTRUMENT)
        decision = {"ts": now.isoformat(), "kronos_ts": None,
                    "direction": "flat", "size_mult": 0.0, "prob": None,
                    "veto": None, "mode": "normal", "target": 0}

        # staleness gate: volume bars breathe with activity, but a forecast
        # older than max-age minutes is an old opinion — stand down flat.
        if kronos and kronos.get("ts") is not None:
            age_min = (now - kronos["ts"]).total_seconds() / 60
            decision["kronos_ts"] = kronos["ts"].isoformat()
            if age_min > self.args.max_age:
                logger.info("forecast is %.0f min old (> %d) — standing down",
                            age_min, self.args.max_age)
                kronos = None
        if kronos is None:
            self._settle(decision, reason="no fresh forecast")
            return decision

        l3 = load_latest_l3(L3_INSTRUMENT)
        t1 = kronos_directional_signal(kronos)
        dir_enc = {"long": 1, "short": -1}.get(t1["direction"], 0)

        fv_get_macro = fv._get_macro_data
        try:
            fv._get_macro_data = lambda d: {}     # no FRED in the demo loop
            vec = fv.assemble_feature_vector(
                now.date(), self.args.contract, {}, {}, {
                    "composite": {"direction_encoded": dir_enc,
                                  "signal_count": 1 if dir_enc else 0,
                                  "composite_strength": t1["strength"],
                                  "tier2_count": 0}},
                kronos=kronos, l3=l3)
        finally:
            fv._get_macro_data = fv_get_macro

        pred = self.meta.predict(vec)
        decision.update(direction=pred["direction"],
                        size_mult=pred["size_multiplier"],
                        prob=pred["probability"],
                        veto=bool(pred.get("structural_veto", False)))

        target = int(round({"long": 1, "short": -1, "flat": 0}[pred["direction"]]
                           * pred["size_multiplier"] * self.args.base_qty))

        if (target == 0 and self.args.mechanics_test and dir_enc != 0):
            target = dir_enc            # 1 lot, routing exercise only
            decision["mode"] = "MECHANICS_TEST"
            logger.warning("MECHANICS_TEST: overriding %s to %+d lot to "
                           "exercise routing (not a strategy signal)",
                           "veto" if decision["veto"] else "flat", target)

        target = max(-self.args.max_pos, min(self.args.max_pos, target))
        decision["target"] = target
        self._settle(decision, reason=decision["mode"])
        return decision

    def _settle(self, decision: dict, reason: str):
        current = self.net_position()
        decision["position_before"] = current
        self.order_to(decision["target"], current, reason)
        DECISION_LOG.parent.mkdir(exist_ok=True)
        with open(DECISION_LOG, "a") as f:
            f.write(json.dumps(decision, default=str) + "\n")
        logger.info("decision: dir=%s prob=%s size=%.1f veto=%s target=%+d "
                    "(pos was %+d)", decision["direction"], decision["prob"],
                    decision["size_mult"], decision["veto"],
                    decision["target"], current)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", default="M6EU6")
    ap.add_argument("--base-qty", type=int, default=2,
                    help="contracts at full size (half size rounds down)")
    ap.add_argument("--max-pos", type=int, default=2)
    ap.add_argument("--max-orders", type=int, default=20,
                    help="per-session order cap")
    ap.add_argument("--max-age", type=int, default=45,
                    help="minutes before a forecast is too stale to act on")
    ap.add_argument("--mechanics-test", action="store_true",
                    help="trade 1 lot on tier-1 direction even when the "
                         "structural veto says flat (routing exercise only)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS")
    args = ap.parse_args()

    loop = DemoLoop(args)
    if args.loop:
        while True:
            try:
                r = loop.cycle()
                logger.info("cycle: %s", r.get("status", "ok"))
            except Exception:
                logger.exception("cycle failed; retrying next interval")
            time.sleep(args.loop)
    else:
        print(json.dumps(loop.cycle(), default=str, indent=2))


if __name__ == "__main__":
    main()
