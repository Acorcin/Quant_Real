#!/usr/bin/env python3
"""
Demo-stack supervisor: runs the three loops that make paper trading live.

  l3_puller   — intraday MBP-1 microstructure (imbalance/VPIN)   every 5 min
  live_loop   — volume bars + Kronos forecast per bar            every 60 s
  demo_loop   — meta-model decision -> Tradovate DEMO orders     every 60 s

Each child is restarted if it dies (with backoff). Creating data/KILL
flattens the position (demo_loop handles it) and shuts the stack down.

    python run_demo.py                     # honest mode (veto respected)
    python run_demo.py --mechanics-test    # exercise order routing anyway
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
KILL = ROOT / "data" / "KILL"

ENV = {
    **os.environ,
    "QUANT_DB_URL": os.environ.get(
        "QUANT_DB_URL", "postgresql://postgres:postgres@localhost:5433/quant_eod"),
    "DB_HOST": "localhost", "DB_PORT": "5433", "DB_NAME": "quant_eod",
    "DB_USER": "postgres", "DB_PASSWORD": "postgres",
    "PYTHONIOENCODING": "utf-8",
}


def specs(mechanics: bool):
    demo = [PY, "demo_loop.py", "--loop", "60"]
    if mechanics:
        demo.append("--mechanics-test")
    return [
        ("l3_puller", ROOT,
         [PY, "-m", "forecasting.l3_puller", "--symbol", "M6E.FUT",
          "--vpin-bucket", "100", "--loop", "300"]),
        ("live_loop", ROOT,
         [PY, "-m", "forecasting.live_loop", "--symbol", "M6E.FUT",
          "--instrument", "M6EU6", "--loop", "60"]),
        ("physics_feed", ROOT,
         [PY, "-m", "forecasting.physics_feed", "--symbol", "M6E.FUT",
          "--instrument", "M6EU6", "--loop", "60"]),
        ("demo_loop", ROOT / "engine", demo),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mechanics-test", action="store_true")
    args = ap.parse_args()

    procs: dict[str, subprocess.Popen] = {}
    restarts: dict[str, int] = {}

    def spawn(name, cwd, cmd):
        (ROOT / "logs").mkdir(exist_ok=True)
        log = open(ROOT / "logs" / f"{name}.log", "a")
        p = subprocess.Popen(cmd, cwd=str(cwd), env=ENV,
                             stdout=log, stderr=subprocess.STDOUT)
        procs[name] = p
        print(f"[supervisor] {name} up (pid {p.pid})", flush=True)

    def shutdown(*_):
        print("[supervisor] shutting down...", flush=True)
        for name, p in procs.items():
            if p.poll() is None:
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for name, cwd, cmd in specs(args.mechanics_test):
        spawn(name, cwd, cmd)

    while True:
        time.sleep(10)
        if KILL.exists():
            print("[supervisor] KILL file found — letting demo_loop flatten, "
                  "then stopping", flush=True)
            time.sleep(90)          # give demo_loop a cycle to flatten
            shutdown()
        for name, cwd, cmd in specs(args.mechanics_test):
            p = procs.get(name)
            if p and p.poll() is not None:
                restarts[name] = restarts.get(name, 0) + 1
                delay = min(60, 5 * restarts[name])
                print(f"[supervisor] {name} exited (code {p.returncode}); "
                      f"restart #{restarts[name]} in {delay}s", flush=True)
                time.sleep(delay)
                spawn(name, cwd, cmd)


if __name__ == "__main__":
    main()
