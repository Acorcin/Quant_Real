"""
Physics-layer correctness on the CME trade path: spike rejection, Kalman
tracking, regime clipping, and state round-trip. Pure in-memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
from physics.engine import PhysicsEngine  # noqa: E402


def _run_ticks(engine, prices, t0=1_752_600_000.0, dt=1.0, **kw):
    outs = []
    for i, p in enumerate(prices):
        outs.append(engine.process_trade(float(p), t0 + i * dt, **kw))
    return outs


def test_spike_rejected_and_substituted():
    eng = PhysicsEngine(spike_window=50, spike_sigma=4.0)
    prices = [1.1440 + 0.0001 * np.sin(i / 3) for i in range(30)]
    prices.append(1.2900)                       # absurd single print
    outs = _run_ticks(eng, prices)
    spike = outs[-1]
    assert spike["is_spike"] is True or spike["is_spike"] == True  # noqa: E712
    # substituted with the rolling median, not the bad print
    assert abs(spike["filtered_mid"] - 1.1440) < 0.001
    # Kalman price never saw the spike
    assert abs(spike["kalman_price"] - 1.1440) < 0.001


def test_genuine_level_shift_eventually_accepted():
    """Out-of-band prints trip the escape hatch every 5th tick, and with a
    sustained shift the rolling median converges to the new level."""
    eng = PhysicsEngine(spike_window=50, spike_sigma=4.0)
    _run_ticks(eng, [1.1440] * 30)
    outs = _run_ticks(eng, [1.1600] * 5, t0=1_752_600_100.0)
    assert outs[0]["is_spike"]                  # first prints rejected
    assert not outs[4]["is_spike"]              # 5th: forced accept (hatch)
    assert abs(outs[4]["filtered_mid"] - 1.1600) < 1e-9
    # sustained shift: the window fills with the new level and rejection ends
    outs = _run_ticks(eng, [1.1600] * 250, t0=1_752_600_200.0)
    assert not outs[-1]["is_spike"]
    assert abs(outs[-1]["filtered_mid"] - 1.1600) < 1e-9


def test_kalman_tracks_trend_with_velocity():
    eng = PhysicsEngine()
    # steady +0.0001/tick trend, 1s apart
    prices = [1.1400 + 0.0001 * i for i in range(120)]
    outs = _run_ticks(eng, prices)
    last = outs[-1]
    assert abs(last["kalman_price"] - prices[-1]) < 0.0005   # tracks level
    assert last["kalman_velocity"] > 0                        # sees the drift
    assert abs(last["kalman_velocity"] - 0.0001) < 0.0001     # ~per-second rate


def test_regime_clip_bounds():
    """The clip mapping itself, incl. the new calm/turbulent labels — Kalman
    stubbed so a known 1% move reaches the clipper (normalized = 20)."""
    for regime, bound in [("calm", 2.5), ("turbulent", 3.5),
                          ("low_vol", 2.5), ("high_vol_crash", 5.0),
                          ("unknown_label", 3.5)]:            # default bucket
        eng = PhysicsEngine()
        eng.prev_smoothed_price = 1.0
        eng.prev_time = 1_752_600_000.0
        eng.kalman_filter.update = lambda price, dt: (1.01, 0.0)  # +1% move
        out = eng.process_trade(1.01, 1_752_600_001.0,
                                regime_label=regime, daily_scale=0.0005)
        assert out["clipped_return"] == pytest.approx(bound), regime
        assert out["normalized_return"] == pytest.approx(20.0, rel=1e-6)


def test_state_roundtrip_resumes_identically():
    a = PhysicsEngine()
    prices = list(1.1440 + 0.0002 * np.sin(np.arange(60) / 5))
    _run_ticks(a, prices)
    state = a.get_state()

    b = PhysicsEngine()
    b.set_state(state)
    nxt = 1.1443
    out_a = a.process_trade(nxt, 1_752_600_060.0)
    out_b = b.process_trade(nxt, 1_752_600_060.0)
    for k in ("kalman_price", "kalman_velocity", "tick_return", "is_spike"):
        assert out_a[k] == pytest.approx(out_b[k]), k
