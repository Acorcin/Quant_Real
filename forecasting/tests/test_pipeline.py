"""
Tests that double as validation of the measurement instrument.

The synthetic generators have KNOWN structure; these tests assert the pipeline
recovers it. If the instrument mislabels a random walk as predictable, or fails to
detect volatility clustering, the whole veto gate is untrustworthy -- so these are
correctness tests, not just smoke tests.

Run:  python -m pytest forecasting/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from forecasting import data, prep, backtest, characterize, metrics, gate
from forecasting.contracts import (ForecastResult, QUANTILE_LEVELS, GateState,
                                   make_signal, VetoReason)
from forecasting.models import default_ladder


# ---- contract-level invariants -------------------------------------------

def test_forecastresult_sorts_crossed_quantiles():
    h = 3
    q = {float(l): np.full(h, 0.0) for l in QUANTILE_LEVELS}
    q[0.90] = np.full(h, -5.0)   # deliberately crossed
    q[0.10] = np.full(h, 5.0)
    fr = ForecastResult("m", 0, 0, np.zeros(h), q)
    for i in range(h):
        vals = [fr.quantiles[float(l)][i] for l in QUANTILE_LEVELS]
        assert all(vals[k] <= vals[k + 1] + 1e-9 for k in range(len(vals) - 1))


def test_gate_state_derives_from_reasons():
    s = make_signal(instrument="X", origin_index=0,
                    reasons=[VetoReason.DEGENERATE_FORECAST],
                    direction=0.0, conviction=0.5, vol_forecast=0.01,
                    regime="calm", calibration_ok=True,
                    forecast_degenerate=True, r2_oos=-0.1)
    assert s.gate == GateState.VETO
    s2 = make_signal(instrument="X", origin_index=0,
                     reasons=[VetoReason.LOW_CONVICTION],
                     direction=0.0, conviction=0.1, vol_forecast=0.01,
                     regime="calm", calibration_ok=True,
                     forecast_degenerate=False, r2_oos=0.1)
    assert s2.gate == GateState.GO_REDUCED
    s3 = make_signal(instrument="X", origin_index=0, reasons=[],
                     direction=0.0, conviction=0.9, vol_forecast=0.01,
                     regime="calm", calibration_ok=True,
                     forecast_degenerate=False, r2_oos=0.1)
    assert s3.gate == GateState.GO


# ---- leakage / windowing --------------------------------------------------

def test_forecast_window_immediately_follows_origin():
    # the test target must be origin+1 so a step-1 forecast is scored against the
    # step-1 realization (the alignment bug that made AR look worthless)
    from forecasting.windows import walk_forward
    for sp in walk_forward(500, horizon=5, min_train=100):
        assert sp.test_start == sp.origin + 1
        assert sp.test_stop == sp.origin + 1 + 5


def test_embargo_purges_training_right_edge():
    from forecasting.windows import walk_forward
    for sp in walk_forward(500, horizon=1, min_train=100, embargo=5):
        # training is purged back from the origin by the embargo, never overlaps test
        assert sp.train_stop == sp.origin + 1 - 5
        assert sp.train_stop <= sp.test_start - 5


def test_embargo_rejects_negative():
    from forecasting.windows import walk_forward
    with pytest.raises(ValueError):
        list(walk_forward(500, horizon=5, embargo=-1))


# ---- preprocessing is causal ---------------------------------------------

def test_anomaly_flags_are_trailing():
    # a spike at t should not flag points before t
    r = np.zeros(300)
    r[150] = 0.5
    flags = prep.flag_anomalies(r, window=30)
    assert not flags.any[:150].any()   # nothing before the spike is flagged
    # the spike itself is compared to its trailing window -> flagged
    assert flags.any[150]


def test_log_return_roundtrip():
    price = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200)))
    r = prep.log_returns(price)
    rebuilt = prep.reconstruct_price(price[0], r)
    assert np.allclose(rebuilt, price[1:], rtol=1e-9)


# ---- the instrument recovers ground truth --------------------------------

@pytest.mark.parametrize("gen,expect_efficient", [
    (data.gen_random_walk, True),
    (data.gen_ar1_returns, False),
])
def test_predictability_direction(gen, expect_efficient):
    series = gen(n=900, seed=1)
    prepared = prep.prepare(series)
    bt = backtest.run_backtest(prepared.returns, default_ladder(False),
                               instrument=series.instrument, min_train=252)
    # random walk: AR should NOT significantly beat naive; AR1: it should
    ar = next((r for r in bt.ladder if r.rung == 1), None)
    assert ar is not None
    if expect_efficient:
        assert not ar.beats_naive
    else:
        assert ar.r2_oos > 0  # positive out-of-sample skill on autocorrelated data


def test_garch_shows_vol_memory_not_mean_memory():
    series = data.gen_garch(n=1000, seed=2)
    prepared = prep.prepare(series)
    r = prepared.returns
    lb_ret = characterize.ljung_box(r)
    lb_sq = characterize.ljung_box(r ** 2)
    # returns look like white noise; squared returns do NOT (vol clustering)
    assert lb_sq < lb_ret
    assert lb_sq < 0.05


def test_hurst_random_walk_near_half():
    series = data.gen_random_walk(n=2000, seed=3)
    prepared = prep.prepare(series)
    h = characterize.hurst_exponent(np.cumsum(prepared.returns))
    assert 0.4 < h < 0.6


def test_pit_uniform_for_correct_distribution():
    # if forecasts are the true generating distribution, PIT ~ Uniform -> high p
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    from scipy.stats import norm
    q = {float(l): np.full(500, norm.ppf(l)) for l in QUANTILE_LEVELS}
    pit = metrics.pit_values(y, q)
    assert metrics.pit_uniformity_pvalue(pit) > 0.05


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 400)
    good = y + rng.normal(0, 0.1, 400)   # nearly perfect
    bad = rng.normal(0, 1, 400)          # useless
    stat, p = metrics.diebold_mariano(y, good, bad)
    assert stat < 0 and p < 0.05         # good (arg A) has lower loss -> negative


@pytest.mark.parametrize("gen,expected", [
    (data.gen_random_walk, "efficient"),
    (data.gen_ar1_returns, "linear"),
    (data.gen_garch, "vol_only"),
    (data.gen_regime_switch, "vol_only"),
])
def test_instrument_recovers_ground_truth_structure(gen, expected):
    """The measurement instrument must label each known series correctly. If this
    regresses, the veto gate's regime/horizon permissions become untrustworthy."""
    from forecasting.run import run_pipeline
    series = gen(n=1200, seed=0)
    out = run_pipeline(series, include_foundation=False, verbose=False)
    assert out["characterization"].structure_type.value == expected


def test_end_to_end_gate_runs():
    series = data.gen_regime_switch(n=1000, seed=4)
    prepared = prep.prepare(series)
    bt = backtest.run_backtest(prepared.returns, default_ladder(False),
                               instrument=series.instrument, min_train=252)
    char = characterize.characterize(series.instrument, prepared.returns,
                                     ladder=bt.ladder,
                                     r2_by_horizon=bt.r2_by_horizon("naive-rw"))
    signals = gate.build_signals(bt, char, prepared)
    assert len(signals) > 0
    gv = gate.validate_gate(signals, prepared.returns)
    assert 0.0 <= gv.veto_rate <= 1.0
