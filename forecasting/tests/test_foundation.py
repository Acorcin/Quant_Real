"""
Unit tests for the FoundationForecaster kronos adapter -- the returns -> price
path -> sampled futures -> return-quantiles conversion. The Kronos predictor is
stubbed so these run fast, deterministically, and without torch/weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.contracts import QUANTILE_LEVELS
from forecasting.models.foundation import FoundationForecaster


class StubKronosPipe:
    """Mimics KronosPredictor.predict_batch: returns one pred DataFrame per input
    df, each a deterministic multiple of the last context close."""

    def __init__(self, factors):
        self.factors = factors          # one growth factor per requested path
        self.calls = []

    def predict_batch(self, df_list, x_ts_list, y_ts_list, pred_len, **kw):
        self.calls.append({"n": len(df_list), "pred_len": pred_len, **kw})
        last = float(df_list[0]["close"].iloc[-1])
        outs = []
        for i in range(len(df_list)):
            f = self.factors[i % len(self.factors)]
            closes = last * np.cumprod(np.full(pred_len, f))
            outs.append(pd.DataFrame({"close": closes}))
        return outs


def _forecaster_with_stub(factors, num_paths):
    fc = FoundationForecaster(backend="kronos", num_paths=num_paths)
    fc._pipe = StubKronosPipe(factors)
    fc._active = "kronos"
    fc.name = "kronos"
    return fc


def test_kronos_path_to_return_conversion():
    """Sampled price paths must convert to log-returns vs the LAST CONTEXT close
    for step 1 and consecutive predicted closes for later steps."""
    # all paths grow 1% per step -> every quantile must equal log(1.01) each step
    fc = _forecaster_with_stub(factors=[1.01], num_paths=8)
    ctx = np.random.default_rng(0).normal(0, 0.01, 300)
    res = fc.predict(ctx, horizon=3)
    for l in QUANTILE_LEVELS:
        np.testing.assert_allclose(res.quantiles[float(l)],
                                   np.log(1.01), rtol=1e-10)
    np.testing.assert_allclose(res.mean, np.log(1.01), rtol=1e-10)


def test_kronos_quantiles_monotone_and_spread():
    """Distinct sampled paths -> monotone, non-degenerate quantile grid."""
    fc = _forecaster_with_stub(factors=[0.98, 0.99, 1.00, 1.01, 1.02],
                               num_paths=25)
    ctx = np.zeros(300)
    res = fc.predict(ctx, horizon=1)
    vals = [float(res.quantiles[float(l)][0]) for l in QUANTILE_LEVELS]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))
    assert vals[-1] - vals[0] > 0, "distinct paths must yield spread"
    # median of symmetric-in-log factors around 1.00 -> ~log(1.00) = 0
    assert abs(float(res.quantiles[0.50][0])) < np.log(1.01)


def test_kronos_context_truncated_and_batched():
    """Adapter must truncate context to context_length and request num_paths
    single-sample draws (the public predict() would average them away)."""
    fc = _forecaster_with_stub(factors=[1.0], num_paths=7)
    fc.context_length = 64
    res = fc.predict(np.zeros(500), horizon=2)
    call = fc._pipe.calls[-1]
    assert call["n"] == 7
    assert call["pred_len"] == 2
    assert call["sample_count"] == 1
    assert res.mean.shape == (2,)


def test_kronos_nonpositive_price_guard():
    """A sampled path that collapses to <= 0 must not produce NaN/inf returns."""
    fc = _forecaster_with_stub(factors=[-0.5, 1.0], num_paths=4)
    res = fc.predict(np.zeros(300), horizon=1)
    for l in QUANTILE_LEVELS:
        assert np.all(np.isfinite(res.quantiles[float(l)]))
