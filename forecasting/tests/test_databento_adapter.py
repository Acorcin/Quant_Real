"""
Adapter correctness: volume-bar aggregation, strict zero-lookahead scaling, and
the scaled-quantile -> absolute-price inversion. All synthetic, no real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import databento_adapter as dba


def _trades(prices, sizes, start="2026-01-05 09:00:00", spacing_s=1):
    ts = pd.date_range(start, periods=len(prices), freq=f"{spacing_s}s", tz="UTC")
    return pd.DataFrame({"ts": ts, "price": np.asarray(prices, float),
                         "size": np.asarray(sizes, np.int64)})


# ---------------------------------------------------------------------------
# volume bars
# ---------------------------------------------------------------------------

def test_volume_bars_ohlc_and_boundaries():
    # bar_size=10: trades of size 4 -> bars of 3 trades (12 contracts, >10 ok)
    t = _trades(prices=[1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 1.05, 1.15, 0.95],
                sizes=[4] * 9)
    bars = dba.volume_bars(t, bar_size=10)
    assert len(bars) == 3
    b0 = bars.iloc[0]
    assert (b0["open"], b0["high"], b0["low"], b0["close"]) == (1.0, 1.2, 0.9, 0.9)
    assert b0["volume"] == 12
    # trade never split: each bar accumulated past the 10-contract threshold
    assert (bars["volume"] >= 10).all()


def test_volume_bars_drops_forming_bar():
    t = _trades(prices=[1.0, 1.1, 1.2], sizes=[10, 10, 3])
    bars = dba.volume_bars(t, bar_size=10)
    assert len(bars) == 2                      # trailing 3-contract bar dropped
    assert bars["close"].iloc[-1] == 1.1


def test_volume_bars_diurnal_independence():
    """The property that motivated volume bars: bar count follows traded volume,
    not elapsed time. A 'session' with 10x volume gets 10x bars."""
    quiet = _trades([1.0] * 10, [5] * 10, start="2026-01-05 00:00:00",
                    spacing_s=600)
    busy = _trades([1.0] * 100, [5] * 100, start="2026-01-05 09:00:00",
                   spacing_s=6)
    bars = dba.volume_bars(pd.concat([quiet, busy], ignore_index=True),
                           bar_size=50)
    # 550 contracts -> 11 full bars, regardless of the 6h quiet stretch
    assert len(bars) == 11


def test_volume_bars_strictly_increasing_ts():
    t = _trades([1.0, 1.1, 1.2, 1.3], [10, 10, 10, 10])
    t.loc[1, "ts"] = t.loc[0, "ts"]            # two bars close on same tape ts
    t.loc[2, "ts"] = t.loc[0, "ts"]
    t.loc[3, "ts"] = t.loc[0, "ts"]
    bars = dba.volume_bars(t, bar_size=10)
    assert bars["ts"].is_monotonic_increasing
    assert bars["ts"].is_unique


# ---------------------------------------------------------------------------
# scaling: strict zero lookahead, drift preserved
# ---------------------------------------------------------------------------

def _fake_parquet(tmp_path, n_trades=30000, seed=7):
    rng = np.random.default_rng(seed)
    r = rng.normal(2e-5, 3e-4, n_trades)       # small positive drift, kept!
    prices = 1.10 * np.exp(np.cumsum(r))
    t = _trades(prices, rng.integers(1, 9, n_trades))
    p = tmp_path / "trades.parquet"
    t.rename(columns={"ts": "ts_event"}).to_parquet(p, index=False)
    return str(p)


def test_prepare_real_shapes_and_alignment(tmp_path):
    rs = dba.prepare_real(_fake_parquet(tmp_path), bar_size=100,
                          vol_window=64, min_sigma_bars=32)
    n = len(rs.scaled)
    assert len(rs.klines) == n
    assert len(rs.bar_ts) == n
    assert len(rs.sigma) == n + 1              # +1 trailing edge scaler
    assert len(rs.raw_returns) == n
    # row i of klines realized return i: close-to-close must reproduce it
    closes = rs.klines["close"].to_numpy()
    got = np.diff(np.log(closes))
    np.testing.assert_allclose(got, rs.raw_returns[1:], rtol=1e-12)
    # scaling relation, drift NOT removed
    np.testing.assert_allclose(rs.scaled, rs.raw_returns / rs.sigma[:-1],
                               rtol=1e-12)


def test_sigma_zero_lookahead(tmp_path):
    """Perturbing the FUTURE must not change past sigmas or scaled returns."""
    p = _fake_parquet(tmp_path)
    rs_full = dba.prepare_real(p, bar_size=100, vol_window=64,
                               min_sigma_bars=32)

    # rebuild from a truncated trade tape (drop the last 20% of trades)
    t = pd.read_parquet(p)
    cut = t.iloc[: int(len(t) * 0.8)]
    p2 = str(p).replace("trades.parquet", "trades_cut.parquet")
    cut.to_parquet(p2, index=False)
    rs_cut = dba.prepare_real(p2, bar_size=100, vol_window=64,
                              min_sigma_bars=32)

    k = len(rs_cut.scaled)
    np.testing.assert_allclose(rs_cut.scaled, rs_full.scaled[:k], rtol=1e-12)
    np.testing.assert_allclose(rs_cut.sigma[:-1], rs_full.sigma[:k], rtol=1e-12)


def test_to_series_roundtrip(tmp_path):
    """prep.prepare over to_series() must recover the scaled returns exactly --
    the invariant that lets the whole ladder run unchanged on real data."""
    from forecasting import prep
    rs = dba.prepare_real(_fake_parquet(tmp_path), bar_size=100,
                          vol_window=64, min_sigma_bars=32)
    prepared = prep.prepare(rs.to_series())
    np.testing.assert_allclose(prepared.returns, rs.scaled, rtol=1e-9)


# ---------------------------------------------------------------------------
# inversion: scaled quantiles -> absolute price levels
# ---------------------------------------------------------------------------

def test_manual_view_inversion_math(tmp_path):
    from forecasting.contracts import QUANTILE_LEVELS

    rs = dba.prepare_real(_fake_parquet(tmp_path), bar_size=100,
                          vol_window=64, min_sigma_bars=32)
    t = len(rs.scaled) - 2                      # an origin near the edge

    class Row:
        model, rung, origin_index, step = "kronos", 3, t, 1
        quantiles = {float(l): float(q) for l, q in
                     zip(QUANTILE_LEVELS, np.linspace(-1.5, 1.5,
                                                      len(QUANTILE_LEVELS)))}

    class BT:
        archive = [Row()]

    class Char:
        class _S:
            value = "nonlinear"
        structure_type = _S()

    view = dba.manual_trading_view(BT(), Char(), rs)
    p_origin = float(rs.klines["close"].iloc[t])
    sigma = float(rs.sigma[t + 1])
    q50 = Row.quantiles[0.5]
    expected_tgt = p_origin * np.exp(q50 * sigma)
    assert f"{expected_tgt:.5f}" in view
    assert f"{p_origin:.5f}" in view
    assert "nonlinear" in view
    # support below target below resistance (monotone quantiles)
    q10 = Row.quantiles[0.1]; q90 = Row.quantiles[0.9]
    assert (p_origin * np.exp(q10 * sigma) < expected_tgt
            < p_origin * np.exp(q90 * sigma))
