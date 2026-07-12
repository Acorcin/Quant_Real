"""
Real-data adapter: Databento CME trades -> volume bars -> the dual-plane pipeline.

Input contract: a *trades-schema* parquet (raw executed trades only -- ts, price,
size). MBO/L3 stays out of local memory; distilling trades from an MBP file is a
one-time upstream step (any row with action == 'T').

Three jobs, in order:

1. **Volume bars, not time bars.** A new candle every `bar_size` contracts.
   Time bars breathe with the diurnal volume cycle -- overnight CME sessions
   produce long stretches of tiny moves whose clustered "calm" reads as spurious
   volatility structure in the GARCH rung (false `vol_only`). Sampling in volume
   time makes bars roughly IID in information content (Ané & Geman: returns are
   closer to Gaussian in volume time), which is exactly what the ladder's
   significance tests assume.

2. **Conditioning for the statistical ladder.** Log returns of bar closes,
   scaled by a trailing `vol_window`-bar sigma (shifted one bar: the scaler for
   the return realized at bar t uses bars <= t-1 ONLY -- strict zero lookahead).
   NOT mean-centered: drift is signal, removing it would erase exactly the edge
   the ladder is trying to measure. A unit-variance series is also what `arch`
   likes numerically. The REAL candles are carried alongside, aligned 1:1 with
   the scaled returns, so the Kronos rung can see true wick/body geometry.

3. **Bifurcated outputs.** One walk-forward, two consumers:
   - XGBoost features (stationary plane): scaled quantiles, spread, sigma,
     structure verdict, gate state -- everything dimensionless.
   - Manual trading view (price plane): quantiles inverted back to absolute CME
     prices via P_target = P_origin * exp(q_scaled * sigma) -- support / target /
     resistance an operator can put on a chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .data import Series


# ---------------------------------------------------------------------------
# Trades -> volume bars
# ---------------------------------------------------------------------------

def load_trades(path: str) -> pd.DataFrame:
    """Load a trades-schema parquet -> DataFrame[ts, price, size], UTC-sorted.

    Accepts Databento column names (ts_event/ts_recv) or plain ts. Prices that
    look like raw DBN fixed-precision ints (1e-9 scale) are converted."""
    df = pd.read_parquet(path)
    ts_col = next((c for c in ("ts_event", "ts_recv", "ts", "timestamp")
                   if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"no timestamp column in {list(df.columns)}")
    if "price" not in df.columns or "size" not in df.columns:
        raise ValueError("trades parquet needs 'price' and 'size' columns")
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col], utc=True),
        "price": df["price"].astype(float),
        "size": df["size"].astype(np.int64),
    })
    # raw DBN int prices are scaled 1e-9 (a $1.17 FX future shows as 1.17e9)
    if out["price"].median() > 1e6:
        out["price"] = out["price"] * 1e-9
    out = out[(out["price"] > 0) & (out["size"] > 0)]
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def volume_bars(trades: pd.DataFrame, bar_size: int) -> pd.DataFrame:
    """Aggregate trades into OHLCV bars of ~`bar_size` contracts.

    A trade is never split: it belongs to the bar that was open when it hit the
    tape (bars can exceed bar_size by up to one trade -- standard practice).
    The final partial bar is DROPPED: it is still forming and using it would put
    an incomplete observation at the series edge. Bar timestamp = last trade's
    (the bar close time; strictly increasing, nudged by 1ns on ties)."""
    if bar_size <= 0:
        raise ValueError("bar_size must be positive")
    # accumulate-until-threshold with reset (NOT floor-division binning, which
    # lets bars undershoot): each bar closes on the trade that pushes its own
    # running volume to >= bar_size.
    cum = trades["size"].cumsum().to_numpy()
    bar_id = np.full(len(trades), -1, dtype=np.int64)
    start, bar, base = 0, 0, 0
    while start < len(cum):
        pos = int(np.searchsorted(cum, base + bar_size, side="left"))
        if pos >= len(cum):          # tape ended before the bar filled
            break                    # -> trailing trades stay id -1 (forming)
        bar_id[start:pos + 1] = bar
        base = cum[pos]
        start, bar = pos + 1, bar + 1

    kept = trades[bar_id >= 0]
    g = kept.groupby(bar_id[bar_id >= 0], sort=True)
    bars = pd.DataFrame({
        "ts": g["ts"].last(),
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "volume": g["size"].sum(),
    }).reset_index(drop=True)

    # strictly-increasing close times (two bars can close on the same tape ts)
    ts = bars["ts"].to_numpy()
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + np.timedelta64(1, "ns")
    bars["ts"] = ts
    return bars


# ---------------------------------------------------------------------------
# Conditioning: scaled returns + aligned K-lines
# ---------------------------------------------------------------------------

@dataclass
class RealSeries:
    """Everything the pipeline needs from real data, aligned so that row i of
    `klines` produced return `scaled[i]` (realized at `bar_ts[i]`).

    `sigma` has length len(scaled)+1: sigma[i] is the trailing scaler for
    return i, and the extra final element is the scaler for the NEXT, not yet
    realized return -- what live inversion at the series edge needs."""

    instrument: str
    bar_ts: pd.DatetimeIndex        # close time of the bar realizing scaled[i]
    klines: pd.DataFrame            # open/high/low/close/volume, row i <-> scaled[i]
    scaled: np.ndarray              # r_i / sigma_i  (unit-ish variance, drift kept)
    sigma: np.ndarray               # len = len(scaled) + 1, strictly trailing
    raw_returns: np.ndarray         # r_i = ln(close_i / close_{i-1})
    bar_size: int
    vol_window: int

    def to_series(self) -> Series:
        """Package the SCALED returns as a price Series for prep/backtest.

        price = exp(cumsum(scaled)) with a base row prepended, so prep's
        log_returns recovers `scaled` exactly -- backtest, characterization and
        gate all run on the conditioned series through the standard code path."""
        px = np.concatenate([[1.0], np.exp(np.cumsum(self.scaled))])
        t0 = self.bar_ts[0] - (self.bar_ts[1] - self.bar_ts[0]
                               if len(self.bar_ts) > 1 else pd.Timedelta("1min"))
        idx = pd.DatetimeIndex([t0]).append(self.bar_ts)
        return Series(self.instrument, idx, px)


def prepare_real(path: str, *, bar_size: int = 250, vol_window: int = 256,
                 min_sigma_bars: int = 64, instrument: str = "") -> RealSeries:
    """trades parquet -> volume bars -> trailing-vol-scaled returns + K-lines."""
    trades = load_trades(path)
    bars = volume_bars(trades, bar_size)
    if len(bars) < min_sigma_bars + 10:
        raise ValueError(f"only {len(bars)} bars; need > {min_sigma_bars + 10} "
                         f"(reduce bar_size or supply more trades)")

    close = bars["close"].to_numpy()
    r = np.diff(np.log(close))                       # r[i] realized at bar i+1

    # trailing sigma with STRICT zero lookahead: the scaler for r[i] sees only
    # returns r[..i-1] (shift(1) before rolling). One extra step at the end is
    # the scaler for the next, unrealized return (live inversion at the edge).
    s = pd.Series(np.concatenate([r, [np.nan]]))
    sigma_full = (s.shift(1)
                   .rolling(vol_window, min_periods=min_sigma_bars)
                   .std(ddof=1)
                   .to_numpy())

    valid = np.isfinite(sigma_full[:-1]) & (sigma_full[:-1] > 0)
    start = int(np.argmax(valid))                    # first usable return index
    if not valid[start:].all():
        raise ValueError("trailing sigma has interior gaps; data too sparse")

    r_v = r[start:]
    sigma_v = sigma_full[start:]                     # len(r_v) + 1 (extra edge)
    scaled = r_v / sigma_v[:-1]

    # klines: bar producing r[i] is bars[i+1]; slice to the valid range
    klines = bars.iloc[1 + start:][["open", "high", "low", "close",
                                    "volume"]].reset_index(drop=True)
    bar_ts = pd.DatetimeIndex(bars["ts"].iloc[1 + start:]).tz_convert("UTC") \
        if bars["ts"].dt.tz is not None else \
        pd.DatetimeIndex(bars["ts"].iloc[1 + start:], tz="UTC")

    name = instrument or path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].split(".")[0]
    return RealSeries(instrument=name, bar_ts=bar_ts, klines=klines,
                      scaled=scaled, sigma=sigma_v, raw_returns=r_v,
                      bar_size=bar_size, vol_window=vol_window)


# ---------------------------------------------------------------------------
# Bifurcated outputs
# ---------------------------------------------------------------------------

def xgboost_features(bt, char, signals, real: RealSeries) -> pd.DataFrame:
    """The stationary plane, one row per gate decision (forecast origin).

    Everything is dimensionless / strictly point-in-time EXCEPT structure_type,
    which is the run-level verdict (full-sample characterization) -- treat it as
    a slowly-varying regime descriptor, not a per-bar causal feature."""
    from .contracts import QUANTILE_LEVELS
    by_model_origin = {}
    for row in bt.archive:
        if row.step == 1:
            by_model_origin.setdefault(row.origin_index, {})[row.model] = row

    lo_l = float(min(QUANTILE_LEVELS, key=lambda l: abs(l - 0.10)))
    hi_l = float(min(QUANTILE_LEVELS, key=lambda l: abs(l - 0.90)))

    rows = []
    for sig in signals:
        t = sig.origin_index
        models = by_model_origin.get(t, {})
        target = t + 1                               # step-1 target return index
        rec = {
            "origin_index": t,
            "ts": real.bar_ts[min(t, len(real.bar_ts) - 1)],
            "sigma_256": real.sigma[min(target, len(real.sigma) - 1)],
            "structure_type": char.structure_type.value,
            "gate_state": sig.gate.value,
            "veto_reasons": "|".join(r.value for r in sig.veto_reasons),
            "y_true_scaled": models[next(iter(models))].y_true if models else np.nan,
        }
        for name, row in models.items():
            key = name.replace("-", "_")
            rec[f"{key}_p50_scaled"] = row.quantiles.get(0.5, np.nan)
            rec[f"{key}_spread_scaled"] = (row.quantiles.get(hi_l, np.nan)
                                           - row.quantiles.get(lo_l, np.nan))
        rows.append(rec)
    return pd.DataFrame(rows)


def manual_trading_view(bt, char, real: RealSeries,
                        model: Optional[str] = None,
                        levels=(0.10, 0.50, 0.90)) -> str:
    """The price plane: invert the newest origin's quantiles into absolute CME
    price levels via P_target = P_origin * exp(q_scaled * sigma_next)."""
    from .contracts import QUANTILE_LEVELS
    # prefer the foundation rung's distribution (it saw real candles), fall back
    # to the highest rung present
    step1 = [r for r in bt.archive if r.step == 1]
    if model is None:
        model = max(step1, key=lambda r: r.rung).model
    rows = [r for r in step1 if r.model == model]
    last = max(rows, key=lambda r: r.origin_index)

    t = last.origin_index
    p_origin = float(real.klines["close"].iloc[min(t, len(real.klines) - 1)])
    sigma = float(real.sigma[min(t + 1, len(real.sigma) - 1)])

    def level(q):
        l = float(min(QUANTILE_LEVELS, key=lambda x: abs(x - q)))
        return p_origin * float(np.exp(last.quantiles[l] * sigma))

    sup, tgt, res = (level(q) for q in levels)
    ts = real.bar_ts[min(t, len(real.bar_ts) - 1)]
    return (
        f"\n--- Manual trading view ({real.instrument}, {model}, "
        f"origin {ts:%Y-%m-%d %H:%M:%S} UTC) ---\n"
        f"  structure type    : {char.structure_type.value}\n"
        f"  last close (P_origin) : {p_origin:.5f}\n"
        f"  absolute support  (P10): {sup:.5f}\n"
        f"  expected target   (P50): {tgt:.5f}\n"
        f"  resistance        (P90): {res:.5f}\n"
        f"  bar sigma (next)  : {sigma:.2e}  [{real.vol_window}-bar trailing]"
    )
