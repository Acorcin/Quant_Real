"""
Rung 3: foundation models (Kronos / Chronos / TimesFM), zero-shot.

These are the top rung of the ladder: if they beat the linear and GARCH rungs on a
significant Diebold-Mariano test, the series contains NONLINEAR structure the
simpler models cannot capture. If they only match the lower rungs, they add cost
without information -- an important, money-saving finding.

Key handling notes baked in:

  * Kronos (github.com/shiyu-coder/Kronos) is the finance-specific backend and the
    default first choice: trained on K-lines from 45+ exchanges. It wants an OHLCV
    DataFrame, so we reconstruct a price path from the returns context (degenerate
    O=H=L=C candles, zero volume) and convert its sampled price paths back to
    return quantiles. Its public `predict` averages samples away, so we instead
    batch the SAME series `num_paths` times through `predict_batch` at
    sample_count=1 -- N independent draws in one forward pass.
  * Chronos self-scales internally (mean scaling + quantization). We therefore feed
    RAW returns and do NOT pre-standardize -- double-scaling degrades it.
  * We take the model's native quantile forecasts (Chronos-Bolt emits quantiles
    directly; classic Chronos samples a predictive distribution we quantize).
  * Context is truncated to the model's supported length; longer history is not
    always better in finance (old regimes are noise) -- Probe 4 sweeps this.
  * If no backend is installed/available, we raise SkipModel so the statistical
    ladder still runs everywhere (CPU, no network).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster, SkipModel
from ..contracts import QUANTILE_LEVELS


def _call_with_series(fn, series, **kwargs):
    """Call a Chronos/TimesFM method whose first argument is the input series,
    tolerating parameter-name churn across releases.

    Chronos-forecasting has renamed this argument between versions
    ('context' -> 'inputs'), and different builds accept it positionally or only
    by keyword. Rather than pin a version, we try each convention in turn and use
    whichever binds. Falls back to introspecting the signature for the first
    non-self parameter name.
    """
    import inspect

    # 1) positional (works when it's a normal positional-or-keyword parameter)
    try:
        return fn(series, **kwargs)
    except TypeError as e:
        last = e
    # 2) explicit keyword names seen across releases
    for name in ("inputs", "context", "series"):
        try:
            return fn(**{name: series}, **kwargs)
        except TypeError as e:
            last = e
    # 3) introspect: bind to the first real parameter name
    try:
        params = [p for p in inspect.signature(fn).parameters
                  if p not in ("self", "kwargs", "args")]
        if params:
            return fn(**{params[0]: series}, **kwargs)
    except (TypeError, ValueError) as e:
        last = e
    raise last


class FoundationForecaster(Forecaster):
    name = "foundation"
    rung = 3
    stateful = False

    def __init__(self, backend: str = "auto", context_length: int = 512,
                 chronos_model: str = "amazon/chronos-bolt-base",
                 num_samples: int = 256, device: str = "auto",
                 kronos_model: str = "NeoQuasar/Kronos-small",
                 kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base",
                 kronos_repo: str = "", num_paths: int = 32):
        self.backend = backend
        self.context_length = context_length
        self.chronos_model = chronos_model
        self.num_samples = num_samples
        self.device = device
        self.kronos_model = kronos_model
        self.kronos_tokenizer = kronos_tokenizer
        self.kronos_repo = kronos_repo      # path to the cloned Kronos repo
        self.num_paths = num_paths          # sampled paths per origin (kronos)
        self._pipe = None
        self._active = None
        self._klines = None                 # real candles aligned to returns
        self._kl_ts = None
        self._kl_sigma = None               # trailing scaler, len(returns)+1

    def attach_klines(self, real) -> "FoundationForecaster":
        """Give the kronos backend the REAL candles behind the scaled series.

        `real` is a databento_adapter.RealSeries: klines row i produced scaled
        return i, and sigma[i] is the trailing scaler for return i (one extra
        trailing element for the next unrealized return). With this attached,
        kronos sees true wick/body geometry instead of reconstructed degenerate
        candles, and its sampled price paths are converted back into the SAME
        scaled-return space the rest of the ladder is scored in."""
        self._klines = real.klines.reset_index(drop=True)
        self._kl_ts = pd.DatetimeIndex(real.bar_ts)
        self._kl_sigma = np.asarray(real.sigma, float)
        return self

    # -- lazy load so import of the package never pulls torch --------------

    _LOADERS = ("kronos", "chronos", "timesfm")

    def _ensure(self):
        if self._pipe is not None:
            return
        if self.backend == "auto":
            order = list(self._LOADERS)
        else:  # requested backend first, the rest as fallback
            order = [self.backend] + [b for b in self._LOADERS
                                      if b != self.backend]
        errors = []
        for b in order:
            try:
                getattr(self, f"_load_{b}")()
                self._active = b
                self.name = f"{b}"
                return
            except Exception as e:  # keep trying the next backend
                errors.append(f"{b}: {e}")
        raise SkipModel("no foundation backend available -> " + " | ".join(errors))

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _kronos_repo_path(self):
        """Locate the cloned Kronos repo: explicit param, env var, or the
        conventional vendor/ dir next to this package's parent."""
        import os
        from pathlib import Path
        cands = [p for p in (self.kronos_repo, os.environ.get("KRONOS_REPO", ""))
                 if p]
        cands.append(str(Path(__file__).resolve().parents[2] / "vendor" / "Kronos"))
        for c in cands:
            if (Path(c) / "model" / "kronos.py").exists():
                return str(Path(c))
        raise FileNotFoundError(
            f"Kronos repo not found (tried {cands}); "
            "clone https://github.com/shiyu-coder/Kronos into vendor/Kronos")

    def _load_kronos(self):
        import sys
        repo = self._kronos_repo_path()
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
        tok = KronosTokenizer.from_pretrained(self.kronos_tokenizer)
        mdl = Kronos.from_pretrained(self.kronos_model)
        dev = self._resolve_device()
        self._pipe = KronosPredictor(mdl, tok, device=dev,
                                     max_context=self.context_length)

    def _load_chronos(self):
        from chronos import BaseChronosPipeline  # type: ignore
        import torch
        dev = self._resolve_device()
        self._pipe = BaseChronosPipeline.from_pretrained(
            self.chronos_model,
            device_map=dev,
            torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        )

    def _load_timesfm(self):
        import timesfm  # type: ignore
        self._pipe = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=self._resolve_device(),
                context_len=self.context_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )

    def fit(self, train: np.ndarray) -> "FoundationForecaster":
        # zero-shot: nothing to fit. We still resolve the backend here so a missing
        # dependency surfaces as SkipModel before the backtest loop starts.
        self._ensure()
        return self

    def predict(self, context: np.ndarray, horizon: int):
        self._ensure()
        n_full = len(context)   # position in the series BEFORE truncation --
                                # indexes the aligned klines/sigma when attached
        ctx = np.asarray(context[-self.context_length:], dtype=float)
        if self._active == "kronos":
            qmat = self._predict_kronos(ctx, horizon, n_full)  # (Q, H)
        elif self._active == "chronos":
            qmat = self._predict_chronos(ctx, horizon)   # (Q, H)
        else:
            qmat = self._predict_timesfm(ctx, horizon)
        levels = np.array(QUANTILE_LEVELS)
        q = {float(l): qmat[i] for i, l in enumerate(levels)}
        mean = q[0.50].copy()
        return self._result(mean, q)

    def _predict_kronos(self, ctx: np.ndarray, horizon: int,
                        n_full: int = 0) -> np.ndarray:
        """Kronos speaks K-lines, our ladder speaks returns.

        With real candles attached (attach_klines): feed the true OHLCV bars up
        to the origin, then convert the sampled PRICE paths back into the
        ladder's SCALED-return space by dividing by the trailing sigma that will
        scale the target return (known at the origin -- zero lookahead).

        Without them (synthetic / plain series): reconstruct a price path from
        the returns context (level arbitrary -- Kronos normalizes per series)
        and feed degenerate O=H=L=C candles.

        Either way: `num_paths` independent futures in one batched pass, return
        quantiles read off the paths."""
        levels = list(QUANTILE_LEVELS)
        n = self.num_paths

        if self._klines is not None and n_full > 0:
            rows = self._klines.iloc[:n_full].tail(self.context_length)
            df = rows[["open", "high", "low", "close", "volume"]]
            x_ts = pd.Series(self._kl_ts[rows.index])
            step_dt = (x_ts.diff().median() if len(x_ts) > 1
                       else pd.Timedelta("1min"))
            y_ts = pd.Series([x_ts.iloc[-1] + step_dt * (h + 1)
                              for h in range(horizon)])
            last_close = float(rows["close"].iloc[-1])
            # scaler for the target return (trailing -> known at the origin);
            # multi-step keeps the origin's scaler (sigma moves slowly vs H)
            sigma = float(self._kl_sigma[min(n_full, len(self._kl_sigma) - 1)])
        else:
            close = 100.0 * np.exp(np.cumsum(ctx))
            df = pd.DataFrame({"open": close, "high": close,
                               "low": close, "close": close})
            # fabricated business-day stamps: absolute dates are fictional, but
            # anchoring the START keeps stamps advancing across origins.
            ts = pd.Series(pd.bdate_range("2015-01-02",
                                          periods=len(close) + horizon))
            x_ts, y_ts = ts.iloc[:len(close)], ts.iloc[len(close):]
            last_close = float(close[-1])
            sigma = 1.0                     # series already in model units

        outs = self._pipe.predict_batch(
            [df] * n, [x_ts] * n, [y_ts] * n, pred_len=horizon,
            T=1.0, top_p=0.9, sample_count=1, verbose=False)
        paths = np.stack([o["close"].to_numpy(dtype=float) for o in outs])
        paths = np.maximum(paths, 1e-9)              # guard sampled non-positives
        prev = np.concatenate([np.full((n, 1), last_close), paths[:, :-1]],
                              axis=1)
        rets = np.log(paths / prev) / max(sigma, 1e-12)   # (n, H) scaled paths
        return np.quantile(rets, levels, axis=0)     # (Q, H)

    def _predict_chronos(self, ctx: np.ndarray, horizon: int) -> np.ndarray:
        import torch
        levels = list(QUANTILE_LEVELS)
        t = torch.tensor(ctx, dtype=torch.float32)
        # Chronos-Bolt exposes predict_quantiles directly.
        if hasattr(self._pipe, "predict_quantiles"):
            q, _ = _call_with_series(
                self._pipe.predict_quantiles, t,
                prediction_length=horizon, quantile_levels=levels)
            arr = np.asarray(q[0].cpu().numpy()).T   # (H, Q) -> (Q, H)
            return arr
        # classic Chronos: sample and quantize
        samples = _call_with_series(self._pipe.predict, t,
                                    prediction_length=horizon,
                                    num_samples=self.num_samples)
        s = np.asarray(samples[0].cpu().numpy())     # (num_samples, H)
        return np.quantile(s, levels, axis=0)

    def _predict_timesfm(self, ctx: np.ndarray, horizon: int) -> np.ndarray:
        # TimesFM returns point + experimental quantile heads. We request quantiles
        # if available, else widen the point forecast with the residual scale.
        point, quantile = self._pipe.forecast([ctx], freq=[0])
        point = np.asarray(point)[0][:horizon]
        levels = np.array(QUANTILE_LEVELS)
        q = np.asarray(quantile)
        if q.ndim == 3 and q.shape[-1] >= len(levels):
            # (batch, H, Q) with TimesFM's own decile grid ~ our levels
            grid = q[0][:horizon].T[:len(levels)]
            return grid
        # fallback: Gaussian around point using trailing vol of the context
        sigma = float(np.std(np.diff(ctx))) or 1e-6
        from scipy.stats import norm
        return np.stack([point + sigma * norm.ppf(l) for l in levels], axis=0)
