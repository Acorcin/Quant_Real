# Robust Probabilistic Forecasting Pipeline

Chronos / TimesFM + a statistical model-ladder, built around a **dual-plane**
design: one shared forecast core feeds (1) a *statistical plane* that
characterizes the data and (2) an *operational plane* that emits a point-in-time
**veto gate** for a downstream strategy lab.

> Core stance: we don't build a forecaster, we build a **self-aware risk gate that
> happens to use forecasts**. Forecasting is used as a *measurement instrument* —
> where a model can predict, structure exists; where it can't, the series is
> efficient (a finding, not a failure).

## Design invariants (enforced in code)

1. **Point-in-time.** Every statistic used at time *t* is computed only from data
   ≤ *t*. Scalers, anomaly thresholds, regime labels, calibration — all trailing.
2. **One code path.** Backtest and live emit `ForecastSignal` from the same
   functions, so backtest results transfer to production.
3. **Model returns, not prices.** Target is log-returns (stationary); price is
   reconstructed only for reporting.
4. **Beat the naive baseline.** Nothing is trusted until it beats a random walk on
   scaled error *and* is non-degenerate. The gate's main job is knowing its own
   failure modes.

## The model ladder — the gaps are the measurement

| Rung | Model | If it *significantly* beats naive, the structure is… |
|-----:|-------|------------------------------------------------------|
| 0 | `naive-rw` | (benchmark floor) |
| 1 | `ar` | **linear** autocorrelation (momentum / mean-reversion) |
| 2 | `garch` | predictable **volatility** (direction may still be random) |
| 3 | `foundation` (Kronos/Chronos/TimesFM) | **nonlinear** structure beyond linear + vol |

`classify_structure` reads the rungs (mean accuracy via **Diebold–Mariano** vs
naive) plus model-free signatures (Ljung–Box on returns vs squared returns, and a
CRPS density improvement) into a single `structure_type ∈ {efficient, linear,
vol_only, nonlinear}`.

## Statistical rigor (metrics.py)

- **MASE** — scaled abs error (<1 beats naive).
- **R²_oos** — Campbell–Thompson out-of-sample R² vs naive.
- **Directional accuracy** + exact binomial test (an edge isn't an edge until it
  clears the noise floor for its sample size).
- **Pinball loss / CRPS** — proper scores for the predictive distribution.
- **PIT + chi-square uniformity** — calibration backbone of the gate.
- **Diebold–Mariano** with Newey–West long-run variance and the
  Harvey–Leybourne–Newbold small-sample correction.

## The veto gate (gate.py) — probes → reasons

| Probe | Veto reason | Hardness |
|-------|-------------|----------|
| degenerate forecast (std ratio) | `DEGENERATE_FORECAST` | hard |
| broken calibration (trailing PIT) | `MISCALIBRATED` | hard |
| unpredictable regime | `UNPREDICTABLE_REGIME` | hard |
| recent structural break | `NONSTATIONARY_BREAK` | hard |
| past predictability decay | `HORIZON_BEYOND_EDGE` | hard |
| data-quality flag | `DATA_QUALITY` | hard |
| model disagreement | `LOW_CONVICTION` | soft → `GO_REDUCED` |
| weak directional edge | `WEAK_DIRECTION` | soft |
| elevated vol forecast | `ELEVATED_VOL` | soft |

`GateState` is always derived from the reasons (`make_signal`), never set
independently — a `__post_init__` check enforces it.

## Quick start

```bash
pip install -r requirements.txt          # core: numpy pandas scipy statsmodels
                                         # optional: arch hmmlearn pyarrow

# run the whole pipeline on a synthetic oracle (known ground truth)
python -m forecasting.run --synthetic garch   --no-foundation
python -m forecasting.run --synthetic ar1      --no-foundation
python -m forecasting.run --synthetic regime   --no-foundation

# on your own data (immutable Parquet source of truth)
python -m forecasting.run --parquet data/AAPL.parquet --instrument AAPL

# tests double as validation that the instrument recovers known structure
python -m pytest forecasting/tests -q
```

### Foundation rung (rung 3)

Backend auto-order is **Kronos → Chronos → TimesFM**; the rung self-skips when no
backend's deps are present, so the statistical ladder and gate run fully on CPU
without them.

**Kronos** (primary) is finance-specific — trained on K-lines from 45+ exchanges
([shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)). It isn't on PyPI:

```bash
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos   # next to forecasting/
pip install torch einops huggingface_hub safetensors tqdm
```

Weights download from HuggingFace on first use (`NeoQuasar/Kronos-small` +
`Kronos-Tokenizer-base`, ~100 MB). The adapter in `models/foundation.py`
reconstructs a price path from the returns context (degenerate O=H=L=C candles),
draws `--num-paths` independent sampled futures in one batched pass, and reads the
return quantile grid off the paths.

CPU cost knobs (per-origin latency on a 14-thread CPU):

```bash
# quality default: Kronos-small, ctx 512, 32 paths  (~12 s/origin)
python -m forecasting.run --synthetic ar1

# ~4 s/origin: shorter context, fewer paths, thinned origins
python -m forecasting.run --synthetic ar1 --step 2 --context-length 256 --num-paths 24

# ~1.5 s/origin: Kronos-mini
python -m forecasting.run --synthetic ar1 --context-length 256 --num-paths 24 \
    --kronos-model NeoQuasar/Kronos-mini --kronos-tokenizer NeoQuasar/Kronos-Tokenizer-2k

# force a specific backend
python -m forecasting.run --synthetic ar1 --backend chronos
```

## Module map

```
contracts.py     typed ForecastResult / ForecastSignal / MarketCharacterization
data.py          Series + synthetic oracles (known-truth) + parquet/postgres loaders
prep.py          causal cleaning, log-returns, anomaly FLAGS (never deletion), scaler
windows.py       walk-forward splits; forecast window immediately follows origin;
                 embargo purges the training right edge (not the test window)
metrics.py       MASE, R2_oos, directional test, pinball, CRPS, PIT, Diebold-Mariano
models/          the ladder: naive, ar, garch, foundation (Chronos/TimesFM)
characterize.py  stationarity (ADF/KPSS), Hurst, Ljung-Box, entropy, regimes,
                 structural breaks, ladder classifier -> structure_type
backtest.py      walk-forward engine -> forecast ARCHIVE + scorecards + ladder
gate.py          probes -> veto reasons -> ForecastSignal; gate validation (Sharpe
                 uplift, veto precision/recall)
run.py           end-to-end orchestration + CLI
```

## Roadmap to production (phased)

- **Phase 0** data foundation + contracts — *done (synthetic + loaders)*
- **Phase 1** baseline & walk-forward harness — *done*
- **Phase 2** zero-shot probes — *done (Kronos primary, Chronos fallback, live)*
- **Phase 3** statistical plane / characterization — *done*
- **Phase 4** gate assembly — *done*
- **Phase 5** gate validation hooks — *done (wire your strategy lab into
  `validate_gate`)*
- **Phase 6** live serving — Databento stream on an always-on host reusing this
  exact code path; VETO → last-known-state fallback + alert.
```
