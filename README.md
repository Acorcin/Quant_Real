# Quant_Real

Intraday CME futures trading system in two layers, bridged:

```
┌─────────────────────────────────────────────────────────────────┐
│  forecasting/ — the measurement instrument                      │
│  Databento trades → volume bars → model ladder                  │
│  (naive → AR → GARCH → Kronos foundation model)                 │
│  → structure_type verdict + per-origin features + veto gate     │
└──────────────────────────┬──────────────────────────────────────┘
                           │  *_features.parquet
                           │  kronos_p50_scaled · kronos_uncertainty
                           │  sigma_256 · structure_type
┌──────────────────────────▼──────────────────────────────────────┐
│  engine/ — the decision layer (López de Prado meta-labeling)    │
│  Tier-1 signals (incl. kronos_directional_signal)               │
│  → XGBoost meta-model (CPCV-validated, Bailey PSR)              │
│  → Kronos structural veto (efficient market ⇒ forced flat)      │
│  → 3-tier probability sizing → CME friction backtest            │
└─────────────────────────────────────────────────────────────────┘
```

**Core stance:** the pipeline is a *measurement instrument* — where a model can
predict, structure exists; where it can't, the market is efficient (a finding,
not a failure) and the system refuses to trade. The engine's meta-model only
sizes positions inside regimes the instrument certifies as predictable.

## Layout

| Path | What |
|------|------|
| `forecasting/` | Dual-plane forecasting pipeline: statistical characterization + point-in-time veto gate. See [forecasting/README.md](forecasting/README.md). |
| `engine/` | Quant EOD engine: XGBoost meta-labeling, Purged CPCV, PSR, tier-1/2 signals, CME futures backtest loop. See [engine/README.md](engine/README.md). |
| `vendor/` | (git-ignored) clone of [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) — rung-3 foundation backend. |
| `data/` | (git-ignored) Databento-licensed market data + derived feature parquets. Bring your own. |

## Quick start

```bash
pip install -r forecasting/requirements.txt
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos

# validate the instrument on synthetic oracles (known ground truth)
python -m forecasting.run --synthetic garch --no-foundation
python -m pytest forecasting/tests -q

# real data: Databento trades parquet → volume bars → full ladder + gate,
# emitting engine features + absolute price levels for manual trading
python -m forecasting.run --real-data data/M6EH6_trades.parquet \
    --instrument M6EH6 --bar-size 250 --step 8 --context-length 256 --num-paths 24
```

The engine has its own Postgres-backed daily loop and backtest — see
[engine/SETUP.md](engine/SETUP.md).

## The bridge contract

`forecasting` emits one row per walk-forward origin; `engine` consumes it:

| features parquet column | engine feature |
|---|---|
| `kronos_p50_scaled` | `kronos_p50_scaled` (tier-1 direction vote) |
| `kronos_spread_scaled` | `kronos_uncertainty` (inverse conviction) |
| `sigma_256` | `kronos_context_vol` |
| `structure_type` | `kronos_regime_encoded` (efficient=0 → **structural veto: forced flat**) |

Validated so far: M6E front-month (Dec 2025–Mar 2026, 6,581 volume bars)
measured **efficient** — no rung beats naive, gate vetoes 98% and improves
reference Sharpe by +0.39. The system correctly refuses to trade noise.
