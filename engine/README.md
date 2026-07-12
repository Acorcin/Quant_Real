# Quant EOD Engine

End-of-Day (T+1) Forex Prediction Engine — EUR/USD primary.

## Architecture

Macro-driven signal system anchored at 5:00 PM EST Forex rollover. Runs a single daily data collection pipeline via cron, assembles a feature snapshot, and delivers signals via Discord.

**Philosophy:** Competing on milliseconds is a losing battle for retail. Competing on macro analysis, regime modeling, and overnight swing trading is where true alpha exists.

## Phase 1: Data Collection Layer

Daily pipeline collects 6 data sources → stores in PostgreSQL → sends Discord signal:

| Fetcher | Source | Data |
|---|---|---|
| OANDA Bars | OANDA V20 API | Daily + 4H candles (EUR/USD, GBP/USD, USD/JPY) |
| FRED Yields | FRED API | US 2Y Treasury yield + spread |
| Sentiment | OANDA ForexLabs | Retail position ratios |
| Swap Rates | OANDA V20 API | Financing/rollover rates |
| Calendar | Placeholder | Economic events (manual for now) |
| AI Sentiment | Perplexity Sonar | LLM-scored macro sentiment with structured JSON |

## Setup

See [SETUP.md](SETUP.md) for full deployment instructions.

```bash
# Quick start with Docker
docker compose up -d postgres
docker compose run --rm quant-engine

# Historical backtest on stored feature vectors
python backtest_loop.py --instrument EUR_USD --start 2025-01-01 --end 2025-12-31
```

## Roadmap

- **Phase 1:** Data Collection Layer ← current
- **Phase 2:** Feature Engineering + HMM Regime Detection
- **Phase 3:** ML Model (meta-labeling, CPCV, deflated Sharpe)
- **Phase 4:** Signal Generation + Risk Management (CVaR, vol-targeting)
- **Phase 5:** Agent Zero Integration + Trust Scoring

## Tech Stack

- Python 3.11
- PostgreSQL 16
- Docker Compose
- OANDA V20 API
- FRED API
- Perplexity Sonar API
- Discord Webhooks
