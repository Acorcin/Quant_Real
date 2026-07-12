# Quant EOD Engine — Operations Runbook

> **Last Updated:** 2026-04-03
> **Engine:** Quant EOD Engine (López de Prado meta-labeling)
> **Primary Instrument:** EUR_USD
> **Cron Schedule:** Daily, Mon–Fri, ≥ 5:15 PM America/New_York

---

## Table of Contents

1. [Quickstart (5-Minute Setup)](#1-quickstart)
2. [Environment Configuration](#2-environment-configuration)
3. [Database Initialization](#3-database-initialization)
4. [Historical Data Backfill](#4-historical-data-backfill)
5. [Daily Pipeline Execution](#5-daily-pipeline-execution)
6. [Backtest Execution](#6-backtest-execution)
7. [Model Training](#7-model-training)
8. [Individual Fetcher Operations](#8-individual-fetcher-operations)
9. [Docker & Live Infrastructure Deployment](#9-docker-deployment)
10. [Cron Scheduling](#10-cron-scheduling)
11. [Log Management](#11-log-management)
12. [Troubleshooting Guide](#12-troubleshooting-guide)
13. [Live Replay Simulation](#13-live-replay-simulation)
14. [File Reference](#14-file-reference)

---

## 1. Quickstart

The fastest path from clone to running prediction:

```bash
# 1. Clone & enter
git clone <repo-url> quant-eod-engine && cd quant-eod-engine

# 2. Configure
cp .env.example .env
# Edit .env — fill in OANDA_API_TOKEN, OANDA_ACCOUNT_ID, FRED_API_KEY (minimum required)

# 3. Start PostgreSQL + Redis via Docker
docker compose up -d postgres redis

# 4. Initialize Database Schema
docker compose run --rm live-trader python -c "from models.database import init_schema; init_schema()"

# 5. Backfill history & fit HMM
docker compose run --rm live-trader python backfill.py --hmm

# 6. Run the daily pipeline
docker compose run --rm live-trader python daily_loop.py

# 7. Run a backtest
docker compose run --rm live-trader python backtest_loop.py --output results.json
```

> [!TIP]
> If running locally (not Docker), replace `docker compose run --rm live-trader` with `python` directly, and ensure PostgreSQL is running on `localhost:5432` and Redis is running on `localhost:6379`.

---

## 2. Environment Configuration

### 2.1 Create `.env` File

```bash
cp .env.example .env
```

### 2.2 Required Variables

| Variable | Source | Required | Notes |
|----------|--------|:--------:|-------|
| `OANDA_API_TOKEN` | [OANDA Developer Portal](https://developer.oanda.com/) | ✅ | V20 API token (practice or live) |
| `OANDA_ACCOUNT_ID` | OANDA account settings | ✅ | Format: `101-001-XXXXXXX-XXX` |
| `OANDA_BASE_URL` | — | ✅ | Practice: `https://api-fxpractice.oanda.com`; Live: `https://api-fxtrade.oanda.com` |
| `FRED_API_KEY` | [FRED API Keys](https://fred.stlouisfed.org/docs/api/api_key.html) | ✅ | Free registration |
| `DB_HOST` | — | ✅ | `localhost` (local) or `postgres` (Docker) |
| `DB_PORT` | — | ✅ | Default `5432` |
| `DB_NAME` | — | ✅ | Default `quant_eod` |
| `DB_USER` | — | ✅ | Default `postgres` |
| `DB_PASSWORD` | — | ✅ | Default `postgres` |

### 2.3 Optional Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PERPLEXITY_API_KEY` | _(empty)_ | Enables AI macro sentiment. Falls back to neutral `{score: 0, confidence: 0.1}` if blank |
| `PERPLEXITY_MODEL` | `sonar-pro` | Perplexity model variant |
| `DISCORD_WEBHOOK_URL` | _(empty)_ | Discord notification delivery. Silently skipped if blank |
| `LOG_DIR` | `./logs` | Log file output directory |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_DIR` | `/tmp/modelartifacts` | Buffer directory for DB-backed model artifacts |
| `SENTIMENT_EXTREME_HIGH` | `0.72` | Retail sentiment fade threshold (upper) |
| `SENTIMENT_EXTREME_LOW` | `0.28` | Retail sentiment fade threshold (lower) |

> [!IMPORTANT]
> **Minimum viable setup requires:** `OANDA_API_TOKEN`, `OANDA_ACCOUNT_ID`, `FRED_API_KEY`, and a running PostgreSQL instance. All other features degrade gracefully.

### 2.4 Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

**Full dependency tree:**

| Package | Version | Purpose |
|---------|---------|---------|
| `python-dotenv` | ≥1.0.0 | `.env` file loading |
| `psycopg2-binary` | ≥2.9.9 | PostgreSQL driver |
| `requests` | ≥2.31.0 | HTTP client (OANDA, Perplexity, Discord) |
| `fredapi` | 0.5.2 | FRED data access |
| `pandas` | ≥2.1.0 | DataFrames, technical indicators |
| `numpy` | ≥1.25.0 | Numerical computation |
| `hmmlearn` | ≥0.3.0 | HMM regime detection |
| `xgboost` | ≥2.0.0 | Meta-model classifier |
| `joblib` | ≥1.3.0 | Model serialization |
| `shap` | ≥0.43.0 | Feature importance |
| `scikit-learn` | ≥1.3.0 | StandardScaler, metrics |
| `scipy` | ≥1.11.0 | Statistical tests (PSR, t-test) |
| `python-json-logger` | ≥2.0.0 | Structured logging |

---

## 3. Database Initialization

### 3.1 Create the PostgreSQL Database

```bash
# If running PostgreSQL locally
createdb quant_eod

# Or via psql
psql -U postgres -c "CREATE DATABASE quant_eod;"
```

### 3.2 Apply Schema Files

The schema is split into three files, applied in order:

```bash
# Option A: Via the Python init_schema() function (recommended — handles all files automatically)
python -c "from models.database import init_schema; init_schema()"

# Option B: Manually via psql
psql -U postgres -d quant_eod -f sql/schema.sql
psql -U postgres -d quant_eod -f sql/schema_migration_yield_spread.sql
psql -U postgres -d quant_eod -f sql/schema_phase2.sql
```

### 3.3 Schema Overview

The `init_schema()` function in [database.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/database.py) globs all `sql/schema*.sql` files sorted alphabetically and executes them in a single transaction.

**Phase 1 tables** ([schema.sql](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/sql/schema.sql)):

| Table | Purpose | Unique Constraint |
|-------|---------|-------------------|
| `bars` | OHLCV candles (Daily + H4) | `(instrument, granularity, bar_time)` |
| `yield_data` | US/DE 2Y yields & spread | `(date, source)` |
| `sentiment` | Retail position ratios | `(instrument, date, source)` |
| `swap_rates` | Overnight financing rates | `(instrument, date, source)` |
| `calendar_events` | High-impact economic events | `(event_name, event_time)` |
| `ai_sentiment` | Perplexity LLM scores | `(date)` |
| `daily_snapshots` | Full daily JSON snapshots | `(date, instrument)` |
| `pipeline_runs` | Pipeline execution log | `(run_date)` |

**Phase 2 tables** ([schema_phase2.sql](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/sql/schema_phase2.sql)):

| Table | Purpose | Unique Constraint |
|-------|---------|-------------------|
| `regimes` | HMM regime classifications | `(date, instrument)` |
| `signals` | Tier 1 + Tier 2 signal outputs | `(date, instrument, detector)` |
| `feature_vectors` | 26-feature vectors for meta-model | `(date, instrument)` |
| `predictions` | Meta-model prediction outputs | `(date, instrument)` |
| `model_runs` | Training run audit log | _(none — append-only)_ |
| `model_artifacts` | BYTEA blob storage of compiled models | `(model_name, version, environment)` |

### 3.4 Verify Schema

```bash
psql -U postgres -d quant_eod -c "\dt"
```

Expected: 13 tables.

---

## 4. Historical Data Backfill

Before the prediction engine can operate, you need historical price data. The backfill script fetches 2 years of OANDA candles and optionally fits the HMM regime model.

### 4.1 Bars Only

```bash
# Default: 504 trading days (~2 years) for all instruments (EUR_USD, GBP_USD, USD_JPY)
python backfill.py

# Custom period
python backfill.py --days 252  # ~1 year
```

**What it does (per instrument):**
1. Fetches `--days` daily bars from OANDA V20 API
2. Fetches `days × 6` H4 bars (capped at 5000)
3. Upserts all bars into the `bars` table

### 4.2 Bars + HMM Fit

```bash
python backfill.py --hmm
```

**Additional steps:**
1. Ensures Phase 2 schema exists
2. Runs the standard bar backfill
3. Fits a 3-state Gaussian HMM on EUR_USD daily returns
4. Saves model to `model_artifacts/hmm_regime.joblib`
5. Prints current regime classification

### 4.3 Expected Output

```
2026-04-03 17:20:01 [INFO] backfill: Starting backfill: 504 daily bars for ['EUR_USD', 'GBP_USD', 'USD_JPY']
2026-04-03 17:20:03 [INFO] oanda_bars: Fetched 504 D bars for EUR_USD
2026-04-03 17:20:03 [INFO] oanda_bars: Stored 504 bars
2026-04-03 17:20:05 [INFO] oanda_bars: Fetched 3024 H4 bars for EUR_USD
...
2026-04-03 17:20:15 [INFO] backfill: Fitting HMM regime detector on backfilled data...
2026-04-03 17:20:16 [INFO] hmm_regime: HMM fitted on 498 samples. State mapping: {0: 0, 2: 1, 1: 2}
2026-04-03 17:20:16 [INFO] hmm_regime: Regime: low_vol (conf=0.923, days=14)
```

---

## 5. Daily Pipeline Execution

The main entry point is [daily_loop.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/daily_loop.py) — a 13-step orchestrator that collects data, generates signals, and outputs predictions.

### 5.1 Run the Pipeline

```bash
python daily_loop.py
```

### 5.2 Pipeline Steps

The pipeline runs in two phases with graceful degradation — individual step failures are caught and logged, and the pipeline continues.

```mermaid
graph TB
    subgraph "Phase 1: Data Collection"
        S0["Step 0: Schema Init"] --> S1["Step 1: OANDA Bars<br/>(Daily + H4, 3 instruments)"]
        S1 --> S2["Step 2: FRED Yields<br/>(US 2Y + DE proxy)"]
        S2 --> S3["Step 3: OANDA Sentiment<br/>(position ratios)"]
        S3 --> S4["Step 4: Swap Rates<br/>(financing costs)"]
        S4 --> S5["Step 5: Economic Calendar<br/>(high-impact events)"]
        S5 --> S6["Step 6: Perplexity AI<br/>(LLM macro sentiment)"]
    end

    subgraph "Phase 2: Prediction Engine"
        S6 --> S7["Step 7: Technical Features<br/>(ATR, RSI, MAs, candles)"]
        S7 --> S8["Step 8: HMM Regime<br/>(3-state classification)"]
        S8 --> S9["Step 9: Signal Generation<br/>(T1 + T2 + composite)"]
        S9 --> S10["Step 10: Feature Vector<br/>(26 features assembled)"]
        S10 --> S11["Step 11: Meta-Model<br/>(XGBoost prediction)"]
    end

    subgraph "Finalize"
        S11 --> S12["Step 12: Daily Snapshot<br/>(JSON assembly + DB store)"]
        S12 --> S13["Step 13: Discord Notify<br/>(webhook delivery)"]
    end

    style S0 fill:#2d5016
    style S13 fill:#16375d
```

### 5.3 Pipeline Status Values

| Status | Condition | Exit Code |
|--------|-----------|:---------:|
| `success` | 0 errors | `0` |
| `partial` | 1–3 errors | `0` |
| `failed` | ≥ 4 errors | `1` |

### 5.4 Step-by-Step Detail

| Step | Module | Data Source | DB Table | Graceful Fallback |
|:----:|--------|-------------|----------|-------------------|
| 0 | `models.database.init_schema()` | local SQL files | all | Continues if schema already exists |
| 1 | `fetchers.oanda_bars` | OANDA V20 `/v3/instruments/{}/candles` | `bars` | Error logged; subsequent steps may lack bars |
| 2 | `fetchers.fred_yields` | FRED API (`DGS2`, `IRLTLT01DEM156N`) | `yield_data` | DE 2Y unavailable → US-only fallback |
| 3 | `fetchers.oanda_sentiment` | OANDA ForexLabs `/labs/v1/historical_position_ratios` | `sentiment` | Deprecated endpoint → neutral 50/50 fallback |
| 4 | `fetchers.swap_rates` | OANDA V20 `/v3/accounts/{}/instruments` | `swap_rates` | Returns `None` on failure |
| 5 | `fetchers.calendar` | _(not yet implemented)_ | `calendar_events` | Always returns empty — Perplexity covers this |
| 6 | `fetchers.perplexity_sentiment` | Perplexity Sonar `/chat/completions` | `ai_sentiment` | No API key → neutral fallback `{score: 0, confidence: 0.1}` |
| 7 | `features.technical` | `bars` table | _(in-memory)_ | Empty dict if < 20 daily bars |
| 8 | `models.hmm_regime` | `bars` table | `regimes` | No model → `high_vol_choppy` default (conf=0.33) |
| 9 | `signals.tier1` + `signals.tier2` + `signals.composite` | DB tables | `signals` | Missing data → flat/no signal |
| 10 | `features.vector` | DB tables + in-memory | `feature_vectors` | `None` values coerced to `0.0` |
| 11 | `models.meta_model` | `model_artifacts/` | `predictions` | No model → flat prediction (prob=0.50) |
| 12 | snapshot assembly | all above | `daily_snapshots` | Stores what's available |
| 13 | `fetchers.discord_notify` | assembled snapshot | _(external)_ | No webhook → silently skipped |

### 5.5 Expected Console Output

```
============================================================
DAILY LOOP STARTED — 2026-04-03 (Friday)
============================================================
─── Step 1: Fetching OANDA bars ───
Fetched 210 D bars for EUR_USD
...
─── Step 8: HMM regime detection ───
Regime: low_vol (conf=0.923, days=14)
─── Step 9: Generating signals ───
Tier 1 signals: [('yield_spread_momentum', 'short', 0.4), ...]
Composite: SHORT (strength=0.452, T1 signals=2, T2 confirms=1)
─── Step 10: Assembling feature vector ───
Feature vector: 26 features assembled
─── Step 11: Meta-model prediction ───
Meta prediction: short (prob=0.683, size=0.5x)
─── Step 12: Assembling daily snapshot ───
Stored daily snapshot for 2026-04-03
─── Step 13: Sending Discord notification ───
Discord signal sent successfully
============================================================
DAILY LOOP COMPLETED — Status: SUCCESS
Steps OK: ['schema_init', 'oanda_bars', 'fred_yields', 'sentiment', ...]
============================================================
```

---

## 6. Backtest Execution

Replay stored feature vectors through the trained meta-model to evaluate historical performance.

### 6.1 Run the Backtest

```bash
# Default: all available data, $10,000 initial equity
python backtest_loop.py

# Custom date range and equity
python backtest_loop.py --instrument EUR_USD --start 2024-06-01 --end 2025-12-31 --equity 25000

# Export results to JSON file
python backtest_loop.py --output backtest_results.json
```

### 6.2 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--instrument` | `EUR_USD` (from `PRIMARY_INSTRUMENT`) | Instrument to backtest |
| `--start` | _(all data)_ | Start date `YYYY-MM-DD` |
| `--end` | _(all data)_ | End date `YYYY-MM-DD` |
| `--equity` | `10000.0` | Initial equity in USD |
| `--output` | _(console)_ | JSON output file path |

### 6.3 Prerequisites

> [!WARNING]
> The backtest requires **both** populated `feature_vectors` and `bars` tables. You must have run the daily pipeline at least ~50 times to generate enough feature vectors for meaningful results.

1. `feature_vectors` table has rows for the target date range
2. `bars` table has daily OHLC data covering T+1 for each feature vector date
3. A trained meta-model exists in `model_artifacts/` (or predictions fall back to flat)

### 6.4 Output Format

**Console output** (no `--output`):
```
Backtest EUR_USD 2024-06-01->2025-12-31 | trades=187 win_rate=54.55% total_return=12.34%
Perf | CAGR=8.12% Sharpe=1.23 Sortino=1.87 MaxDD=-6.45% Calmar=1.26 Exposure=74.20% Turnover/day=0.342
```

**JSON output** (`--output results.json`): See the [Backtest Report](file:///C:/Users/angel/.gemini/antigravity/brain/560110ab-13ef-46a9-a70d-364fac2edbda/backtest_report.md) for the full schema.

---

## 7. Model Training

### 7.1 HMM Regime Model

The HMM is automatically fitted during `backfill.py --hmm` or on first `daily_loop.py` run if no model exists.

**Manual re-training:**
```python
from models.hmm_regime import RegimeDetector

detector = RegimeDetector(n_states=3, lookback_days=504)
version = detector.fit("EUR_USD")
regime = detector.predict_regime("EUR_USD")
print(f"Regime: {regime['state_label']} (conf={regime['confidence']:.3f})")
```

**Artifacts produced:**
- Postgres `model_artifacts` table: stores XGBoost `.json` and joblib dependencies as BYTEA blobs (using `models.model_store.py`).
- `/tmp/modelartifacts/`: Write-through memory buffer generated upon runtime.

### 7.2 XGBoost Meta-Model

Training requires labeled feature vectors (50+ samples minimum).

```python
from models.meta_model import MetaModel
from models.database import fetch_all
from datetime import date

# Load labeled feature vectors from DB
rows = fetch_all("""
    SELECT date, features, label FROM feature_vectors
    WHERE instrument = 'EUR_USD' AND label IS NOT NULL
    ORDER BY date ASC
""")

feature_vectors = [r['features'] for r in rows]
labels = [r['label'] for r in rows]
sample_dates = [r['date'] for r in rows]

model = MetaModel()
results = model.train(
    feature_vectors=feature_vectors,
    labels=labels,
    sample_dates=sample_dates,
    instrument="EUR_USD",
)

print(f"Model: {results['model_version']}")
print(f"CPCV PSR: {results['cpcv'].get('probabilistic_sharpe_ratio')}")
print(f"Top features: {results['top_features'][:5]}")
```

**Artifacts produced:**
- `model_artifacts/meta_model_xgb.json` — XGBoost native format
- `model_artifacts/meta_model.joblib` — scaler, version, CPCV results, SHAP importance

**DB record:** Each training run is logged to the `model_runs` table with CPCV results, SHAP importances, and in-sample metrics.

---

## 8. Individual Fetcher Operations

Each fetcher can be run independently for testing or debugging.

### 8.1 OANDA Bars

```bash
python fetchers/oanda_bars.py
```
Fetches 210 daily + 120 H4 bars for all 3 instruments. Output: `{EUR_USD_D: 210, EUR_USD_H4: 120, ...}`.

### 8.2 FRED Yields

```bash
python fetchers/fred_yields.py
```
Fetches~30 days of US 2Y + DE 2Y yields. Computes spread and 5d/20d momentum. Output: yield dict.

### 8.3 Sentiment

```bash
python fetchers/oanda_sentiment.py
```
Fetches retail positioning for all instruments. Output: `{EUR_USD: {pct_long: 0.68, ...}, ...}`.

### 8.4 Swap Rates

```bash
python fetchers/swap_rates.py
```
Fetches overnight financing rates from instrument details. Output: daily pip costs.

### 8.5 Calendar

```bash
python fetchers/calendar.py
```
Currently returns empty — no calendar API configured. The Perplexity step partially covers this.

### 8.6 Perplexity AI Sentiment

```bash
python fetchers/perplexity_sentiment.py
```
Calls Perplexity Sonar API with structured output schema. Returns macro sentiment score with confidence and central bank stances. Requires `PERPLEXITY_API_KEY`.

---

## 9. Docker & Heroku Deployment

### 9.1 Heroku Eco Dyno Deployment (CI/CD)
The HERO branch introduces a persistent Postgres model cache and a web health endpoint for Heroku Eco dynos. 
Deployment is fully automated through `.github/workflows/deploy.yml`.

1. Merge new commits into `HERO` or `main`.
2. GitHub Actions trigger the workflow: `akhileshns/heroku-deploy@v3.13.15`.
3. Heroku spins up `gunicorn health:app` mapped by the `Procfile`.
4. Ensure the Heroku Scheduler is enabled and scheduled for 5:15 ET `python daily_loop.py`.

### 9.2 Full Stack Docker Infrastructure

The deployment stack runs in four containers:
- **`quant_postgres`**: PostgreSQL database storing bars, yields, sentiment, and trade logs.
- **`quant_redis`**: Redis server providing the message queue and dynamic model state storage.
- **`physics_engine`**: Continuous worker running the Kalman filters for tick smoothing.
- **`live_trader`**: Continuous event loop that reads from Redis and manages dry-run/live execution.

```bash
# Start the database and Redis services
docker compose up -d postgres redis

# Wait for Postgres and Redis to show "healthy"
docker compose ps

# Initialize schema and run historical backfill via the live-trader environment
docker compose run --rm live-trader python -c "from models.database import init_schema; init_schema()"
docker compose run --rm live-trader python backfill.py --hmm

# Start the continuous live/simulated trading stack
docker compose up -d physics-engine live-trader
```

### 9.3 Docker Environment Overrides

The `docker-compose.yml` automatically sets:

| Variable | Docker Value | Reason |
|----------|-------------|--------|
| `DB_HOST` | `postgres` | Docker service name |
| `DB_PORT` | `5432` | Standard |
| `REDIS_HOST` | `redis` | Docker service name for Redis stream |
| `PYTHONPATH` | `/app` | Standard application import path |
| `LOG_DIR` | `/app/logs` | Mounted volume |
| `MODEL_DIR` | `/app/model_artifacts` | Model storage volume shared between containers |

### 9.4 Persistent Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `pgdata` | PostgreSQL data dir | Database persistence across restarts |
| `./logs` | `/app/logs` | Log file access from host |
| `./model_artifacts` | `/app/model_artifacts` | Shared model files between simulator, engine, and live containers |

### 9.5 Rebuild After Code Changes

```bash
docker compose build physics-engine live-trader
docker compose up -d physics-engine live-trader
```

---

## 10. Cron Scheduling

### 10.1 Linux/Mac Cron

```cron
# Run daily at 5:20 PM ET (Mon-Fri)
# Adjust for your timezone — this assumes the server is in ET
20 17 * * 1-5 cd /path/to/quant-eod-engine && /path/to/.venv/bin/python daily_loop.py >> /path/to/logs/cron.log 2>&1
```

### 10.2 Windows Task Scheduler

1. Open Task Scheduler → Create Task
2. **Trigger:** Daily, 5:20 PM, repeat Mon–Fri
3. **Action:** Start a program
   - Program: `C:\path\to\.venv\Scripts\python.exe`
   - Arguments: `daily_loop.py`
   - Start in: `C:\path\to\quant-eod-engine`

### 10.3 Docker Cron

```bash
# Using a simple cron container or host cron
20 17 * * 1-5 cd /path/to/quant-eod-engine && docker compose run --rm live-trader python daily_loop.py
```

> [!IMPORTANT]
> **Timing matters.** Schedule at or after **5:15 PM ET** (Mon–Fri). OANDA marks the daily bar as `complete` after the 5 PM ET fix/rollover. Running too early will cause Step 1 to miss today's completed daily bar.

---

## 11. Log Management

### 11.1 Log Locations

| Log | Path | Format |
|-----|------|--------|
| Daily pipeline log | `logs/daily_YYYY-MM-DD.log` | `%(asctime)s [%(levelname)s] %(name)s: %(message)s` |
| Console output | stdout | Same format as file |
| DB pipeline_runs | `pipeline_runs` table | JSON status, steps, errors |

### 11.2 Checking Pipeline History

```sql
-- Last 10 runs
SELECT run_date, status, steps_completed, errors
FROM pipeline_runs ORDER BY run_date DESC LIMIT 10;

-- Failed runs
SELECT run_date, errors FROM pipeline_runs
WHERE status = 'failed' ORDER BY run_date DESC;

-- Partial runs (some steps failed)
SELECT run_date, errors FROM pipeline_runs
WHERE status = 'partial' ORDER BY run_date DESC;
```

### 11.3 Checking Predictions

```sql
-- Latest predictions
SELECT date, prediction_for, direction, probability, size_multiplier, regime_state
FROM predictions WHERE instrument = 'EUR_USD'
ORDER BY date DESC LIMIT 10;
```

---

## 12. Troubleshooting Guide

This section maps every `try/except` block in the codebase to its diagnostic path.

---

### ERR-01: `psycopg2.OperationalError` — Database Connection Failed

**Raised by:** [`models/database.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/database.py#L12-L14) — `get_connection()`

**Symptoms:**
```
DB fetch_all error executing SELECT... : could not connect to server: Connection refused
```

**Root Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| PostgreSQL not running | `docker compose up -d postgres` or `pg_ctl start` |
| Wrong `DB_HOST` | Docker: set to `postgres`; Local: set to `localhost` |
| Wrong `DB_PORT`, `DB_NAME`, `DB_USER`, or `DB_PASSWORD` | Check `.env` matches your PostgreSQL config |
| Database doesn't exist | `createdb quant_eod` or `psql -c "CREATE DATABASE quant_eod;"` |
| Connection string malformed | Verify `DATABASE_URL` in `config/settings.py` resolves correctly |

**Diagnostic:**
```bash
# Test connection directly
psql "postgresql://postgres:postgres@localhost:5432/quant_eod" -c "SELECT 1;"
```

---

### ERR-02: `Schema init error` — SQL Schema Application Failed

**Raised by:** [`models/database.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/database.py#L78-L101) — `init_schema()`

**Symptoms:**
```
Schema init error: relation "bars" already exists
```

**Root Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Schema already applied | Safe to ignore — all `CREATE TABLE` use `IF NOT EXISTS` |
| SQL syntax error in custom migration | Check `sql/schema*.sql` files for valid PostgreSQL syntax |
| Insufficient privileges | Ensure `DB_USER` has `CREATE TABLE` rights on `DB_NAME` |

> [!NOTE]
> The pipeline is designed to **continue** if `init_schema()` fails, because the schema likely already exists. This error is logged but is non-fatal.

---

### ERR-03: `requests.exceptions.HTTPError` — OANDA API Request Failed

**Raised by:** [`fetchers/oanda_bars.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/oanda_bars.py#L42-L43) — `response.raise_for_status()`

**Symptoms:**
```
OANDA bars failed: 401 Client Error: Unauthorized
Failed to fetch bars for EUR_USD: 403 Forbidden
```

**Root Causes & Fixes:**

| HTTP Code | Cause | Fix |
|-----------|-------|-----|
| `401` | Invalid or expired API token | Regenerate token at [OANDA Developer Portal](https://developer.oanda.com/) |
| `403` | Wrong account type for URL | Practice token → use `api-fxpractice`; Live token → use `api-fxtrade` |
| `404` | Invalid instrument name | Check `INSTRUMENTS` in `settings.py`; use OANDA format (`EUR_USD` not `EURUSD`) |
| `429` | Rate limited | Wait 1 second, try again. OANDA allows ~120 requests/sec |
| Timeout | Network issue | Default timeout is 30s. Check network/firewall |

**Diagnostic:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles?count=5&granularity=D"
```

---

### ERR-04: `FRED API failure` — US 2Y or DE 2Y Yield Fetch Failed

**Raised by:** [`fetchers/fred_yields.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/fred_yields.py#L31-L45)

**Symptoms:**
```
Failed to fetch US 2Y yield: 400 Bad Request
German 2Y series unavailable from FRED: This series does not exist
```

**Root Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Invalid `FRED_API_KEY` | Re-register at [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) |
| US 2Y (`DGS2`) fetch fails | Critical — pipeline returns `{"error": "US 2Y yield data unavailable"}` |
| DE 2Y (`IRLTLT01DEM156N`) unavailable | Non-fatal — spread calculations become partial; yield spread momentum signal falls back to US-only mode |
| FRED API down on weekends/holidays | Expected — FRED data updates Mon-Fri only |

> [!TIP]
> The DE 2Y series `IRLTLT01DEM156N` is a **long-term rate proxy**, not true German 2Y. Consider integrating ECB Statistical Data Warehouse for production accuracy.

---

### ERR-05: `Sentiment fetch failed` — OANDA ForexLabs Deprecated

**Raised by:** [`fetchers/oanda_sentiment.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/oanda_sentiment.py#L71-L73)

**Symptoms:**
```
ForexLabs sentiment endpoint returned status 404 for EUR_USD. This endpoint may have been deprecated.
Sentiment fetch failed for EUR_USD: ConnectionError. Using fallback.
```

**Behavior:** Automatically falls back to neutral `{pct_long: 0.50, pct_short: 0.50}`. This is safe — the sentiment_extreme_fade signal will simply never fire.

**Fix:** Implement an alternative sentiment source (myfxbook, DailyFX SSI, web scraping).

---

### ERR-06: `Perplexity API error` — AI Sentiment Generation Failed

**Raised by:** [`fetchers/perplexity_sentiment.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/perplexity_sentiment.py#L96-L149)

**Symptoms:**
```
No Perplexity API key configured. Using fallback.
Perplexity API HTTP error: 429 Too Many Requests
Perplexity API error: JSONDecodeError
```

**Root Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| No API key | Set `PERPLEXITY_API_KEY` in `.env` (or accept neutral fallback) |
| `429` rate limit | Perplexity free tier has low limits. Upgrade or reduce call frequency |
| Invalid JSON response | Perplexity model didn't return valid structured output — retry |
| `500` server error | Perplexity service issue — fallback will activate |

**Behavior:** All failures produce a neutral fallback: `{score: 0.0, confidence: 0.1, fallback_used: true}`. The `ai_macro_sentiment` Tier-1 signal explicitly checks `fallback_used` and generates **no signal** if true.

---

### ERR-07: `Swap rates failed` — OANDA Instrument Details Error

**Raised by:** [`fetchers/swap_rates.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/swap_rates.py#L72-L74)

**Symptoms:**
```
Failed to fetch swap rates for EUR_USD: 401 Unauthorized
```

**Behavior:** Returns `None`. The feature vector will use `long_swap_pips: 0.0` and `short_swap_pips: 0.0`.

**Fix:** Same as ERR-03 — verify OANDA credentials and account access.

---

### ERR-08: `HMM model unavailable` — Regime Detection Failed

**Raised by:** [`models/hmm_regime.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L139-L193) — `predict_regime()`

**Symptoms:**
```
No HMM model available — returning default regime
No HMM model file at model_artifacts/hmm_regime.joblib
Need at least 60 bars for HMM, got 23
```

**Root Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| No model file | Run `python backfill.py --hmm` |
| < 60 daily bars in DB | Backfill more data: `python backfill.py --days 252` |
| HMM convergence failure | Rare — increase `n_iter` or check data for NaN/Inf |

**Behavior:** Falls back to `{state_id: 1, state_label: "high_vol_choppy", confidence: 0.33}` — the most conservative default.

---

### ERR-09: `No meta-model available` — XGBoost Prediction Fallback

**Raised by:** [`models/meta_model.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L218-L223) — `predict()`

**Symptoms:**
```
No meta-model at model_artifacts/meta_model.joblib
No meta-model available — returning default prediction
```

**Behavior:** Returns `{direction: "flat", probability: 0.50, size_multiplier: 0.0}` — never trades.

**Fix:** Train the model (see §7.2). Requires ≥ 50 labeled `feature_vectors` rows.

---

### ERR-10: `Discord webhook error` — Notification Delivery Failed

**Raised by:** [`fetchers/discord_notify.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/discord_notify.py#L30-L55)

**Symptoms:**
```
No Discord webhook URL configured. Skipping notification.
Discord webhook HTTP error: 404 Not Found
Discord webhook error: ConnectionError
```

**Root Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| No webhook URL | Set `DISCORD_WEBHOOK_URL` in `.env` |
| Webhook deleted/invalid | Create a new webhook in Discord channel settings |
| `429` rate limit | Discord webhooks have 30 msg/min/channel limit |
| Embed too large | Snapshot truncation may be needed for very large payloads |

**Behavior:** Non-fatal. Returns `False`, logged as warning. Pipeline status unaffected.

---

### ERR-11: `No feature_vectors rows found` — Backtest Has No Data

**Raised by:** [`backtest_loop.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L113-L114) — `run_backtest()`

**Symptoms:**
```
No feature_vectors rows found for selection
```

**Fix:** Run the daily pipeline for enough days to build up feature vectors, or check your `--start`/`--end` dates match the available data range.

```sql
-- Check available feature vector date range
SELECT MIN(date), MAX(date), COUNT(*) FROM feature_vectors WHERE instrument = 'EUR_USD';
```

---

### ERR-12: `HMM state-label flip detected` — Regime Labels Changed

**Raised by:** [`models/hmm_regime.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L119-L126)

**Symptoms:**
```
HMM state-label flip detected on re-train. Raw states [0, 2] changed semantic meaning vs previous model.
Old map: {0: 0, 1: 1, 2: 2} | New map: {0: 2, 1: 1, 2: 0}
```

**Impact:** Warning only — the new model is still correct by construction (lowest vol = state 0). However, historical `regimes` table rows retain old labels.

**Action:** Review regime history if signal behavior changed unexpectedly after a re-train.

---

### ERR-13: `Failed to record model_run to DB`

**Raised by:** [`models/meta_model.py`](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L195-L197)

**Symptoms:**
```
Failed to record model_run to DB: relation "model_runs" does not exist
```

**Fix:** Apply Phase 2 schema:
```bash
psql -U postgres -d quant_eod -f sql/schema_phase2.sql
```

**Impact:** Non-fatal — the model is still trained and saved to disk. Only the DB audit record is missing.

---

### General Diagnostic Commands

```bash
# Check database connectivity
python -c "from models.database import get_connection; c=get_connection(); print('OK'); c.close()"

# Check table row counts
psql -U postgres -d quant_eod -c "
  SELECT 'bars' as t, count(*) FROM bars
  UNION ALL SELECT 'feature_vectors', count(*) FROM feature_vectors
  UNION ALL SELECT 'predictions', count(*) FROM predictions
  UNION ALL SELECT 'regimes', count(*) FROM regimes
  UNION ALL SELECT 'pipeline_runs', count(*) FROM pipeline_runs;"

# Check OANDA API access
python -c "from fetchers.oanda_bars import fetch_candles; print(len(fetch_candles('EUR_USD', 'D', 5)))"

# Check FRED API access
python -c "from fetchers.fred_yields import fetch_yields; print(fetch_yields(5))"

# Check model artifacts exist
python -c "import os; [print(f) for f in os.listdir('model_artifacts')]"
```

---

## 13. Live Replay Simulation

The live trading stack is designed as an event-driven system decoupled via Redis streams and run inside Docker containers. To validate this setup locally before executing with real capital, the system provides a high-fidelity **CME Tick Stream Simulator** ([databento_stream_simulator.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/databento_stream_simulator.py)).

### 13.1 Architecture Overview
The simulation spans the following interconnected components:
1. **Host/Simulator**: Runs `fetchers/databento_stream_simulator.py`, which reads Databento MBP-1 tick files (`.csv.zst`) and streams them chronologically into Redis stream `ticks:EUR_USD`. It also loads `walkforward_results.json` and updates daily model state keys (`EUR_USD:regime`, `EUR_USD:metamodel`, `EUR_USD:daily_atr`) in Redis as time advances.
2. **Redis (`quant_redis` container)**: Serves as the central message bus and state store.
3. **Physics Engine (`physics_engine` container)**: Runs `physics/worker.py`, consuming raw ticks from `ticks:EUR_USD` and publishing conditioned ticks (filtered and Kalman-smoothed) into `cond_ticks:EUR_USD`.
4. **Live Trader (`live_trader` container)**: Runs `execution/live_trader.py`, consuming conditioned ticks, executing the live trading loop, evaluating risk limits, and logging to `live_trades`.
5. **Database (`quant_postgres` container)**: Stores historical execution logs in the `live_trades` table.

### 13.2 Running the Simulation Step-by-Step

#### Step 1: Start the Infrastructure
Spin up the database, Redis cache, Kalman worker, and live trader container:
```bash
docker compose up -d postgres redis physics-engine live-trader
```
Ensure all containers are healthy:
```bash
docker compose ps
```

#### Step 2: Prepare Walkforward Predictions
Ensure `walkforward_results.json` exists in the repository root. This is generated by running `python walkforward_test.py` or training your models. The simulator uses this file to update the model parameters dynamically in Redis as time advances.

#### Step 3: Run the Stream Simulator
Run the simulator from your host machine (where your Databento tick data resides):
```bash
python fetchers/databento_stream_simulator.py \
  --data-dir "C:\Users\angel\OneDrive\Apps\Quant EOD\GLBX-20260405-TX6GF64XBR" \
  --speed 10.0 \
  --start 2025-04-05 \
  --end 2026-04-03
```
*Note:* Use `--speed 0.0` for maximum speed (replays all ticks without delay), or set to a factor like `10.0` (10x real-time) to observe logging.

#### Step 4: Monitor Execution
1. **Container Logs**:
   To watch trades and Kalman filtering in real time:
   ```bash
   docker compose logs -f live-trader
   docker compose logs -f physics-engine
   ```
2. **Redis Monitor**:
   Check if ticks are streaming through Redis:
   ```bash
   docker exec -it quant_redis redis-cli xlen ticks:EUR_USD
   ```
3. **Database Queries**:
   Inspect executed orders and PnL by querying PostgreSQL:
   ```bash
   docker exec -it quant_postgres psql -U postgres -d quant_eod -c "SELECT * FROM live_trades ORDER BY entry_time DESC LIMIT 10;"
   ```

### 13.3 Important Caveats & Risk Safeguards
- **Risk Manager Drawdown Breach**: If running at `--speed 0.0`, the system will execute multiple trades in quick succession. If those trades hit the daily drawdown limit of 2% (default `-$200`), the `RiskManager` will breach and halt further trading. 
- **Time-Based DB Queries**: `RiskManager._get_today_realized_pnl` uses `CURRENT_DATE` in its SQL query, which references the host's current time. For simulated back-testing / replay of past years, this logic checks today's date against historical trades. To reset the daily risk limit when re-running simulations on different dates, you must truncate the `live_trades` table:
  ```bash
  docker exec -it quant_postgres psql -U postgres -d quant_eod -c "TRUNCATE TABLE live_trades;"
  ```

---

## 14. File Reference

```
quant-eod-engine/
├── daily_loop.py              # Main entry point — 13-step daily pipeline
├── backtest_loop.py           # Historical backtest simulator
├── backfill.py                # Historical data loader + HMM bootstrap
├── walkforward_test.py        # Walkforward validation and out-of-sample backtester
├── process_databento.py       # Databento tick data aggregator to bar format
├── generate_historical_features.py  # Pre-computes training features from raw data
├── config/
│   └── settings.py            # All configuration & environment variables
├── models/
│   ├── database.py            # PostgreSQL connection & helpers
│   ├── hmm_regime.py          # 3-state Gaussian HMM regime detector
│   └── meta_model.py          # XGBoost meta-labeling model + CPCV
├── features/
│   ├── technical.py           # ATR, RSI, MA, candle body analysis
│   └── vector.py              # 28-feature vector assembly
├── signals/
│   ├── tier1.py               # 4 primary alpha signal generators
│   ├── tier2.py               # 4 confirmation filters
│   └── composite.py           # Direction + strength assembly
├── fetchers/
│   ├── oanda_bars.py          # OANDA V20 candle data
│   ├── fred_yields.py         # FRED yield spread data
│   ├── oanda_sentiment.py     # Retail positioning ratios
│   ├── swap_rates.py          # Overnight financing rates
│   ├── calendar.py            # Economic calendar (stub)
│   ├── perplexity_sentiment.py # AI LLM macro analysis
│   ├── databento_stream_simulator.py # Replays Databento ticks to Redis stream
│   └── discord_notify.py      # Discord webhook notifications
├── physics/
│   └── worker.py              # Kalman filter stream-conditioned tick processor
├── execution/
│   ├── live_trader.py         # Continuously reads cond_ticks, sizing, and entry/exit
│   └── risk_manager.py        # Daily drawdown (2%) and risk boundaries guard
├── utils/
│   └── trading_calendar.py    # Next trading day + holiday calendar
├── sql/
│   ├── schema.sql             # Phase 1 tables (bars, yields, sentiment, etc.)
│   ├── schema_phase2.sql      # Phase 2 tables (regimes, signals, predictions, model_artifacts)
│   ├── schema_migration_yield_spread.sql  # Add spread columns to existing DBs
│   └── schema_physics.sql     # Live execution logs (live_trades table)
├── model_artifacts/           # Serialized models (gitignored)
│   ├── hmm_regime.joblib
│   ├── meta_model.joblib
│   └── meta_model_xgb.json
├── logs/                      # Daily log files (gitignored)
├── Dockerfile                 # Python 3.11-slim container
├── docker-compose.yml         # Postgres + Redis + Physics + Live Trader services
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── .env                       # Your secrets (gitignored)
```
