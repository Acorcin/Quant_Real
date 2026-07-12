# Quant EOD Engine — Technical API Documentation

> **Auto-generated from codebase scan** | **Version:** v1.0.0 | **Date:** 2026-04-03
> Covers **18 modules**, **4 classes**, **65+ functions**

---

## Table of Contents

1. [config](#1-config)
2. [models](#2-models)
3. [fetchers](#3-fetchers)
4. [features](#4-features)
5. [signals](#5-signals)
6. [utils](#6-utils)
7. [scripts (Entry Points)](#7-scripts-entry-points)
8. [Cross-Reference Index](#8-cross-reference-index)

---

## 1. config

### 1.1 `config/settings.py`

[settings.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/config/settings.py)

Module-level configuration loaded from environment variables via `.env`.

#### Constants

| Name | Type | Default | Source | Description |
|------|:----:|---------|:------:|-------------|
| `OANDA_API_TOKEN` | `str` | `""` | `$OANDA_API_TOKEN` | OANDA V20 API bearer token |
| `OANDA_ACCOUNT_ID` | `str` | `""` | `$OANDA_ACCOUNT_ID` | OANDA account identifier |
| `OANDA_BASE_URL` | `str` | `"https://api-fxpractice.oanda.com"` | `$OANDA_BASE_URL` | OANDA API base URL (practice or live) |
| `FRED_API_KEY` | `str` | `""` | `$FRED_API_KEY` | FRED API key for yield data |
| `PERPLEXITY_API_KEY` | `str` | `""` | `$PERPLEXITY_API_KEY` | Perplexity Sonar API key |
| `PERPLEXITY_MODEL` | `str` | `"sonar-pro"` | `$PERPLEXITY_MODEL` | Perplexity model variant |
| `PERPLEXITY_BASE_URL` | `str` | `"https://api.perplexity.ai"` | hardcoded | Perplexity API endpoint |
| `DISCORD_WEBHOOK_URL` | `str` | `""` | `$DISCORD_WEBHOOK_URL` | Discord webhook URL for signal delivery |
| `DB_HOST` | `str` | `"localhost"` | `$DB_HOST` | PostgreSQL host |
| `DB_PORT` | `int` | `5432` | `$DB_PORT` | PostgreSQL port |
| `DB_NAME` | `str` | `"quant_eod"` | `$DB_NAME` | PostgreSQL database name |
| `DB_USER` | `str` | `"postgres"` | `$DB_USER` | PostgreSQL user |
| `DB_PASSWORD` | `str` | `"postgres"` | `$DB_PASSWORD` | PostgreSQL password |
| `DATABASE_URL` | `str` | constructed | derived | Full `postgresql://` connection string |
| `INSTRUMENTS` | `list[str]` | `["EUR_USD", "GBP_USD", "USD_JPY"]` | hardcoded | Trading instruments for data collection |
| `PRIMARY_INSTRUMENT` | `str` | `"EUR_USD"` | hardcoded | Primary instrument for predictions |
| `FRED_US_2Y_SERIES` | `str` | `"DGS2"` | hardcoded | FRED series ID for US 2-Year Treasury |
| `FRED_DE_2Y_SERIES` | `str` | `"IRLTLT01DEM156N"` | hardcoded | FRED series ID for German long-term rate proxy |
| `LOG_DIR` | `str` | `"../logs"` | `$LOG_DIR` | Log file directory |
| `LOG_LEVEL` | `str` | `"INFO"` | `$LOG_LEVEL` | Logging level |
| `SENTIMENT_EXTREME_HIGH` | `float` | `0.72` | `$SENTIMENT_EXTREME_HIGH` | Upper sentiment fade threshold |
| `SENTIMENT_EXTREME_LOW` | `float` | `0.28` | `$SENTIMENT_EXTREME_LOW` | Lower sentiment fade threshold |

---

## 2. models

### 2.1 `models/database.py`

[database.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/database.py)

PostgreSQL database utilities via `psycopg2`.

---

#### `get_connection()`

```python
def get_connection() -> psycopg2.connection
```

| | |
|-|-|
| **Returns** | `psycopg2.connection` — New database connection using `DATABASE_URL` |
| **Description** | Creates a new psycopg2 connection. Caller is responsible for closing. |

---

#### `execute(query, params)`

```python
def execute(query: str, params: tuple = None) -> None
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `query` | `str` | SQL query string (INSERT/UPDATE/DELETE) |
| `params` | `tuple \| None` | Query parameters for `%s` placeholders |

| | |
|-|-|
| **Returns** | `None` |
| **Description** | Executes a write query with auto-commit. Rolls back and re-raises on error. Always closes connection. |

---

#### `execute_returning(query, params)`

```python
def execute_returning(query: str, params: tuple = None) -> tuple | None
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `query` | `str` | SQL query with RETURNING clause |
| `params` | `tuple \| None` | Query parameters |

| | |
|-|-|
| **Returns** | `tuple \| None` — First row from `fetchone()` |
| **Description** | Executes a query and returns the result of `fetchone()`. Used for INSERT ... RETURNING patterns. |

---

#### `fetch_all(query, params)`

```python
def fetch_all(query: str, params: tuple = None) -> list[dict]
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `query` | `str` | SQL SELECT query |
| `params` | `tuple \| None` | Query parameters |

| | |
|-|-|
| **Returns** | `list[dict]` — All rows as list of dictionaries (via `RealDictCursor`) |
| **Description** | Fetches all matching rows. Each row is a `dict` keyed by column name. |

---

#### `fetch_one(query, params)`

```python
def fetch_one(query: str, params: tuple = None) -> dict | None
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `query` | `str` | SQL SELECT query |
| `params` | `tuple \| None` | Query parameters |

| | |
|-|-|
| **Returns** | `dict \| None` — Single row as dict, or `None` if no match |
| **Description** | Fetches at most one row using `RealDictCursor`. |

---

#### `init_schema()`

```python
def init_schema() -> None
```

| | |
|-|-|
| **Returns** | `None` |
| **Description** | Reads all `sql/schema*.sql` files in sorted order and executes them against the database. Creates all tables, indexes, and constraints. Idempotent via `IF NOT EXISTS` in SQL files. |

---

### 2.2 `models/meta_model.py`

[meta_model.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py)

XGBoost binary classifier with CPCV validation (meta-labeling framework).

#### Module Constants

| Name | Type | Value | Description |
|------|:----:|-------|-------------|
| `MODEL_DIR` | `str` | `$MODEL_DIR` or `/tmp/modelartifacts` | Buffer directory for persisted model files |
| `FEATURE_COLS` | `list[str]` | 28 feature names | Ordered feature columns expected by the model |

---

#### `class MetaModel`

```python
class MetaModel:
    model: xgb.XGBClassifier | None
    scaler: StandardScaler | None
    model_version: str
    cpcv_results: dict
    shap_importance: list[dict]
```

---

##### `MetaModel.__init__()`

```python
def __init__(self) -> None
```

Initializes all attributes to empty/None defaults. No model is loaded until `predict()` or `train()` is called.

---

##### `MetaModel.train(feature_vectors, labels, ...)`

```python
def train(
    self,
    feature_vectors: list[dict],
    labels: list[int],
    sample_dates: list[date] | None = None,
    instrument: str | None = None,
    allow_synthetic_return_proxy: bool = False,
) -> dict
```

| Parameter | Type | Required | Description |
|-----------|:----:|:--------:|-------------|
| `feature_vectors` | `list[dict]` | ✅ | List of 28-feature dicts (one per day) |
| `labels` | `list[int]` | ✅ | Binary labels: 1=profitable next-day trade, 0=not |
| `sample_dates` | `list[date] \| None` | ⚠️ | Calendar dates for realized return lookup in CPCV |
| `instrument` | `str \| None` | ⚠️ | Instrument code (e.g., `"EUR_USD"`) for bar returns |
| `allow_synthetic_return_proxy` | `bool` | ❌ | If `True`, allow CPCV fallback to fixed ±0.1% returns |

| | |
|-|-|
| **Returns** | `dict` — `{"model_version", "in_sample_train_metrics", "cpcv", "top_features"}` |
| **Raises** | `ValueError` if < 50 samples, or if `sample_dates`/`instrument` missing without proxy flag |
| **Description** | Fits StandardScaler on all data, runs CPCV (15 paths), trains final XGBoost (200 trees, depth=4, η=0.05, L1=0.1, L2=1.0), computes SHAP, saves model to disk, logs run to `model_runs` table. |

---

##### `MetaModel.predict(feature_vector)`

```python
def predict(self, feature_vector: dict) -> dict
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `feature_vector` | `dict` | Single day's 28-feature dict |

| | |
|-|-|
| **Returns** | `dict` — `{"direction": str, "probability": float, "size_multiplier": float, "model_version": str, "top_shap": list}` |
| **Description** | Loads model from disk if needed. Applies scaler, gets probability from `predict_proba`. Applies 3-tier sizing: `<0.55→flat/0.0×`, `0.55–0.70→0.5×`, `≥0.70→1.0×`. Enforces primary signal veto: if `primary_signal_direction == 0`, always returns flat. |

---

##### `MetaModel.store_prediction(run_date, instrument, prediction, regime_state, composite_strength)`

```python
def store_prediction(
    self, run_date: date, instrument: str,
    prediction: dict, regime_state: int,
    composite_strength: float
) -> None
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `run_date` | `date` | Signal generation date |
| `instrument` | `str` | e.g., `"EUR_USD"` |
| `prediction` | `dict` | Output from `predict()` |
| `regime_state` | `int` | Current HMM regime {0,1,2} |
| `composite_strength` | `float` | Composite signal strength [0,1] |

| | |
|-|-|
| **Returns** | `None` |
| **Description** | Upserts prediction into `predictions` table with `prediction_for` set to `next_trading_day(run_date)`. |

---

##### `MetaModel._run_cpcv(X, y, ...)`

```python
def _run_cpcv(
    self, X: np.ndarray, y: np.ndarray,
    n_groups: int = 6, k_test: int = 2,
    purge: int = 5, embargo: int = 2,
    sample_dates: list[date] | None = None,
    instrument: str | None = None,
    allow_synthetic_return_proxy: bool = False,
) -> dict
```

| | |
|-|-|
| **Returns** | `dict` — `{"paths_tested", "sharpe_mean", "sharpe_std", "path_sharpe_t_statistic", "path_sharpe_p_value", "probabilistic_sharpe_ratio", "uses_synthetic_returns", "statistically_significant", "path_sharpes"}` |
| **Description** | Purged Combinatorial Cross-Validation. Generates C(6,2)=15 train/test splits. Each fold has its own StandardScaler. Trains XGBClassifier(100 trees) per fold. Uses realized next-day returns (or synthetic 0.1% proxy). Computes per-path Sharpe ratios, t-test, and PSR. |

---

##### `MetaModel._compute_shap(X)`

```python
def _compute_shap(self, X: np.ndarray) -> list[dict]
```

| | |
|-|-|
| **Returns** | `list[dict]` — `[{"feature": str, "mean_abs_shap": float}, ...]` sorted descending |
| **Description** | Uses SHAP TreeExplainer. Falls back to XGBoost native `feature_importances_` on error. |

---

##### `MetaModel._save_model()` / `MetaModel._load_model()`

```python
def _save_model(self) -> None
def _load_model(self) -> None
```

| | |
|-|-|
| **Description** | Serializes XGBoost model pipeline natively to Postgres `model_artifacts` table as `.BYTEA` blob (via `models.model_store.save_model_to_db`). Utilizes `/tmp/modelartifacts` as a write-through buffer. Load retrieves version="latest", environment="prod" from Postgres and reverses the process. |

---

##### `MetaModel._default_prediction(feature_vector)`

```python
def _default_prediction(self, feature_vector: dict) -> dict
```

| | |
|-|-|
| **Returns** | `dict` — `{"direction": "flat", "probability": 0.50, "size_multiplier": 0.0, "model_version": "no_model", "top_shap": []}` |
| **Description** | Safe default when no model is available. |

---

#### Module-Level Functions

##### `_normalize_date(d)`

```python
def _normalize_date(d) -> date
```

| | |
|-|-|
| **Description** | Converts `str` (ISO format) or `date` to `date`. Identity for `date` inputs. |

---

##### `_next_trading_day_pct_returns(instrument, sample_dates)`

```python
def _next_trading_day_pct_returns(instrument: str, sample_dates: list) -> np.ndarray | None
```

| | |
|-|-|
| **Returns** | `np.ndarray` of close-to-close returns for each sample date → next trading day, or `None` if no bar data. |
| **Description** | Queries `bars` table for daily closes. Computes `(close_next / close_today) - 1.0` for each provided date. NaN for missing dates. |

---

##### `_probabilistic_sharpe_ratio_from_returns(returns)`

```python
def _probabilistic_sharpe_ratio_from_returns(returns: np.ndarray) -> float
```

| | |
|-|-|
| **Returns** | `float` — PSR value in [0, 1] |
| **Description** | Bailey & López de Prado PSR implementation. Adjusts for skewness and excess kurtosis of return series. Returns `Φ(z)` where `z = SR / √(Var(SR))`. |

---

### 2.3 `models/hmm_regime.py`

[hmm_regime.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py)

Hidden Markov Model for market regime detection.

#### `class RegimeDetector`

```python
class RegimeDetector:
    model: GaussianHMM | None
    state_map: dict
    version: str
```

---

##### `RegimeDetector.fit(instrument)`

```python
def fit(self, instrument: str) -> str
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `instrument` | `str` | e.g., `"EUR_USD"` |

| | |
|-|-|
| **Returns** | `str` — Model version string `"hmm_v1_YYYY-MM-DD"` |
| **Description** | Loads 504 daily bars from DB. Computes `log_return` and `vol_5d` (5-day rolling std). Fits 3-state `GaussianHMM` (diagonal covariance, n_iter=200, tol=1e-4). Sorts states by mean vol_5d ascending → `{0: low_vol, 1: choppy, 2: crash}`. Applies state-flip guard. Persists object via `models.model_store.save_model_to_db()`. |

---

##### `RegimeDetector.predict_regime(instrument)`

```python
def predict_regime(self, instrument: str) -> dict
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `instrument` | `str` | e.g., `"EUR_USD"` |

| | |
|-|-|
| **Returns** | `dict` — `{"state": int, "state_label": str, "confidence": float, "days_in_regime": int, "regime_history": list}` |
| **Description** | Loads model if needed. Queries latest bars, computes observations, predicts state probabilities. Reports the highest-probability state, its confidence, and how many consecutive days the market has been in this regime. |

---

### 2.4 `models/model_store.py`

[model_store.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/model_store.py)

 Handles `joblib` memory buffering to/from Postgres `.BYTEA` seamlessly utilizing `/tmp/modelartifacts/` as a rapid write-through cache mechanism.

#### `save_model_to_db(model_name, version, obj)`
Saves Python object logic to `model_artifacts` Postgres DB.

#### `load_model_from_db(model_name, version)`
Retrieves Python object from `model_artifacts` using local cache if present.

---

## 3. fetchers

### 3.1 `fetchers/oanda_bars.py`

[oanda_bars.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/oanda_bars.py)

---

#### `fetch_candles(instrument, granularity, count)`

```python
def fetch_candles(instrument: str, granularity: str, count: int) -> list[dict]
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `instrument` | `str` | e.g., `"EUR_USD"` |
| `granularity` | `str` | `"D"` (daily) or `"H4"` (4-hour) |
| `count` | `int` | Number of bars to fetch (max 5000) |

| | |
|-|-|
| **Returns** | `list[dict]` — Each dict: `{"instrument", "granularity", "bar_time", "open", "high", "low", "close", "volume", "complete"}` |
| **Description** | Calls OANDA V20 `/v3/instruments/{}/candles` for mid prices. Filters for `complete=True` bars only (skip current/incomplete bar). |

---

#### `store_candles(candles)`

```python
def store_candles(candles: list[dict]) -> None
```

| | |
|-|-|
| **Description** | Upserts candle data into `bars` table. Conflict key: `(instrument, granularity, bar_time)`. |

---

#### `fetch_and_store_all()`

```python
def fetch_and_store_all() -> dict
```

| | |
|-|-|
| **Returns** | `dict` — `{"EUR_USD_D": count, "EUR_USD_H4": count, ...}` |
| **Description** | For each instrument in `INSTRUMENTS`: fetches 210 daily bars and 120 H4 bars, stores both. |

---

### 3.2 `fetchers/fred_yields.py`

[fred_yields.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/fred_yields.py)

---

#### `fetch_yields(lookback_days)`

```python
def fetch_yields(lookback_days: int = 30) -> dict
```

| Parameter | Type | Default | Description |
|-----------|:----:|:-------:|-------------|
| `lookback_days` | `int` | `30` | Days of history to fetch |

| | |
|-|-|
| **Returns** | `dict` — `{"date", "us_2y_yield", "de_2y_yield", "yield_spread_bps", "spread_change_5d_bps", "spread_change_20d_bps", "us_2y_change_1d_bps", "us_2y_change_5d_bps", "us_2y_change_20d_bps", "source"}` |
| **Description** | Fetches US 2Y Treasury (`DGS2`) and German proxy (`IRLTLT01DEM156N`) from FRED. Aligns on common dates, computes spread in bps, 5d/20d deltas. Falls back to US-only changes if DE series unavailable. |

---

#### `store_yields(data)`

```python
def store_yields(data: dict) -> None
```

| | |
|-|-|
| **Description** | Upserts into `yield_data` table. Skips if `data` contains `"error"` key. |

---

#### `fetch_and_store()`

```python
def fetch_and_store() -> dict
```

| | |
|-|-|
| **Returns** | `dict` — Same as `fetch_yields()` output |
| **Description** | Convenience wrapper: fetch + store. |

---

### 3.3 `fetchers/oanda_sentiment.py`

[oanda_sentiment.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/oanda_sentiment.py)

---

#### `fetch_sentiment(instrument)`

```python
def fetch_sentiment(instrument: str) -> dict | None
```

| | |
|-|-|
| **Returns** | `dict` — `{"instrument", "date", "pct_long", "pct_short", "long_short_ratio", "source"}` or `None` |
| **Description** | Queries OANDA ForexLabs legacy endpoint for position ratios. Falls back to `_fallback_sentiment()` (neutral 0.50/0.50) if endpoint deprecated or fails. |

---

#### `_fallback_sentiment(instrument)`

```python
def _fallback_sentiment(instrument: str) -> dict
```

| | |
|-|-|
| **Returns** | `dict` — Neutral sentiment `{pct_long: 0.50, source: "fallback_neutral"}` |

---

#### `store_sentiment(data)` / `fetch_and_store_all()`

```python
def store_sentiment(data: dict) -> None
def fetch_and_store_all() -> dict
```

| | |
|-|-|
| **Description** | Upserts sentiment to DB / loops over all `INSTRUMENTS`. |

---

### 3.4 `fetchers/swap_rates.py`

[swap_rates.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/swap_rates.py)

---

#### `fetch_swap_rate(instrument)`

```python
def fetch_swap_rate(instrument: str) -> dict | None
```

| | |
|-|-|
| **Returns** | `dict` — `{"instrument", "date", "long_rate_annual", "short_rate_annual", "long_swap_pips", "short_swap_pips", "triple_swap_day", "source"}` |
| **Description** | Calls OANDA `/v3/accounts/{}/instruments` to get financing rates. Converts annual rate to daily pips: `rate / 365 × 10000`. Flags Wednesdays as triple-swap days. |

---

#### `store_swap_rate(data)` / `fetch_and_store_all()`

```python
def store_swap_rate(data: dict) -> None
def fetch_and_store_all() -> dict
```

---

### 3.5 `fetchers/calendar.py`

[calendar.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/calendar.py)

---

#### `fetch_calendar_events()`

```python
def fetch_calendar_events() -> dict
```

| | |
|-|-|
| **Returns** | `dict` — `{"today_events": list, "tomorrow_events": list, "is_friday": bool, "source": str}` |
| **Description** | **Stub module.** Attempts `_try_fetch_online_calendar()` (always returns `None`). Returns empty events structure with a warning. Relies on Perplexity AI step for macro event coverage. |

---

#### `_try_fetch_online_calendar(today, tomorrow)`

```python
def _try_fetch_online_calendar(today: date, tomorrow: date) -> dict | None
```

| | |
|-|-|
| **Returns** | `None` — **Always.** Placeholder for future calendar API integration. |

---

#### `store_calendar_events(events)` / `fetch_and_store()`

```python
def store_calendar_events(events: list[dict]) -> None
def fetch_and_store() -> dict
```

---

### 3.6 `fetchers/perplexity_sentiment.py`

[perplexity_sentiment.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/perplexity_sentiment.py)

---

#### Module Constants

| Name | Description |
|------|-------------|
| `DAILY_PROMPT` | System prompt requesting EUR/USD analysis from last 24h of financial news |
| `RESPONSE_SCHEMA` | JSON Schema for structured output: `{macro_sentiment_score, confidence, dominant_driver, key_events, rationale, fed_stance, ecb_stance, risk_sentiment}` |

---

#### `fetch_ai_sentiment()`

```python
def fetch_ai_sentiment() -> dict
```

| | |
|-|-|
| **Returns** | `dict` — `{"date", "macro_sentiment_score": [-1,1], "confidence": [0,1], "dominant_driver", "key_events": list, "rationale", "fed_stance", "ecb_stance", "risk_sentiment", "model_used", "fallback_used", "sources_consulted", "raw_response"}` |
| **Description** | POSTs to Perplexity Sonar API with structured output schema. Parses JSON response. Falls back to neutral (score=0.0, confidence=0.1) on any error or if no API key configured. |

---

#### `_fallback_sentiment(reason)`

```python
def _fallback_sentiment(reason: str) -> dict
```

| | |
|-|-|
| **Returns** | `dict` — Neutral sentiment with `fallback_used=True` and `confidence=0.1` |

---

#### `store_sentiment(data)` / `fetch_and_store()`

```python
def store_sentiment(data: dict) -> None
def fetch_and_store() -> dict
```

| | |
|-|-|
| **Description** | Upserts to `ai_sentiment` table (unique on `date`). Stores `raw_response` as JSON. |

---

### 3.7 `fetchers/discord_notify.py`

[discord_notify.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/discord_notify.py)

---

#### `send_signal(snapshot, status)`

```python
def send_signal(snapshot: dict, status: str) -> bool
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `snapshot` | `dict` | Assembled daily snapshot (from `assemble_daily_snapshot()`) |
| `status` | `str` | One of `"success"`, `"partial"`, `"failed"` |

| | |
|-|-|
| **Returns** | `bool` — `True` if webhook POST succeeded |
| **Description** | Builds a rich Discord embed with prediction, regime, sentiment, yields, key events, and pipeline status. POSTs to `DISCORD_WEBHOOK_URL`. |

---

#### `_build_embed(snapshot, status)`

```python
def _build_embed(snapshot: dict, status: str) -> dict
```

| | |
|-|-|
| **Returns** | `dict` — Discord embed object with color-coded fields |
| **Description** | Formats prediction direction (🟢 LONG / 🔴 SHORT / ⚪ FLAT), regime icon, AI sentiment, central bank stances, yield data, and pipeline status into Discord embed fields. |

---

#### `send_error_alert(error_msg)`

```python
def send_error_alert(error_msg: str) -> bool
```

| | |
|-|-|
| **Returns** | `bool` — `True` if alert sent |
| **Description** | Sends a red-colored error embed with truncated error message (max 1800 chars). Used for pipeline critical failures. |

---

### 3.8 `fetchers/databento_stream_simulator.py`

[databento_stream_simulator.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/databento_stream_simulator.py)

Reads historical CME Globex MBP-1 tick files (`.zst`), parses trade records, and streams them into Redis. Synchronizes simulated dates with daily model parameters (regime, meta-model, and ATR) pushed to Redis keys dynamically.

---

#### `simulate_stream(data_dir, instrument, start_date, end_date, speed_multiplier)`

```python
def simulate_stream(
    data_dir: str,
    instrument: str,
    start_date: date,
    end_date: date,
    speed_multiplier: float
) -> None
```

| Parameter | Type | Default | Description |
|-----------|:----:|:-------:|-------------|
| `data_dir` | `str` | `C:\Users\angel\OneDrive\Apps\Quant EOD\GLBX-20260405-TX6GF64XBR` | Folder with Databento `.zst` files |
| `instrument` | `str` | `"EUR_USD"` | Target instrument code |
| `start_date` | `date` | `2025-04-05` | Start date of simulation |
| `end_date` | `date` | `2026-04-03` | End date of simulation |
| `speed_multiplier`| `float` | `10.0` | Playback speed (1.0=realtime, 0.0=max speed) |

| | |
|-|-|
| **Returns** | `None` |
| **Description** | Scans and sorts files. Loads pre-computed daily models from `walkforward_results.json`. At the beginning of each day, pushes model data to Redis (`{instrument}:metamodel`, `{instrument}:regime`, `{instrument}:daily_atr`). Filters ticks for the front-month contract and streams them chronologically into Redis stream `ticks:{instrument}`. |

---

## 4. features

### 4.1 `features/technical.py`

[technical.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py)

---

#### `compute_all_features(bars, h4_bars)`

```python
def compute_all_features(bars: list[dict], h4_bars: list[dict] | None = None) -> dict
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `bars` | `list[dict]` | Daily OHLCV bars (oldest first), at least 200 bars for MA-200 |
| `h4_bars` | `list[dict] \| None` | Optional 4H bars for multi-timeframe features |

| | |
|-|-|
| **Returns** | `dict` — All technical features for the latest bar |
| **Description** | Computes the full technical feature set from daily bars. Internally calls `_compute_indicators()`, `_compute_candle_patterns()`, `_compute_rolling_vol()`, and optionally `_compute_h4_features()`. |

**Output keys include:**

| Feature | Computation |
|---------|-------------|
| `atr_14` | EMA span=14 (α=2/15) of True Range |
| `rsi_14` | Wilder smoothing (com=13), NaN→50.0 |
| `ma_50_value`, `price_vs_ma50` | SMA-50 and % deviation from close |
| `ma_200_value`, `price_vs_ma200` | SMA-200 and % deviation from close |
| `ema_20` | EMA span=20, adjust=False |
| `body_direction` | +1 (bullish), -1 (bearish), 0 (doji) |
| `body_pct_of_range` | \|close-open\| / (high-low) |
| `upper_wick_pct`, `lower_wick_pct` | Wick lengths as fraction of total range |
| `is_engulfing_bull/bear` | 2-bar engulfing pattern detection |
| `is_pin_bar_bull/bear` | Pin bar: small body + long wick ≥ 60% range |
| `is_inside_bar` | Current bar inside prior bar's range |
| `is_doji` | Body < 10% of range |
| `vol_5d`, `vol_20d` | Rolling std of pct_change (sample, ddof=1) |
| `h4_trend_*` (4 features) | Computed from H4 bars if provided |

---

### 4.2 `features/vector.py`

[vector.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/vector.py)

---

#### `assemble_feature_vector(run_date, instrument, technical, regime, composite, tier2_confirmations)`

```python
def assemble_feature_vector(
    run_date: date, instrument: str,
    technical: dict, regime: dict,
    composite: dict, tier2_confirmations: list[dict],
) -> dict
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `run_date` | `date` | Pipeline execution date |
| `instrument` | `str` | e.g., `"EUR_USD"` |
| `technical` | `dict` | Output from `compute_all_features()` |
| `regime` | `dict` | Output from `RegimeDetector.predict_regime()` |
| `composite` | `dict` | Output from `compute_composite()` |
| `tier2_confirmations` | `list[dict]` | Output from `generate_all_tier2()` |

| | |
|-|-|
| **Returns** | `dict` — 28-feature dict matching `FEATURE_COLS` order |
| **Description** | Assembles all 28 features into a single dict. Pulls macro/sentiment/swap data from DB via helper functions. Encodes categoricals (`fed_stance` → {hawk:-1, neutral:0, dovish:1}). Applies global `None→0.0` coercion via `_safe_float()`. |

---

#### `store_feature_vector(run_date, instrument, vector)`

```python
def store_feature_vector(run_date: date, instrument: str, vector: dict) -> None
```

| | |
|-|-|
| **Description** | Upserts to `feature_vectors` table as JSON. Sanitizes all values to JSON-serializable types (Decimal→float, None→0.0). |

---

#### Helper Functions (Private)

| Function | Signature | Description |
|----------|-----------|-------------|
| `_is_sentiment_extreme` | `(pct_long: float) -> bool` | Returns `True` if `pct_long > 0.72` or `< 0.28` |
| `_get_macro_data` | `(run_date: date) -> dict` | Latest row from `yield_data` where `date <= run_date` |
| `_get_ai_sentiment` | `(run_date: date) -> dict` | Today's row from `ai_sentiment` |
| `_get_sentiment` | `(run_date: date, instrument: str) -> dict` | Latest row from `sentiment` |
| `_get_swap_rates` | `(run_date: date, instrument: str) -> dict` | Latest row from `swap_rates` |

---

## 5. signals

### 5.1 `signals/tier1.py`

[tier1.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py)

#### Module Constants

| Name | Type | Value | Description |
|------|:----:|-------|-------------|
| `YIELD_THRESHOLDS` | `dict` | `{0: 8.0, 1: 15.0, 2: 20.0}` | Regime-adaptive yield spread change thresholds (bps) |
| `SENTIMENT_STRENGTH_SPAN` | `float` | `0.18` | Span for sentiment strength normalization |

---

#### `yield_spread_momentum(run_date, instrument, regime_state)`

```python
def yield_spread_momentum(run_date: date, instrument: str, regime_state: int) -> dict
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `run_date` | `date` | Signal date |
| `instrument` | `str` | Trading instrument |
| `regime_state` | `int` | HMM regime {0,1,2} |

| | |
|-|-|
| **Returns** | `dict` — `{"tier": 1, "detector": "yield_spread_momentum", "direction": str, "strength": float, "detail": str, "metadata": dict}` |
| **Description** | Queries latest yield data. If 5d spread change > regime threshold → SHORT (USD strength). If < -threshold → LONG (EUR strength). Strength = `min(|Δ5d| / (2×τ), 1.0)`. Falls back to US-only 5d change if spread unavailable. |

---

#### `sentiment_extreme_fade(run_date, instrument)`

```python
def sentiment_extreme_fade(run_date: date, instrument: str) -> dict
```

| | |
|-|-|
| **Returns** | Signal dict |
| **Description** | If retail `pct_long > 0.72` → SHORT (fade longs). If `pct_long < 0.28` → LONG (fade shorts). Strength = `min((excess / 0.18), 1.0)`. |

---

#### `ai_macro_sentiment(run_date)`

```python
def ai_macro_sentiment(run_date: date) -> dict
```

| | |
|-|-|
| **Returns** | Signal dict |
| **Description** | If AI score > 0.5 AND confidence > 0.6 → LONG. If < -0.5 AND > 0.6 → SHORT. Always FLAT if fallback was used. Strength = `min(|score| × confidence, 1.0)`. |

---

#### `eod_event_reversal(run_date, instrument, technical)`

```python
def eod_event_reversal(run_date: date, instrument: str, technical: dict) -> dict
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `technical` | `dict` | Must contain `body_direction` |

| | |
|-|-|
| **Returns** | Signal dict |
| **Description** | Queries high-impact calendar events for today. Computes net USD surprise score. If surprise opposes candle body direction → institutional reversal signal. Strength = `min(0.85 + 0.02 × (N-1), 1.0)` where N = count of non-neutral events. Handles conflicting surprises (net=0 → flat). |

---

#### `generate_all_tier1(run_date, instrument, regime_state, technical)`

```python
def generate_all_tier1(run_date: date, instrument: str, regime_state: int, technical: dict) -> list[dict]
```

| | |
|-|-|
| **Returns** | `list[dict]` — 4 signal dicts (one per detector) |
| **Description** | Runs all 4 Tier 1 generators in sequence and returns the combined list. |

---

#### Helper Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `_signal(detector, direction, strength, detail, metadata)` | `dict` | Constructs a standard Tier 1 signal dict |
| `_no_signal(detector, reason)` | `dict` | Flat signal with `{no_data: True}` metadata |

---

### 5.2 `signals/tier2.py`

[tier2.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier2.py)

---

#### `candle_pattern_confirmation(technical, proposed_direction)`

```python
def candle_pattern_confirmation(technical: dict, proposed_direction: str) -> dict
```

| | |
|-|-|
| **Returns** | `dict` — `{"tier": 2, "detector": "candle_pattern", "confirmed": bool, "direction": None, "strength": 1.0|0.0, "detail", "metadata"}` |
| **Description** | Checks if engulfing or pin bar patterns match the proposed direction. Inside bars and dojis noted but don't confirm. |

---

#### `rsi_extreme_confirmation(technical, proposed_direction)`

```python
def rsi_extreme_confirmation(technical: dict, proposed_direction: str) -> dict
```

| | |
|-|-|
| **Description** | RSI < 30 confirms LONG (oversold). RSI > 70 confirms SHORT (overbought). 30–70 = no confirmation. |

---

#### `ma_alignment_confirmation(technical, proposed_direction)`

```python
def ma_alignment_confirmation(technical: dict, proposed_direction: str) -> dict
```

| | |
|-|-|
| **Description** | Price > MA50 AND MA50 > MA200 confirms LONG. Price < MA50 AND MA50 < MA200 confirms SHORT. |

---

#### `multi_timeframe_confirmation(run_date, instrument, proposed_direction)`

```python
def multi_timeframe_confirmation(run_date: date, instrument: str, proposed_direction: str) -> dict
```

| | |
|-|-|
| **Description** | Checks if last 2 completed H4 bars both agree with proposed direction. Requires 2/2 agreement for confirmation. |

---

#### `generate_all_tier2(run_date, instrument, technical, proposed_direction)`

```python
def generate_all_tier2(
    run_date: date, instrument: str,
    technical: dict, proposed_direction: str,
) -> list[dict]
```

| | |
|-|-|
| **Returns** | `list[dict]` — 4 confirmation dicts, or empty list if `proposed_direction == "flat"` |

---

### 5.3 `signals/composite.py`

[composite.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/composite.py)

---

#### `compute_composite(tier1_signals, tier2_confirmations)`

```python
def compute_composite(tier1_signals: list[dict], tier2_confirmations: list[dict]) -> dict
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `tier1_signals` | `list[dict]` | 4 Tier 1 signal dicts from `generate_all_tier1()` |
| `tier2_confirmations` | `list[dict]` | 4 Tier 2 confirmation dicts from `generate_all_tier2()` |

| | |
|-|-|
| **Returns** | `dict` — `{"composite_direction": str, "composite_strength": float, "primary_signal_direction": int, "primary_signal_count": int, "tier1_details": list, "tier2_details": list, "tier2_confirmation_count": int}` |
| **Description** | Performs strength-weighted directional voting across Tier 1 signals. Majority side wins. Base strength = average of winning side's strengths. Applies T2 adjustments: `+0.05` per confirmation, `−0.02` per non-confirmation. Floors at 0.15 (below = flat). Clamps to [0.0, 1.0]. Encodes direction as int: long=1, short=-1, flat=0. |

---

#### `store_signals(run_date, instrument, composite, tier1, tier2)`

```python
def store_signals(
    run_date: date, instrument: str,
    composite: dict, tier1: list[dict], tier2: list[dict],
) -> None
```

| | |
|-|-|
| **Description** | Upserts composite + raw signal data to `signals` table as JSON. |

---

## 6. utils

### 6.1 `utils/trading_calendar.py`

[trading_calendar.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/utils/trading_calendar.py)

#### Module Constants

| Name | Type | Description |
|------|:----:|-------------|
| `MARKET_HOLIDAYS_2026` | `list[date]` | 7 hardcoded US/EU market holidays for 2026 |

---

#### `next_trading_day(run_date)`

```python
def next_trading_day(run_date: date) -> date
```

| Parameter | Type | Description |
|-----------|:----:|-------------|
| `run_date` | `date` | Reference date |

| | |
|-|-|
| **Returns** | `date` — Next valid trading day (skipping weekends + holidays) |
| **Description** | Iterates forward from `run_date + 1 day`, skipping Sat/Sun (`weekday() >= 5`) and dates in `MARKET_HOLIDAYS_2026`. |

---

#### `is_trading_day(d)`

```python
def is_trading_day(d: date) -> bool
```

| | |
|-|-|
| **Returns** | `bool` — `True` if `d` is a weekday not in the holiday list |

---

## 7. Scripts (Entry Points)

### 7.1 `daily_loop.py` — Main Pipeline Orchestrator

[daily_loop.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/daily_loop.py)

**Schedule:** cron at 5:15 PM ET, Mon–Fri

---

#### `main()`

```python
def main() -> None
```

| | |
|-|-|
| **Description** | 13-step pipeline orchestrator. Runs Phase 1 (data collection: 6 steps) and Phase 2 (prediction: 7 steps) in sequence. Each step is try/except wrapped. Assembles snapshot, sends Discord notification, logs pipeline run to DB. |

**Pipeline Steps:**

| Step | Function Called | Phase |
|:----:|----------------|:-----:|
| 0 | `init_schema()` | Setup |
| 1 | `fetch_bars()` | Data |
| 2 | `fetch_yields()` | Data |
| 3 | `fetch_sentiment()` | Data |
| 4 | `fetch_swaps()` | Data |
| 5 | `fetch_calendar()` | Data |
| 6 | `fetch_ai_sentiment()` | Data |
| 7 | `RegimeDetector.predict_regime()` | Predict |
| 8 | `compute_all_features()` + `generate_all_tier1()` + `generate_all_tier2()` | Predict |
| 9 | `compute_composite()` | Predict |
| 10 | `assemble_feature_vector()` + `store_feature_vector()` | Predict |
| 11 | `MetaModel.predict()` + `store_prediction()` | Predict |
| 12 | `send_signal()` | Notify |
| 13 | `log_pipeline_run()` | Log |

---

#### `log_pipeline_run(run_date, started_at, status, steps, errors)`

```python
def log_pipeline_run(
    run_date: date, started_at: datetime,
    status: str, steps: dict, errors: dict,
) -> None
```

| | |
|-|-|
| **Description** | Upserts to `pipeline_runs` table with steps completed and errors as JSON. |

---

#### `assemble_daily_snapshot(...)`

```python
def assemble_daily_snapshot(
    bars_result, yields_result, sentiment_result,
    swaps_result, calendar_result, ai_result,
) -> dict
```

| | |
|-|-|
| **Returns** | `dict` — Combined snapshot with all Phase 1 data, date, and Friday flag |

---

#### `store_snapshot(snapshot)`

```python
def store_snapshot(snapshot: dict) -> None
```

| | |
|-|-|
| **Description** | Upserts snapshot as JSON to `daily_snapshots` table. |

---

### 7.2 `backtest_loop.py` — Backtesting Engine

[backtest_loop.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py)

---

#### `run_backtest(instrument, start_date, end_date, ...)`

```python
def run_backtest(
    instrument: str = "EUR_USD",
    start_date: str | None = None,
    end_date: str | None = None,
    leverage: float = 10.0,
    spread_pips: float = 1.5,
) -> dict
```

| Parameter | Type | Default | Description |
|-----------|:----:|:-------:|-------------|
| `instrument` | `str` | `"EUR_USD"` | Instrument to backtest |
| `start_date` | `str \| None` | `None` | ISO date string, or earliest available |
| `end_date` | `str \| None` | `None` | ISO date string, or latest available |
| `leverage` | `float` | `10.0` | Position leverage multiplier |
| `spread_pips` | `float` | `1.5` | Transaction cost in pips |

| | |
|-|-|
| **Returns** | `dict` — `{"instrument", "period", "metrics": {sharpe, sortino, mdd, total_return, cagr, win_rate, total_trades, avg_trade_return, max_consecutive_loss, profit_factor}, "equity_curve": list, "trade_log": list}` |
| **Description** | Iterates over daily bars. For each day: loads feature vector from DB, runs `MetaModel.predict()`, computes PnL on T+1 Open→Close with leverage and spread. Builds equity curve via multiplicative compounding. |

---

#### `_max_drawdown(equity_curve)`

```python
def _max_drawdown(equity_curve: list[dict]) -> float
```

| | |
|-|-|
| **Returns** | `float` — Maximum peak-to-trough drawdown as a negative percentage |
| **Description** | Tracks running peak of equity. Returns the worst `(trough - peak) / peak` value. |

---

### 7.3 `backfill.py` — Historical Data Bootstrap

[backfill.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backfill.py)

---

#### `backfill_bars(days)`

```python
def backfill_bars(days: int = 504) -> None
```

| Parameter | Type | Default | Description |
|-----------|:----:|:-------:|-------------|
| `days` | `int` | `504` | Number of daily bars (504 ≈ 2 trading years) |

| | |
|-|-|
| **Description** | For each instrument in `INSTRUMENTS`: fetches `days` daily bars and `min(days×6, 5000)` H4 bars from OANDA and stores in DB. |

---

#### `backfill_and_fit_hmm()`

```python
def backfill_and_fit_hmm() -> dict
```

| | |
|-|-|
| **Returns** | `dict` — Current regime prediction from the newly-fit HMM |
| **Description** | Initializes Phase 2 schema, runs `backfill_bars()`, then fits `RegimeDetector` on EUR_USD and returns the current regime state. |

---

#### CLI Usage

```bash
python backfill.py [--days 504] [--hmm]
```

| Flag | Description |
|------|-------------|
| `--days N` | Override lookback period (default 504) |
| `--hmm` | Also fit HMM after backfilling bars |

---

### 7.4 `process_databento.py` — CME Globex Ingestion Tool

[process_databento.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/process_databento.py)

Orchestrates decompression and parsing of CME Globex Micro EUR/USD futures (`M6E`) trade ticks from Databento `.zst` files, aggregating ticks to Daily and 4-Hour bars and batch-upserting them to Postgres.

---

#### `process_directory(data_dir, instrument)`
Parses all `.zst` files in `data_dir`. Automatically identifies the front-month contract for each day (symbol with most trade counts) and aggregates trades into Daily and H4 bars. Daily bars are aligned to 5:00 PM New York time (+7h shift) and H4 bars are aligned on standard 4-hour boundaries starting at 5:00 PM NY time.

---

### 7.5 `generate_historical_features.py` — Training Feature Generator

[generate_historical_features.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/generate_historical_features.py)

Processes daily bars historically to compile the 28-feature vectors for the meta-model training period.

---

#### `generate_features(instrument, start_date, end_date)`
Loops through each trading day, computes daily technical indicators, fits/predicts HMM regimes on a lookahead-free historical data slice, compiles Tier 1 & Tier 2 signals, fetches aligned FRED yields, assembles the feature vectors, and inserts them into Postgres.

---

### 7.6 `walkforward_test.py` — Out-of-Sample Walkforward Backtester

[walkforward_test.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/walkforward_test.py)

Runs the out-of-sample (OOS) walkforward test loop. Simulates intraday executions (Open → Close) using ticks from Databento `.zst` files, tracking trade metrics, maximum excursions, and slippage/spreads.

---

#### `run_walkforward(instrument, start_date, end_date, initial_equity, data_dir, force_train)`
Loads or trains the meta-model on training history up to `start_date`. Loops through candidate OOS dates, predicts direction/sizes, simulates executions using replayed tick files, and outputs performance summaries (Sharpe, drawdown, CAGR, win rate).

---

## 8. Cross-Reference Index

### By Functionality

| Need | Module | Function |
|------|--------|----------|
| **Fetch OHLCV bars** | `fetchers/oanda_bars.py` | `fetch_candles()` |
| **Fetch yield data** | `fetchers/fred_yields.py` | `fetch_yields()` |
| **Fetch sentiment** | `fetchers/oanda_sentiment.py` | `fetch_sentiment()` |
| **Fetch swap rates** | `fetchers/swap_rates.py` | `fetch_swap_rate()` |
| **Fetch calendar** | `fetchers/calendar.py` | `fetch_calendar_events()` |
| **Fetch AI analysis** | `fetchers/perplexity_sentiment.py` | `fetch_ai_sentiment()` |
| **Ingest CME ticks** | `process_databento.py` | `process_directory()` |
| **Compile features** | `generate_historical_features.py` | `generate_features()` |
| **Simulate stream** | `fetchers/databento_stream_simulator.py` | `simulate_stream()` |
| **Compute indicators** | `features/technical.py` | `compute_all_features()` |
| **Build feature vector** | `features/vector.py` | `assemble_feature_vector()` |
| **Detect regime** | `models/hmm_regime.py` | `RegimeDetector.predict_regime()` |
| **Generate signals** | `signals/tier1.py` | `generate_all_tier1()` |
| **Confirm signals** | `signals/tier2.py` | `generate_all_tier2()` |
| **Score composite** | `signals/composite.py` | `compute_composite()` |
| **Predict probability** | `models/meta_model.py` | `MetaModel.predict()` |
| **Train model** | `models/meta_model.py` | `MetaModel.train_from_db()` |
| **Run backtest** | `backtest_loop.py` | `run_backtest()` |
| **Run walkforward** | `walkforward_test.py` | `run_walkforward()` |
| **Send notification** | `fetchers/discord_notify.py` | `send_signal()` |
| **Run full pipeline** | `daily_loop.py` | `main()` |
| **Bootstrap data** | `backfill.py` | `backfill_and_fit_hmm()` |

### By Database Table

| Table | Writer Module(s) | Reader Module(s) |
|-------|-------------------|-------------------|
| `bars` | `oanda_bars.py`, `backfill.py`, `process_databento.py` | `technical.py`, `hmm_regime.py`, `tier2.py`, `meta_model.py` |
| `yield_data` | `fred_yields.py` | `vector.py`, `tier1.py` |
| `sentiment` | `oanda_sentiment.py` | `vector.py`, `tier1.py` |
| `swap_rates` | `swap_rates.py` | `vector.py` |
| `calendar_events` | `calendar.py` | `tier1.py` |
| `ai_sentiment` | `perplexity_sentiment.py` | `vector.py`, `tier1.py` |
| `feature_vectors` | `vector.py`, `generate_historical_features.py` | `backtest_loop.py`, `meta_model.py` |
| `predictions` | `meta_model.py` | — |
| `signals` | `composite.py` | — |
| `daily_snapshots` | `daily_loop.py` | — |
| `pipeline_runs` | `daily_loop.py` | — |
| `model_runs` | `meta_model.py` | — |
| `model_artifacts` | `model_store.py` | `health.py`, `meta_model.py`, `hmm_regime.py` |
| `live_trades` | `order_manager.py` | `risk.py` |
