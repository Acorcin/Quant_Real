# Technical Systems & Architecture Audit Report
**Date:** May 29, 2026  
**Auditor:** Antigravity (Senior Systems Architect)  
**Target:** Quant EOD Engine — Data Pipelines, Persistence & Infrastructure  
**Status:** Completed  

---

## Executive Summary

This systems architecture audit evaluates the FX trading engine (`quant-eod-engine`) for architectural flaws, data pipeline integrity issues, transaction isolation failures, and container infrastructure risks. 

**Overall Assessment:** The system implements a sophisticated multi-stage pipeline, including a newly added 8-layer closed-loop architecture. However, several critical architectural risks exist in database connection management, pipeline failure isolation, Redis stale-safety, and container configuration. Left unaddressed, these issues can lead to connection exhaustion under load, silent trading on stale daily predictions, database schema corruption or missing tables on deployment, and bypassed local caches.

A summary table of vulnerabilities, rated by severity, is provided at the end of this report.

---

## 1. Connection Management

### Findings & Analysis
The database helper functions in [database.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/database.py) successfully implement a `try...finally` block pattern (e.g., [database.py:L17-L29](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/database.py#L17-L29)) which guarantees that connections are closed when queries execute or exceptions are raised. 

However, the architecture has two major issues:
1. **TCP Connection Exhaustion / High Overhead:** Every database helper function and script (e.g., [oanda_bars.py:L72](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/fetchers/oanda_bars.py#L72), [swap_rates.py:L79](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/fetchers/swap_rates.py#L79), [daily_loop.py:L73](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L73)) calls `get_connection()` directly to open a fresh TCP/IP socket connection, executes a single statement (or a small set of queries), and immediately closes the connection. In high-frequency or database-intensive scripts (such as the live trader tick-processing loop in [live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py)), this causes high latency, excessive CPU usage on the database server, and will rapidly exhaust PostgreSQL's socket pool under load (running out of file descriptors or ephemeral ports).
2. **Lack of Connection Pooling & Context Manager:** The project does not utilize a connection pool (like `psycopg2.pool.SimpleConnectionPool` or `ThreadedConnectionPool`). Moreover, the connections are managed manually with `try/finally` blocks instead of the standard Python context manager (`with`) pattern, making the code more verbose and prone to manual closure omissions in future updates.

### Code References
* Opening fresh connections on every query: [database.py:L17-L75](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/database.py#L17-L75)
* Manual connection handling in daily loop: [daily_loop.py:L73](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L73), [daily_loop.py:L122](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L122)

### Severity: Medium (Performance & Resource Risk)
### Recommendations
* **Implement Connection Pooling:** Refactor [database.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/database.py) to initialize a global `SimpleConnectionPool` or `ThreadedConnectionPool` at startup, and modify `get_connection()` to lease connections from the pool.
* **Context Manager Implementation:** Wrap the connection retrieval in a context manager to automate acquisition and release:
  ```python
  from contextlib import contextmanager

  @contextmanager
  def db_connection():
      conn = db_pool.getconn()
      try:
          yield conn
      finally:
          db_pool.putconn(conn)
  ```

---

## 2. Pipeline Failure Modes & Propagation

### Findings & Analysis
In [daily_loop.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py), each phase of the pipeline is isolated in its own `try/except Exception` block. While this prevents the entire orchestrator from crashing, it introduces a severe risk of **silent failure propagation** where downstream stages process invalid or empty data.

If Step 7 (computing technical indicators, [daily_loop.py:L243-L281](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L243-L281)) fails:
1. `technical_result` remains an empty dictionary `{}`.
2. Step 9 ([daily_loop.py:L305-L329](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L305-L329)) runs and passes `technical_result` (now `{}`) into `generate_all_tier1()`. In [tier1.py:L172](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/tier1.py#L172), `eod_event_reversal` tries to get `body_direction`. Since it's empty, it returns `0` (flat) and does not raise an error.
3. `proposed_dir` is computed. If it is not flat, `generate_all_tier2()` is called with `technical_result = {}`. In [tier2.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/tier2.py), functions like `candle_pattern_confirmation` and `rsi_extreme_confirmation` default to unconfirmed/neutral states (e.g., RSI defaults to `50.0` at [tier2.py:L69](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/tier2.py#L69) and MA alignment fails due to missing values at [tier2.py:L101](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/tier2.py#L101)).
4. Step 10 ([daily_loop.py:L330-L342](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L330-L342)) calls `assemble_feature_vector()` with the empty dict. In [vector.py:L82-L89](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/features/vector.py#L82-L89), `technical.get()` falls back to default values (`atr_14 = 0.0`, `rsi_14 = 50.0`, `price_vs_ma50 = 0.0`, etc.).
5. Step 11 ([daily_loop.py:L343-L361](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L343-L361)) feeds this **garbage/neutral feature vector** into `MetaModel.predict()`.
6. XGBoost accepts these float inputs (`0.0`, `50.0`, etc.) and outputs a probability score. Since the model was trained on real market features, feeding it artificial static/neutral inputs can cause it to output arbitrary, highly confident buy/sell signals (e.g., if the model splits on `atr_14 <= 0.001`).
7. This prediction is stored in the database, pushed to Redis, and executed by [live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py) without raising any warnings.

### Code References
* Silently swallowed Step 7 error: [daily_loop.py:L278-L280](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L278-L280)
* Propagation to Tier 1: [daily_loop.py:L307-L309](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L307-L309)
* Propagation to Vector Assembly: [daily_loop.py:L333-L335](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L333-L335)
* Propagation to Prediction: [daily_loop.py:L347-L352](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L347-L352)

### Severity: High (Trading System Loss Risk)
### Recommendations
* **Early Pipeline Halting:** Establish hard checkpoints in `daily_loop.py`. If critical data generation steps fail (such as fetching daily bars or computing technical indicators), the pipeline must immediately raise a fatal error and halt downstream execution.
* **Feature Validity Checks:** Add schema-like validation for feature vectors before prediction (e.g., raise an exception if `atr_14 == 0.0` or if important keys are missing).

---

## 3. Schema Consistency & Initializations

### Findings & Analysis
A comparative review of the database schema files in [sql/](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql) reveals two issues:
1. **Floating-point vs. Decimal Inconsistency:** 
   * The `live_trades` table in [schema_physics.sql](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_physics.sql) uses `DOUBLE PRECISION` for financial metrics: `entry_price`, `exit_price`, `pnl_pips`, and `pnl_amount` (lines 11-16).
   * The `opportunity_measurements` table in [schema_closed_loop.sql](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_closed_loop.sql) uses `NUMERIC(10, 6)` and `NUMERIC(8, 2)` for prices and returns (lines 30-36).
   * In a financial database, using binary floating-point representation (`DOUBLE PRECISION`) is a design flaw due to precision errors. All pricing, yield, swap, and PnL fields must use fixed-point `NUMERIC` types to prevent floating-point drift.
2. **Missing Schema Mounts in Docker-Compose:**
   * In [docker-compose.yml:L13-L16](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml#L13-L16), only 4 of the 6 schema files are mounted in `/docker-entrypoint-initdb.d/`.
   * Specifically, [schema_closed_loop.sql](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_closed_loop.sql) and [schema_migration_model_blobs.sql](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_migration_model_blobs.sql) are omitted.
   * If a developer starts the system via `docker-compose up`, tables like `model_artifacts`, `events`, `opportunity_measurements`, and `learning_runs` will be missing. Although Python's `init_schema()` attempts to run them on startup, if the Python app crashes before execution, the workers will immediately fail with "table not found" errors.

### Code References
* Double precision columns: [schema_physics.sql:L11-L16](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_physics.sql#L11-L16)
* Omitted mounts: [docker-compose.yml:L11-L16](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml#L11-L16)
* Python schema initialization order: [database.py:L84](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/database.py#L84)

### Severity: High (Database Integrity & Deployment Risk)
### Recommendations
* **Enforce Numeric Types:** Migrate `live_trades` columns to `NUMERIC` (e.g., `NUMERIC(12, 5)` for prices and `NUMERIC(15, 2)` for fiat PnL).
* **Sync Docker Compose Mounts:** Update the `postgres` service volumes in `docker-compose.yml` to include all schema files, ensuring correct alphabetical execution ordering:
  ```yaml
      volumes:
        - pgdata:/var/lib/postgresql/data
        - ./sql/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
        - ./sql/schema_closed_loop.sql:/docker-entrypoint-initdb.d/02-schema-closed-loop.sql
        - ./sql/schema_migration_model_blobs.sql:/docker-entrypoint-initdb.d/03-migration-model-blobs.sql
        - ./sql/schema_migration_yield_spread.sql:/docker-entrypoint-initdb.d/04-migration-yield-spread.sql
        - ./sql/schema_phase2.sql:/docker-entrypoint-initdb.d/05-schema-phase2.sql
        - ./sql/schema_physics.sql:/docker-entrypoint-initdb.d/06-schema-physics.sql
  ```

---

## 4. Redis-Postgres Synchronization

### Findings & Analysis
The communication between `daily_loop.py` and `live_trader.py` is decoupled using Redis. The daily loop writes predictions and regime states to Redis (Step 11b), and the live trader polls these keys every 60 seconds (lines 143-170).

This mechanism is **not stale-safe**:
1. **Silent Fallback to Old Predictions:** If the daily loop fails to run (e.g., due to API rate limits, network outages, or pipeline errors) or crashes before Step 11b, the Redis keys `{instrument}:metamodel` and `{instrument}:regime` are **never updated**.
2. **Missing Metadata Verification:** The JSON strings stored in Redis do not contain a date or timestamp. Even if they did, `load_daily_signals()` in [live_trader.py:L46-L80](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L46-L80) does not perform any date validation. 
3. **Implication:** The live trader will read the stale daily direction and execution size from yesterday (or older) and continuously execute trades based on outdated analysis.

### Code References
* Daily Loop Redis push: [daily_loop.py:L369-L384](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L369-L384)
* Live Trader loading: [live_trader.py:L46-L80](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L46-L80) and [live_trader.py:L145-L146](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L145-L146)

### Severity: Critical (Financial Loss Risk)
### Recommendations
* **Add Generation Timestamp:** Update the JSON payload pushed to Redis to include a `generated_at` timestamp.
* **Enforce Staleness Checks:** In [live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py), check the `generated_at` timestamp. If it is older than 24 hours (or does not correspond to today's date), the trader must immediately halt trading, log a critical error, and alert operators.

---

## 5. Transaction Isolation

### Findings & Analysis
The database inserts for `regimes`, `signals`, `feature_vectors`, `predictions`, `events`, `opportunity_measurements`, `learning_runs`, `daily_snapshots`, and `pipeline_runs` occur in separate database connections and transactions.

1. **Non-Atomic Daily Updates:** Because they are not wrapped in a single database transaction, intermediate database states are immediately visible to concurrent database readers. A query that executes mid-pipeline will read mismatched dates (e.g., today's feature vectors combined with yesterday's predictions).
2. **Orphaned Entries on Failure:** If the pipeline fails or is killed midway through execution, only a subset of the tables will be populated. This corrupts historical datasets used for model backtesting, as walk-forward testing will read incomplete entries.

### Code References
* Separate executions and commits in [daily_loop.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py):
  * `detector.store_regime` (Step 8): Independent transaction.
  * `store_signals` (Step 9): Independent transaction.
  * `store_feature_vector` (Step 10): Independent transaction.
  * `meta.store_prediction` (Step 11): Independent transaction.
  * `extractor.extract_and_store` (Step 11c): Independent transaction.
  * `store_snapshot` (Step 12): Independent transaction.
  * `log_pipeline_run` (Step 13): Independent transaction.

### Severity: High (Data Integrity & Backtest Risk)
### Recommendations
* **Single Connection and Transaction Scope:** Refactor [daily_loop.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py) to manage a single database connection and transaction. Pass the cursor to all storing functions, and call `conn.commit()` only at the very end of the daily loop. If any step fails, call `conn.rollback()` to discard the partial state.

---

## 6. Error Handling

### Findings & Analysis
The daily loop uses wide `try/except Exception` blocks around every stage. This represents a "swallow and continue" approach.
1. **Silent Failures:** Critical data fetching failures in Step 1 (bars) or Step 2 (yields) do not raise fatal errors. Instead, they log an error, assign an error dictionary to `bars_result` or `yields_result`, and proceed.
2. **Cascading Side Effects:** If `bars_result` is an error dict, Step 7 will try to calculate technical indicators on empty data. This produces a second exception, which is also swallowed. Eventually, the pipeline completes with a status of `"partial"` or `"failed"` and sends a Discord alert, but the database features are already corrupted with default neutral values.

### Code References
* Wide swallow of bar errors: [daily_loop.py:L170-L174](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L170-L174)
* Wide swallow of technical features errors: [daily_loop.py:L278-L281](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L278-L281)

### Severity: High (Operational Risk)
### Recommendations
* **Define Critical vs. Optional Steps:** Classify steps. Critical steps (Steps 1, 2, 7, 10, 11) must raise a custom `PipelineFatalError` on failure, halting the orchestrator immediately. Non-critical steps (e.g., Step 6 Perplexity sentiment or Step 13 Discord notification) can be swallowed with a fallback mechanism.

---

## 7. Docker Infrastructure

### Findings & Analysis
There are two major infrastructure bugs in [docker-compose.yml](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml):
1. **Hardcoded pg_isready Healthcheck:**
   * The healthcheck for `postgres` uses `pg_isready -U postgres -d quant_eod` (line 20).
   * However, the environment variables on lines 8-10 allow overriding the database user (`DB_USER`) and database name (`DB_NAME`).
   * If a user overrides these in `.env` (e.g., `DB_NAME=fx_prod`), the healthcheck will fail because the database `quant_eod` does not exist or user `postgres` cannot connect. This will mark the container as unhealthy, preventing `physics-engine` and `live-trader` from starting due to the `condition: service_healthy` check.
2. **Model Persistence Bypassed:**
   * The docker-compose mounts a local volume to `/app/model_artifacts` in the containers (lines 74, 95).
   * However, `MODELDIR` in [settings.py:L63](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/config/settings.py#L63) defaults to `/tmp/modelartifacts`.
   * Neither of the docker-compose services overrides the `MODELDIR` environment variable.
   * As a result, the code writes local joblib cache files to `/tmp/modelartifacts` inside the container filesystem, bypassing the volume mount entirely. If the container restarts, the cache is wiped. Since `physics-engine` and `live-trader` run in separate containers, they cannot share their local caches, forcing the live trader to query the PostgreSQL database on every load.

### Code References
* Hardcoded health check: [docker-compose.yml:L20](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml#L20)
* Bypassed volume mounts: [docker-compose.yml:L74](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml#L74), [docker-compose.yml:L95](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml#L95) and [settings.py:L63](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/config/settings.py#L63)

### Severity: High (Operational & Deployment Risk)
### Recommendations
* **Dynamic Healthcheck:** Update `docker-compose.yml` to use environment variables for pg_isready:
  ```yaml
  test: ["CMD-SHELL", "pg_isready -U $${DB_USER:-postgres} -d $${DB_NAME:-quant_eod}"]
  ```
* **Inject MODELDIR Env Var:** Set `MODELDIR=/app/model_artifacts` under the environment block of all services in `docker-compose.yml`.

---

## 8. Model Store & Serialization

### Findings & Analysis
The serialization mechanism uses `joblib` (in [model_store.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/model_store.py)).
1. **Cross-Platform Compatibility Issues:** Joblib uses Python's `pickle` under the hood. Pickle is highly sensitive to library versions. If scikit-learn or XGBoost versions differ between the training server and the production container, loading will fail with `AttributeError` or `ModuleNotFoundError`.
2. **GPU/CPU Serialization Mismatch:** If the XGBoost model is trained on a GPU-enabled machine (which activates CUDA in `_get_xgb_device()` at [meta_model.py:L78](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L78)) and saved, it will fail to load or run on a CPU-only production container unless the XGBoost load config is forced to CPU.
3. **Unhandled Deserialization Exceptions:** In [model_store.py:L84-L86](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/model_store.py#L84-L86), the database blob retrieval and deserialization are not wrapped in a `try...except` block:
  ```python
  # Deserialize
  buffer = io.BytesIO(raw_bytes)
  return joblib.load(buffer)
  ```
  If the bytea blob in Postgres is corrupted or truncated, `joblib.load()` will throw an unhandled exception, causing the prediction engine or daily loop to crash.
4. **Security Vulnerability:** Joblib/pickle deserialization is vulnerable to arbitrary code execution. Loading untrusted or modified bytea blobs from the database could compromise the system host.

### Code References
* Unwrapped load: [model_store.py:L84-L86](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/model_store.py#L84-L86)
* XGBoost device selection: [meta_model.py:L66-L82](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L66-L82)

### Severity: High (Robustness & Security Risk)
### Recommendations
* **XGBoost Native Persistence:** Replace `joblib` with XGBoost's native serialization (`model.save_model("model.json")` and `model.load_model("model.json")`) which is JSON-based, cross-platform, secure, and independent of Python/library versions.
* **Wrap Deserialization in Try/Except:** Wrap `joblib.load` (or the native load function) in a try-except block, logging a warning and returning `None` if the model load fails.

---

## 9. Variable Scoping & Control Flow

### Findings & Analysis
In [daily_loop.py:L369-L379](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L369-L379), the script checks variable existence using `locals()`:
```python
if "regime_result" in locals() and regime_result:
    r.set(f"{PRIMARY_INSTRUMENT}:regime", json.dumps(regime_result))
```
1. **Redundant Checks:** Variables like `regime_result`, `prediction_result`, and `technical_result` are explicitly initialized at the start of the function ([daily_loop.py:L234-L240](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L234-L240)). They are guaranteed to be in `locals()`, making the check redundant.
2. **Unbound Snapshot Risk:** The `snapshot` variable (checked on line 474: `if 'snapshot' in locals() and snapshot`) is NOT initialized at the top of the function. If an exception occurs in `assemble_daily_snapshot()`, `snapshot` is never bound. The check `if 'snapshot' in locals()` correctly catches this, but relying on `locals()` for control flow is a code smell. It makes code refactoring difficult and introduces a risk of silent failures if variables are renamed.

### Code References
* Variable initializations: [daily_loop.py:L234-L240](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L234-L240)
* Redis push checks: [daily_loop.py:L369-L379](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L369-L379)
* Snapshot check: [daily_loop.py:L474](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L474)

### Severity: Low (Code Smell & Maintainability Risk)
### Recommendations
* **Initialize Variables to None:** Initialize all pipeline variables (including `snapshot`) to `None` at the top of the function. Check their validity using `if snapshot is not None:` instead of searching the `locals()` namespace.

---

## Summary of Vulnerabilities

| # | Vulnerability | Severity | Impact | File & Line Reference | Mitigation |
|---|---|---|---|---|---|
| 1 | **Stale Daily Signals in Redis** | **CRITICAL** | Trader will execute trades based on outdated/stale daily predictions if the daily loop fails. | [live_trader.py:L46-L80](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L46-L80) | Add a timestamp to daily predictions and validate freshness in the live trader. |
| 2 | **Silent Pipeline Failure Propagation** | **HIGH** | Downstream predictors receive garbage inputs (e.g., ATR=0) and output arbitrary trades. | [daily_loop.py:L278-L361](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L278-L361) | Halt the pipeline immediately on critical errors rather than continuing with empty data. |
| 3 | **Omitted Docker Schema Mounts** | **HIGH** | Database containers initialized without closed-loop or model storage schemas. | [docker-compose.yml:L11-L16](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml#L11-L16) | Add schema files to volumes in `docker-compose.yml`. |
| 4 | **Bypassed Model Artifact Volume Mounts** | **HIGH** | Cache files are written to ephemeral dyno storage instead of persistent container volumes. | [settings.py:L63](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/config/settings.py#L63) | Set `MODELDIR=/app/model_artifacts` in `docker-compose.yml`. |
| 5 | **Unsafe Joblib Deserialization** | **HIGH** | Corrupted Postgres data will crash the pipeline; unsecure loading allows arbitrary code execution. | [model_store.py:L84-L86](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/model_store.py#L84-L86) | Wrap deserialization in try-except; migrate to native XGBoost/JSON formatting. |
| 6 | **Non-Atomic DB Transactions** | **HIGH** | Concurrent readers see partial states; database contains orphaned rows on mid-pipeline crashes. | [daily_loop.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py) | Wrap the daily loop database pipeline in a single database transaction. |
| 7 | **Inconsistent Financial Types** | **MEDIUM** | Rounding errors in prices/returns due to binary float representation. | [schema_physics.sql:L11-L16](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_physics.sql#L11-L16) | Migrate `live_trades` to use `NUMERIC` types instead of `DOUBLE PRECISION`. |
| 8 | **Hardcoded Postgres Health Check** | **MEDIUM** | Docker database startup fails to report healthy if environment overrides DB names/users. | [docker-compose.yml:L20](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml#L20) | Use Postgres environment variables inside the compose health check. |
| 9 | **Unpooled TCP Connections** | **MEDIUM** | Network socket exhaustion and latency spikes from daily/live execution loops. | [database.py:L17-L75](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/database.py#L17-L75) | Implement connection pooling and context managers. |
| 10 | **Locals Namespace Control Flow** | **LOW** | Code smells, redundant checks, and potential unbound variable bugs. | [daily_loop.py:L369-L379](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L369-L379) | Initialize all local variables to `None` and check explicitly. |
