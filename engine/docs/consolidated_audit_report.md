# Consolidated Multi-Agent Audit Report — Quant EOD Engine

**Date:** May 29, 2026  
**Lead Auditor:** Antigravity (Orchestrator & Principal Quantitative Systems Architect)  
**Target:** Quant EOD Engine (EUR/USD, GBP/USD, USD_JPY)  
**Status:** Completed  

---

## Executive Summary

This consolidated report compiles, cross-references, and prioritizes findings from the five independent multi-agent style audits conducted on the `quant-eod-engine` codebase. The audits evaluated the system across five key dimensions:
1. **Quantitative Strategy & Validation (Agent 1)**
2. **Technical Systems & Architecture (Agent 2)**
3. **Execution Infrastructure & Risk Management (Agent 3)**
4. **Alpha Signals & Feature Engineering (Agent 4)**
5. **Closed-Loop Learning & Parameter Adaptation (Agent 5)**

### Overall Assessment
While the trading engine features a highly advanced modular layout combining OANDA/FRED/Perplexity data collection, HMM regime classification, XGBoost meta-labeling, and a closed-loop system adaptation layer, the audit has identified **5 CRITICAL** and **11 HIGH** severity vulnerabilities. 

Many of these issues are structural mismatch bugs that render parts of the system unusable or dangerous in live production. For instance, trading on `USD_JPY` is blocked by spread validation and will instantly stop out due to a hardcoded pip value mismatch. Similarly, the live meta-model is predicting on feature vectors that have been zeroed out due to a nested dictionary wrapping mismatch. 

Below, all 27 findings are categorized and ranked by business severity, followed by a roadmap for immediate remediation.

---

## Master Severity Rankings

### Severity Definitions:
* **CRITICAL:** Vulnerabilities that cause immediate execution failure, guaranteed monetary loss, or complete invalidation of core statistical validation. Must be fixed before any live execution.
* **HIGH:** Serious structural, logic, or data-integrity flaws that cause incorrect trading behaviour, stale predictions, data leakage, or silent failures.
* **MEDIUM:** Important deviations from best practices in risk management, database isolation, performance, or indicator calculations.
* **LOW:** Minor issues, performance bottlenecks, or code cleanliness issues.

---

### CRITICAL Severity Findings

#### F-1: Hardcoded Pip Value & Instrument Mismatch (Execution)
* **Vulnerability:** `order_manager.py` and `risk.py` hardcode the pip value as `0.0001` (representing 1 pip = 0.0001 in standard pairs). 
* **Impact:** For `USD_JPY` (where 1 pip = 0.01), a 20-pip Stop Loss is calculated as `20 * 0.0001 = 0.002`, placing the stop loss 0.2 pips from the entry price. The position will stop out immediately on the next tick, causing a guaranteed loss. Additionally, JPY spreads will be scaled incorrectly (e.g. 1.5 pips is computed as 150 pips), causing spread validation to permanently block all JPY trades.
* **Files:** [execution/risk.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/risk.py#L93), [execution/order_manager.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py#L155)

#### F-2: Live Vector Assembly Signal Mismatch (Signals)
* **Vulnerability:** At prediction time, `daily_loop.py` passes the flat `composite_result` directly into `assemble_feature_vector()` as `signals_summary`. However, `vector.py` extracts signal-related features using nested gets like `signals_summary.get("composite", {}).get(...)`.
* **Impact:** The features `"composite_strength"`, `"primary_signal_direction"`, `"primary_signal_count"`, `"tier2_confirmation_count"`, `"eod_event_reversal"`, and `"event_surprise_magnitude"` are silently zeroed out during live predictions. The meta-model predicts on garbage neutral features, completely destroying its live filtering capability.
* **Files:** [daily_loop.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L333-L335), [features/vector.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/features/vector.py#L91)

#### F-3: Syntax Error in PostgreSQL Interval Parameterization (Closed-Loop Learning)
* **Vulnerability:** In `feedback.py`, SQL parameters (`%s`) are embedded inside string quotes inside the `INTERVAL` clause: `INTERVAL '%s days'`.
* **Impact:** PostgreSQL does not parse placeholders inside quotes. psycopg2 fails to substitute the value, throwing a `ProgrammingError` because of a mismatch in the parameter count. This crashes the feedback cycle on every daily execution.
* **Files:** [learning/feedback.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L44)

#### F-4: Look-Ahead Return Signing Bias in CPCV Validation (Quantitative Strategy)
* **Vulnerability:** The CPCV path return simulation signs the absolute magnitude of the Close-to-Close returns (`r_next`) using the ground-truth label `y_test`: `signals * (2 * y_test - 1) * abs(r_next)`. However, `y_test` is generated from Open-to-Close returns.
* **Impact:** This methodology signs the Close-to-Close returns based on the *future knowledge* of the Open-to-Close direction. For example, if a day has a negative close-to-close return but a positive open-to-close return, a long signal will simulate a positive return, generating massive artificial profits in backtesting and invalidating validation.
* **Files:** [models/meta_model.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L469)

#### F-5: Mathematical Scaling Error in Probabilistic Sharpe Ratio (Quantitative Strategy)
* **Vulnerability:** `_probabilistic_sharpe_ratio_from_returns` calculates the annualized Sharpe ratio (`sr`) and plugs it directly into the Bailey & López de Prado daily variance formula.
* **Impact:** Because the variance formula is derived for daily (non-annualized) Sharpe ratios, using the annualized Sharpe ratio inflates the quadratic terms in the denominator by a factor of $252$, overestimating variance and rendering the resulting PSR metric statistically invalid.
* **Files:** [models/meta_model.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L689-L697)

---

### HIGH Severity Findings

#### F-6: Forex Quote to Account Base Currency Mismatch (Execution)
* **Vulnerability:** The P&L logger calculates `pnl_amount = pnl_pips * pip_val * position_size` in the quote currency (e.g. JPY for USD_JPY), but logs it directly as the account base currency (USD).
* **Impact:** A 1,000 JPY profit on USD_JPY is logged as a $1,000.00 profit, overstating dollar P&L and equity curve calculations by ~150x.
* **Files:** [execution/order_manager.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py#L289)

#### F-7: Position Duplication & Restart Vulnerability (Execution)
* **Vulnerability:** The live trader relies solely on Redis state to check for active positions at startup. It does not query OANDA directly.
* **Impact:** If Redis memory is cleared or the key expires, the system will assume no position is active and may open duplicate concurrent positions on OANDA, violating portfolio risk limits.
* **Files:** [execution/live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L109-L114)

#### F-8: Unreliable Broker Sync Modulo Check (Execution)
* **Vulnerability:** In `live_trader.py`, the periodic OANDA trade status check runs when the message ID timestamp ends in `00` modulo 100: `int(msg_id.split("-")[0]) % 100 == 0`.
* **Impact:** Because ticks arrive at irregular intervals, the millisecond timestamp can skip `00` modulo 100 for hours during low-activity periods, leaving the trader unaware of broker-side closures. Conversely, multiple ticks inside the same millisecond will trigger duplicate API queries.
* **Files:** [execution/live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L250)

#### F-9: Database Value Truncation on Signal Storage (Signals)
* **Vulnerability:** Unconfirmed Tier 2 signals write the string `"not_confirmed"` to the `direction` column of the `signals` table.
* **Impact:** The column is defined as `VARCHAR(10)`. Inserting the 13-character string `"not_confirmed"` causes a database truncation exception that halts the daily loop.
* **Files:** [signals/composite.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/composite.py#L134), [sql/schema_phase2.sql](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_phase2.sql#L33)

#### F-10: Timezone / Session Mismatch in Slippage Join (Closed-Loop Learning)
* **Vulnerability:** The slippage join connects `live_trades` to `opportunity_measurements` using `t.entry_time::date = o.trade_date`.
* **Impact:** `entry_time` is a UTC timestamp, but `trade_date` is a NY trading session date (5 PM NY to 5 PM NY). Trades opened between 5:00 PM NY and midnight UTC (belonging to the next session date) are joined to the wrong session calendar date, corrupting the slippage analysis.
* **Files:** [learning/feedback.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L126)

#### F-11: Feedback Loop Instability & Trade Starvation (Closed-Loop Learning)
* **Vulnerability:** Proportional parameter adaptation runs on a rolling 20-day lookback window.
* **Impact:** Since daily FX strategies have low trade frequencies, a 20-day window contains only a few samples. A brief sequence of losses will cause the win rate to plunge, inflating calibration drift. This raises probability gates to the maximum and slashes Kelly sizes. The high gates block all new trades, trapping the system in a permanent trade starvation deadlock.
* **Files:** [learning/feedback.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L20), [learning/feedback.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L100-L103)

#### F-12: Look-Ahead Feature Leakage via Future H4 Bars in Backfill (Quantitative Strategy)
* **Vulnerability:** Feature generation loads H4 bars using `date(bar_time) <= run_date` in UTC.
* **Impact:** This query includes H4 bars completed in the evening of `run_date` (which are ahead of the 5:15 PM NY daily loop cutoff), leaking future intraday price information into today's daily feature vectors during training and backtesting.
* **Files:** [generate_historical_features.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/generate_historical_features.py#L120-L136)

#### F-13: Overfitting via Static Out-of-Sample Backtesting (Quantitative Strategy)
* **Vulnerability:** The walk-forward test uses a single pre-trained meta-model rather than re-fitting it periodically.
* **Impact:** The model does not adapt to structural regime shifts or parameter drift over the test period, overstating the model's out-of-sample stability.
* **Files:** [walkforward_test.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/walkforward_test.py#L365-L375)

#### F-14: Silent Failure Propagation in daily_loop.py (Architecture)
* **Vulnerability:** Steps in the pipeline are caught in individual try/except blocks, logging errors but continuing execution.
* **Impact:** If Step 7 (technical indicator calculation) or Step 9 (signal generation) fails, the model receives a feature vector filled with default values (such as `atr_14 = 0.0` or `rsi_14 = 50.0`). The meta-model predicts on this garbage data and publishes a live prediction.
* **Files:** [daily_loop.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L278-L280)

#### F-15: Stale Predictions Read by Live Trader (Architecture)
* **Vulnerability:** Redis prediction and regime keys do not store a target date, and `live_trader.py` does not check for timestamps.
* **Impact:** If the daily pipeline fails to run, the live trader will trade on yesterday's stale prediction today.
* **Files:** [execution/live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L58-L65)

#### F-16: Missing Database Tables in Docker-Compose Mounts (Architecture)
* **Vulnerability:** `schema_closed_loop.sql` and `schema_migration_model_blobs.sql` are missing from the Postgres mounts in `docker-compose.yml`.
* **Impact:** On a fresh Docker deployment, database tables like `model_artifacts`, `events`, `opportunity_measurements`, and `learning_runs` will be missing, causing worker failures.
* **Files:** [docker-compose.yml](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/docker-compose.yml#L13-L16)

---

### MEDIUM Severity Findings

#### F-17: German Government Bond Yield Curve Mismatch (Signals)
* **Vulnerability:** Configures `FRED_DE_2Y_SERIES` to use `IRLTLT01DEM156N` (the German 10-Year yield) as a proxy for the 2-Year yield.
* **Impact:** The US-DE spread subtracts a 10Y yield from a 2Y yield, creating a term-structure mismatch that distorts the monetary policy differential signal.
* **Files:** [config/settings.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/config/settings.py#L50)

#### F-18: Kalman Filter Parameter Instability (Execution)
* **Vulnerability:** The Kalman filter's measurement noise covariance is initialized at `R_val = 1e-3` (316 pips standard deviation).
* **Impact:** Because true tick noise is less than 1 pip, the filter is excessively sluggish, introducing high lag. The estimated velocity is heavily dampened, making it difficult to cross entry thresholds during genuine momentum.
* **Files:** [physics/kalman.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/physics/kalman.py#L10), [execution/live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L128)

#### F-19: Drawdown Timezone Mismatch (Execution)
* **Vulnerability:** Daily realized PnL queries use PostgreSQL `CURRENT_DATE` in UTC.
* **Impact:** FX trading days roll at 5:00 PM NY time. Trades closed in the evening of NY time are grouped under the wrong trading day's drawdown limits.
* **Files:** [execution/risk.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/risk.py#L143)

#### F-20: Database Connection Leak / Overhead (Architecture)
* **Vulnerability:** Database helpers open and close a new database connection for every query rather than using a pool.
* **Impact:** Incurs high latency and risks socket exhaustion under load.
* **Files:** [models/database.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/database.py#L17-L75)

#### F-21: Transaction Isolation Failures (Architecture)
* **Vulnerability:** Pipeline writes are run in separate connection scopes.
* **Impact:** Database updates are non-atomic. Incomplete state (e.g. regime saved but features missing) is visible to concurrent readers.
* **Files:** [daily_loop.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L296)

#### F-22: Float vs Numeric Data Types (Architecture)
* **Vulnerability:** `live_trades` uses `DOUBLE PRECISION` for prices and P&L.
* **Impact:** Violates financial data integrity guidelines by introducing binary floating-point rounding errors.
* **Files:** [sql/schema_physics.sql](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_physics.sql#L11-L16)

#### F-23: Violations of IID Assumptions in CPCV Significance Testing (Quantitative Strategy)
* **Vulnerability:** Uses a one-sample t-test to evaluate the significance of path Sharpes.
* **Impact:** Since CPCV paths share the same underlying data, they are highly correlated. The t-test underestimates variance, inflating statistical significance.
* **Files:** [models/meta_model.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L490)

---

### LOW Severity Findings

#### F-24: Performance Bottleneck via Repeated GPU Probing (Quantitative Strategy)
* **Vulnerability:** Calls `_get_xgb_device()` doing a full model fit inside every CPCV fold.
* **Impact:** Creates a CPU/GPU initialization bottleneck inside the cross-validation loop.
* **Files:** [models/meta_model.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L458)

#### F-25: Asymmetric Composite Signal Bias (Signals)
* **Vulnerability:** Asymmetric confirm/deny weights (+0.05 vs -0.02) favor trade-taking on weak signals.
* **Files:** [signals/composite.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/composite.py#L77-L86)

#### F-26: Lack of PostgreSQL Parameter Fallback (Closed-Loop Learning)
* **Vulnerability:** The live trader does not check PostgreSQL if Redis is down, reverting to default parameters.
* **Files:** [execution/live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L149)

#### F-27: Dead Code in Parameter Adaptation (Closed-Loop Learning)
* **Vulnerability:** A loop building `expected_prices` contains only `pass`.
* **Files:** [learning/feedback.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L114-L118)

---

## Remediation Roadmap

To prepare the codebase for secure live execution, we recommend a 3-phase remediation plan:

### Phase 1: Immediate Critical Fixes (Before Live Testing)
1. **JPY Sizing & Spread Fix:** Refactor [risk.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/risk.py) and [order_manager.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py) to dynamically fetch JPY-specific pip values (`0.01`). Convert JPY P&L to USD using current rates.
2. **Signal Feature Wrapping:** Fix `daily_loop.py` to wrap the composite dictionary inside a `"composite"` key before vector assembly.
3. **Fix Interval syntax:** Correct the SQL queries in `feedback.py` to concatenate parameters securely outside the quotes: `o.date >= %s - (%s || ' days')::INTERVAL`.
4. **CPCV Return Simulation:** Refactor `_run_cpcv` to simulate returns using the pre-calculated actual realized open-to-close returns of the proposed trades, removing the Frankenstein return calculation.
5. **Daily PSR scaling:** Correct the PSR variance calculation by using the non-annualized (daily) Sharpe ratio inside the variance formula, and annualize the final output.

### Phase 2: System Architecture & Data Integrity (High Severity)
1. **OANDA Broker Check:** Implement a time-based check (using `time.time()`) inside `live_trader.py` to replace the millisecond modulo check. Query OANDA at startup to reconstruct `active_trade` state.
2. **Docker Compose Sync:** Add mounts for `schema_closed_loop.sql` and `schema_migration_model_blobs.sql` in `docker-compose.yml`.
3. **Pipeline Failure Halting:** Update `daily_loop.py` to halt execution immediately if critical data collection or technical indicator steps throw exceptions.
4. **Timezone Adjustments:** Adjust database queries for daily PnL and slippage calculation to filter based on explicit UTC timestamps representing NY session boundaries (5 PM NY).
5. **H4 Leakage Fix:** Restrict the H4 bar query in `generate_historical_features.py` to bars ending before 5 PM NY time on the run date.
6. **Walk-Forward Refitting:** Update `walkforward_test.py` to refit the XGBoost meta-model periodically (e.g., monthly).

### Phase 3: Risk & Performance (Medium & Low Severity)
1. **Implement Connection Pool:** Introduce `psycopg2.pool.ThreadedConnectionPool` in `database.py` and manage connections using context managers.
2. **Numeric Migration:** Migrate the `live_trades` table columns (`price`, `pnl`) from `DOUBLE PRECISION` to `NUMERIC`.
3. **Feedback Loop Stabilization:** Increase the lookback window in the learner to a minimum of 60 days or 30 trades, and apply exponential smoothing to parameter updates.
4. **FRED DE 2Y Source:** Replace the FRED 10Y German interest rate series with actual German 2Y government bond yields.
