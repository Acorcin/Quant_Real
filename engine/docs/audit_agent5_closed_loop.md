# Closed-Loop Learning & Feedback Loop Audit Report

**Date:** May 29, 2026  
**Auditor:** Antigravity (ML Systems Engineer)  
**Target:** Quant EOD Engine — Layer 8 Closed-Loop Learning & Parameter Adaptation  
**Status:** Completed  

---

## Executive Summary

This machine learning and closed-loop systems audit evaluates the `quant-eod-engine` for parameter stability, SQL injection risks, feedback loop dynamics, and data pipeline integrity.

**Overall Assessment:** The newly added Closed-Loop Learning architecture (Layers 3, 4, and 8) contains critical logic errors that prevent reliable online parameter adaptation. Most notably, a SQL query syntax error in the parameterization of the `INTERVAL` clause will crash the learning cycle in production. Furthermore, a timezone/session boundary mismatch in the slippage join will cause the learner to misalign or entirely miss execution trades. The feedback loop also suffers from statistical instability due to an extremely short 20-day lookback window on low-frequency data, which will cause parameter oscillation and trade starvation.

A summary table of vulnerabilities, rated by severity, is provided at the end of this report.

---

## 1. Syntax Error in PostgreSQL Interval parameterization (CRITICAL)

### Findings & Analysis
In [feedback.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L44), the SQL queries to fetch opportunity measurements and live trades contain a syntax error in the parameterization of the `INTERVAL` clause:
```sql
WHERE o.instrument = %s AND o.date >= %s - INTERVAL '%s days'
```
In PostgreSQL, parameters (`%s`) cannot be placed inside single quotes of a string literal such as `INTERVAL '%s days'`. The database parser treats `'%s days'` as a literal string containing the characters `%` and `s`. 
Because psycopg2 expects three parameters but only finds two valid placeholders outside of quotes (for `instrument` and `date`), it will throw a `psycopg2.ProgrammingError` on execution. This completely breaks the closed-loop learning cycle, preventing `run_feedback_cycle` from ever completing successfully in production.

### Code References
* Broken interval query: [feedback.py:L44](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L44), [feedback.py:L55](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L55), [feedback.py:L127](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L127)

### Severity: CRITICAL
### Recommendations
* Refactor the interval clause to use proper SQL parameterization by passing the interval string as a single parameter or concatenating it using Postgres operators:
  ```sql
  WHERE o.date >= %s - (%s || ' days')::INTERVAL
  ```
  And pass `self.lookback_days` as the third parameter.

---

## 2. Timezone / Session Mismatch in Slippage Join (HIGH)

### Findings & Analysis
The slippage calculation query in [feedback.py:L121-L130](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L121-L130) joins the `live_trades` table with `opportunity_measurements` using:
```sql
FROM live_trades t
JOIN opportunity_measurements o ON t.entry_time::date = o.trade_date AND t.instrument = o.instrument
```
This join assumes that the calendar date of the trade entry in UTC matches the trading calendar date of the session (`trade_date`).
However:
* `live_trades.entry_time` is a `TIMESTAMPTZ` recorded in UTC.
* `opportunity_measurements.trade_date` is a `DATE` representing the trading session (5:00 PM NY to 5:00 PM NY).
For example, a trade entered on Monday at 6:00 PM New York time belongs to the **Tuesday trading session** (and OANDA daily bar).
* In `live_trades`, `entry_time` will be Monday 10:00 PM UTC (or 11:00 PM UTC depending on DST). Thus, `t.entry_time::date` will return **Monday**.
* In `opportunity_measurements`, `o.trade_date` will be **Tuesday**.
Because Monday does not equal Tuesday, the join will fail for all trades entered during the first 7 hours of the trading session (between 5:00 PM NY and 12:00 AM UTC). This results in missed joins, incorrect slippage calculations, and parameters adapted on corrupted execution data.

### Code References
* Broken date join: [feedback.py:L126](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L126)

### Severity: HIGH
### Recommendations
* Modify the join condition to align timestamps based on Eastern Time (NY) session boundaries:
  ```sql
  ON timezone('America/New_York', t.entry_time)::date = o.trade_date
  ```
  This ensures that a trade opened at 6:00 PM NY on Monday is correctly mapped to the Tuesday session date (which starts on Monday EOD).

---

## 3. Feedback Loop Instability & Trade Starvation (HIGH)

### Findings & Analysis
The learner adapts position sizing (`kelly_fraction`), probability thresholds (`prob_threshold_half`/`prob_threshold_full`), and entry velocity thresholds based on a rolling **20-day lookback window**:
1. **Low Sample Size:** EOD daily FX strategies typically generate fewer than 10-15 trades per month. If the meta-model filters out trades, the actual number of executed trades inside the 20-day window can drop to 3-5.
2. **Statistical Noise:** Calculating win rates and calibration drift on 3-5 samples is statistically meaningless. A sequence of 2-3 consecutive losses due to market noise will cause the win rate to plunge to 0%, resulting in a massive calibration drift (e.g. 0.70 confidence - 0% win rate = 0.70 drift).
3. **Trade Starvation:** In response to the high drift, the learner will immediately raise probability gates to the maximum (`0.65` for half size, `0.80` for full size) and slash the Kelly fraction to `0.05`. Because the gates are now extremely high, the system will block almost all new trades.
4. **Deadlock / Stagnation:** Since no new trades are executed, the sample size remains zero. The system enters a deadlock state where it cannot collect new trade data to lower the gates, resulting in permanent trade starvation.

### Code References
* Proportional adjustments: [feedback.py:L100-L103](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L100-L103)
* Lookback window: [feedback.py:L20](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L20)

### Severity: HIGH
### Recommendations
* Increase the lookback window to a statistically meaningful sample size (e.g., minimum 60-90 days, or a minimum threshold of 30 historical trades).
* Implement damping/smoothing (e.g., an Exponential Moving Average) on parameter updates to prevent abrupt, volatile shifts from a single bad sequence.
* Introduce a minimum trade frequency guard that prevents the probability gates from rising to levels that halt all trading activity.

---

## 4. Lack of PostgreSQL Parameter Fallback (MEDIUM)

### Findings & Analysis
When `ClosedLoopLearner` finishes adapting parameters, it writes them to Redis (`{instrument}:learning_params`) and logs the run in the PostgreSQL `learning_runs` table.
During live execution, `live_trader.py` queries Redis for the adapted parameters:
```python
lp_data = redis_client.get(f"{instrument}:learning_params")
```
If Redis is down, or if the keys have expired/been cleared, `live_trader.py` silently falls back to hardcoded system default parameters (`kelly_fraction = 0.15`, `velocity_entry_threshold = 1.5e-6`, etc.).
The system does not attempt to query the PostgreSQL `learning_runs` table as a secondary fallback. This means the engine loses its entire historical learning memory upon Redis restarts, reverting back to uncalibrated defaults.

### Code References
* Redis check in live trader: [live_trader.py:L149-L169](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L149-L169)

### Severity: MEDIUM
### Recommendations
* Modify `live_trader.py` to query the PostgreSQL `learning_runs` table for the latest adapted parameters if the Redis key is missing or Redis is unavailable. Cache these values in Redis upon successful database lookup.

---

## 5. Dead Code and Mocks in Tests (LOW)

### Findings & Analysis
* **Dead Code:** In [feedback.py:L114-L118](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L114-L118), there is a loop building `expected_prices` by date that does nothing and contains only `pass`. This should be removed to maintain code cleanliness.
* **Test Mocks:** The test suite in `test_closed_loop.py` mocks all database interactions (`fetch_all`, `get_connection`). While this is standard for unit testing, it masked the critical syntax error in the `INTERVAL` SQL parameterization, which went undetected until manual code review.

### Code References
* Dead code loop: [feedback.py:L114-L118](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/learning/feedback.py#L114-L118)

### Severity: LOW
### Recommendations
* Delete the dead code block.
* Add integration tests that execute queries against a test database instance (e.g., PostgreSQL in Docker) to validate SQL syntax and data types.

---

## Summary of Findings

| ID | Finding Title | Severity | Files Affected | Status |
|---|---|---|---|---|
| F-1 | Syntax Error in PostgreSQL Interval parameterization | **CRITICAL** | `learning/feedback.py` | Open |
| F-2 | Timezone / Session Mismatch in Slippage Join | **HIGH** | `learning/feedback.py` | Open |
| F-3 | Feedback Loop Instability & Trade Starvation | **HIGH** | `learning/feedback.py` | Open |
| F-4 | Lack of PostgreSQL Parameter Fallback | **MEDIUM** | `execution/live_trader.py` | Open |
| F-5 | Dead Code and Mocks in Tests | **LOW** | `learning/feedback.py`, `tests/test_closed_loop.py` | Open |
