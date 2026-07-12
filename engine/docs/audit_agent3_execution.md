# Execution Infrastructure & Risk Management Audit Report

**Date:** May 29, 2026  
**Auditor:** Antigravity (Trading Systems Engineer)  
**Target:** Quant EOD Engine — Live Execution & Risk Infrastructure  
**Status:** Completed  

---

## Executive Summary

This execution and risk audit evaluates the `quant-eod-engine` for transaction integrity, latency risk, broker-state synchronization, positioning bugs, and execution logic correctness.

**Overall Assessment:** The system has critical execution flaws that will cause immediate losses or prevent normal operation on certain currency pairs. Most notably, the hardcoded `pip_value = 0.0001` across the execution and risk modules will lead to instant stop-outs and spread-validation failures on `USD_JPY` (where 1 pip = 0.01). Furthermore, the broker-side synchronization uses an unreliable modulo check on millisecond timestamps, leading to missed exit events, and the system is vulnerable to position duplication upon service restarts.

A summary table of vulnerabilities, rated by severity, is provided at the end of this report.

---

## 1. Hardcoded Pip Value & Instrument Mismatch (CRITICAL)

### Findings & Analysis
The system configures `USD_JPY` as one of its tradable instruments in [settings.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/config/settings.py#L45). However, the execution and risk modules hardcode the pip value as `0.0001` (appropriate only for EUR/USD, GBP/USD, etc.):
1. **Spread Validation:** In `validate_spread()` in [risk.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/risk.py#L93), the default `pip_value` is `0.0001`. At prediction/execution time, `live_trader.py` calls `risk_manager.validate_spread(bid, ask)` without passing an instrument-specific pip value. For `USD_JPY` (trading around 150.00), a standard spread of 1.5 pips is $150.015 - 150.000 = 0.015$. The function calculates:
   $$spread\_pips = \frac{0.015}{0.0001} = 150 \text{ pips}$$
   Since $150 > 2.0$ (the max spread threshold), the system will flag the spread as too wide and refuse to trade. Consequently, **no trades will ever be executed on USD_JPY**.
2. **Order SL/TP Placement:** In `execute_market_order()` in [order_manager.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py#L155), `pip_val` is hardcoded to `0.0001`. For a short `USD_JPY` order at $150.00$, the stop loss is computed as:
   $$sl\_price = 150.00 + (20 \times 0.0001) = 150.002$$
   In reality, a 20-pip stop loss on USD_JPY should be at $150.20$. By setting it to $150.002$, the stop loss is placed **0.2 pips** away from the entry price. The trade will be stopped out immediately on the next tick, causing a guaranteed loss.
3. **PnL & Pips Logging:** In `_log_exit_to_db()` in [order_manager.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py#L283), `pip_val = 0.0001` is hardcoded. For USD_JPY, a 10-pip win ($150.10 - 150.00$) will be logged as a 1000-pip win.

### Code References
* Hardcoded in order manager: [order_manager.py:L155](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py#L155), [order_manager.py:L283](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py#L283)
* Hardcoded in risk manager: [risk.py:L93](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/risk.py#L93)

### Severity: CRITICAL
### Recommendations
* Dynamically retrieve the pip value based on the instrument name (e.g., check if `"JPY"` is in the instrument name and return `0.01`, otherwise `0.0001`).
* Refactor [risk.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/risk.py) and [order_manager.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py) to pass the active instrument to all pip and spread calculations.

---

## 2. Forex Quote to Account Base Currency Mismatch (HIGH)

### Findings & Analysis
The dollar P&L calculation in [order_manager.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py#L289) uses the formula:
```python
pnl_amount = pnl_pips * pip_val * position_size
```
For `EUR_USD`, this is dimensionally correct because the quote currency is USD, and the account base currency is USD. A 10-pip gain on 10,000 units is $10 \times 0.0001 \times 10,000 = \$10.00$.
However, for any pair where the quote currency is NOT USD (e.g., `USD_JPY`, where the quote is JPY):
* The P&L is denominated in JPY.
* For a 10-pip gain on 10,000 units of `USD_JPY` (with correct JPY pip value `0.01`), the P&L is $10 \times 0.01 \times 10,000 = 1,000 \text{ JPY}$.
* The system logs this directly as `pnl_amount = 1000.0`, storing it in the database as **$1,000.00 USD**!
In reality, the 1,000 JPY profit must be converted back to USD by dividing by the USD_JPY exchange rate (e.g., $1,000 / 150 = \$6.67 \text{ USD}$). Logging unconverted quote currency directly as USD creates massive errors in equity tracking and drawdown checks.

### Code References
* P&L calculation: [order_manager.py:L289](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/order_manager.py#L289)

### Severity: HIGH
### Recommendations
* If the quote currency of the instrument does not match the account base currency (USD), convert the realized PnL to USD using the current exchange rate of the quote pair before logging to `pnl_amount` and the database.

---

## 3. Position Duplication & Restart Vulnerability (HIGH)

### Findings & Analysis
In [live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L109-L114), the trader attempts to synchronize its local `active_trade` dictionary from Redis at startup:
```python
    active_trade = sync_active_trade_from_redis(redis_client, instrument)
```
If this key does not exist (e.g., because Redis was restarted, memory flushed, or the key expired), `active_trade` remains `None`. 
However, an active position might still be open broker-side on OANDA. Because `active_trade` is `None`, the live trader's entry logic is active. When a new entry signal fires, the trader will call `execute_market_order()`, opening a **second concurrent position** for the same instrument on OANDA.
This violates the core risk assumption of the engine, which mandates a maximum of one active position per instrument. Opening duplicate positions doubles the risk exposure and will violate the portfolio's drawdown constraints.

### Code References
* Local active trade check: [live_trader.py:L109-L114](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L109-L114)
* Entry condition check: [live_trader.py:L291-L302](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L291-L302)

### Severity: HIGH
### Recommendations
* At startup, the trader must query the OANDA API directly (using `requests.get` to `/v3/accounts/{accountID}/trades`) to check if any open trades exist for the instrument.
* If a trade exists on OANDA but is missing in Redis, reconstruct the `active_trade` dictionary from OANDA's response and save it to Redis before entering the loop.

---

## 4. Unreliable Broker Sync Modulo Check (HIGH)

### Findings & Analysis
In [live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L250), the system implements a check to see if the trade has been closed broker-side (e.g., hit SL or TP):
```python
if int(msg_id.split("-")[0]) % 100 == 0:
```
`msg_id` is a Redis stream ID formatted as `<timestamp_ms>-<sequence>`.
The condition checks whether the timestamp in milliseconds is exactly divisible by 100. This is highly unreliable for rate-limiting OANDA queries:
* **Missing Check:** Ticks arrive at irregular intervals. If there are periods of low volatility or low tick activity, consecutive tick timestamps might never end in exactly `00` modulo 100. The system would completely skip checking the broker for hours, failing to realize a trade was stopped out.
* **Double Check:** If multiple ticks arrive within the same millisecond ending in `00` (e.g., during high volatility or processing backlog), the system will execute multiple HTTP requests to OANDA within a few milliseconds, hammering the broker API.

### Code References
* Modulo check: [live_trader.py:L250](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L250)

### Severity: HIGH
### Recommendations
* Replace the modulo check with a simple elapsed time check. Track the last epoch timestamp when the broker was queried, and query only if `current_time - last_query_time >= 5.0` seconds:
  ```python
  now = time.time()
  if now - last_broker_check > 5.0:
      last_broker_check = now
      # Check broker status...
  ```

---

## 5. Kalman Filter Parameter Instability (MEDIUM)

### Findings & Analysis
The Kalman filter in [kalman.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/physics/kalman.py) is initialized with measurement noise covariance `R_val = 1e-3` (0.001) and process noise acceleration variance `Q_accel = 1e-5`.
For `EUR_USD` (trading around 1.08):
* A variance of `1e-3` corresponds to a standard deviation of $\sqrt{0.001} \approx 0.0316$ price points (or **316 pips**).
* Ticks on EUR_USD typically move in increments of 0.1 pips ($1 \times 10^{-5}$). Setting the measurement noise to 316 pips tells the filter that every tick is massive noise.
* This over-estimation of measurement noise causes the filter to smooth the price extremely heavily. The smoothed price will lag the actual price by several minutes, and the estimated velocity will be severely dampened.
* In [live_trader.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L128), the entry velocity threshold is `1.5e-6` (0.015 pips/sec). Because the filter is so sluggish, the estimated velocity will rarely cross this threshold during genuine short-term price momentum, or it will cross it with a significant time lag, leading to poor entry execution.

### Code References
* Filter parameters: [kalman.py:L10](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/physics/kalman.py#L10)
* Entry threshold: [live_trader.py:L128](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/live_trader.py#L128)

### Severity: MEDIUM
### Recommendations
* Recalibrate the Kalman filter parameters using historical tick data.
* A more realistic measurement noise `R_val` for tick data is around `2.5e-9` (0.5 pips standard deviation), which will make the filter responsive to price moves while still filtering out micro-structural noise.

---

## 6. Drawdown Timezone Mismatch (MEDIUM)

### Findings & Analysis
In `_get_today_realized_pnl()` in [risk.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/risk.py#L128-L149), the system queries database realized PnL using the database server's local date (`CURRENT_DATE`, which is UTC):
```sql
SELECT COALESCE(SUM(pnl_amount), 0.0) 
FROM live_trades 
WHERE exit_time >= CURRENT_DATE 
```
However, the FX trading day is anchored at **5:00 PM Eastern Time (New York)**. 
Because of this timezone mismatch:
* Trades closed between 5:00 PM NY time and 12:00 AM UTC are counted as belonging to the *previous* day's database PnL in UTC, even though they belong to the *next* trading session.
* Cumulative daily losses can be calculated incorrectly, allowing trading to continue when the 2% daily drawdown limit has actually been breached, or halting trading prematurely based on a timezone offset.

### Code References
* SQL timezone query: [risk.py:L140-L144](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/execution/risk.py#L140-L144)

### Severity: MEDIUM
### Recommendations
* Calculate the 5:00 PM NY session cutoff time in Python using the local timezone, and pass the explicit UTC timestamp boundary as a parameter to the SQL query instead of using `CURRENT_DATE`.

---

## Summary of Findings

| ID | Finding Title | Severity | Files Affected | Status |
|---|---|---|---|---|
| F-1 | Hardcoded Pip Value & Instrument Mismatch | **CRITICAL** | `execution/risk.py`, `execution/order_manager.py` | Open |
| F-2 | Forex Quote to Account Base Currency Mismatch | **HIGH** | `execution/order_manager.py` | Open |
| F-3 | Position Duplication & Restart Vulnerability | **HIGH** | `execution/live_trader.py` | Open |
| F-4 | Unreliable Broker Sync Modulo Check | **HIGH** | `execution/live_trader.py` | Open |
| F-5 | Kalman Filter Parameter Instability | **MEDIUM** | `physics/kalman.py`, `execution/live_trader.py` | Open |
| F-6 | Drawdown Timezone Mismatch | **MEDIUM** | `execution/risk.py` | Open |
