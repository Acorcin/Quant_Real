# Code Fixes Guide — Quant EOD Engine

**Comprehensive fix instructions for all critical bugs identified in code review.**

---

## ✅ Already Fixed

### 1. `models/hmm_regime.py`
- ✅ Changed `REGIME_LABELS[0]` from `"low_vol_trend"` to `"low_vol"`
- ✅ Added `_load_persisted_state_map()` method
- ✅ Added state-flip guard in `fit()` method with warning logs

### 2. `utils/trading_calendar.py`
- ✅ Created new utility module
- ✅ Added `next_trading_day(run_date)` function
- ✅ Added `is_trading_day(d)` helper
- ✅ Includes 2026 holiday calendar

---

## 🔧 Fixes to Apply

### 3. `models/meta_model.py` — Fix prediction_for date

**Line ~306 (in `store_prediction` method):**

```python
# BEFORE:
prediction_for = run_date  # In practice, next trading day

# AFTER:
from utils.trading_calendar import next_trading_day
prediction_for = next_trading_day(run_date)
```

**Also add import at top of file (after line 22):**
```python
from utils.trading_calendar import next_trading_day
```

---

### 4. `models/meta_model.py` — Fix CPCV Deflated Sharpe Ratio

**Lines ~210-235 (in `_run_cpcv` method, after calculating `sharpe_array`):**

```python
# BEFORE (lines ~230-235):
n_paths = len(sharpe_array)
se_sharpe = sharpe_std / np.sqrt(max(n_paths, 1))
deflated_sharpe = sharpe_mean / se_sharpe if se_sharpe > 0 else 0.0
return {
    "deflated_sharpe": round(deflated_sharpe, 4),
    "statistically_significant": deflated_sharpe > 1.96,
    ...
}

# AFTER:
# NOTE: True DSR requires scipy.stats. This is a simplified conservative approach.
# For production, implement full Lopez de Prado DSR with skew/kurtosis correction.
n_paths = len(sharpe_array)
if n_paths < 3:
    deflated_sharpe = 0.0
    stat_sig = False
else:
    # Use t-test: is mean Sharpe significantly different from zero?
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(sharpe_array, popmean=0.0)
    deflated_sharpe = float(t_stat)  # Conservative proxy for DSR
    stat_sig = (p_value < 0.05) and (sharpe_mean > 0)

return {
    "deflated_sharpe": round(deflated_sharpe, 4),
    "statistically_significant": stat_sig,
    "p_value": round(float(p_value), 4) if n_paths >= 3 else 1.0,
    ...
}
```

**Add scipy import at top:**
```python
from scipy import stats
```

---

### 5. `models/meta_model.py` — Fix CPCV return proxy

**Lines ~205-207 (in `_run_cpcv` method):**

```python
# BEFORE:
signals = (probs > 0.55).astype(float)
# Assume direction is correct when label=1
daily_returns = signals * (2 * y_test - 1) * 0.001  # simplified return proxy

# AFTER:
# Load actual returns from DB for backtest realism
from models.database import fetch_all
bar_dates = df.iloc[test_idx].index.date.tolist()  # Assumes df has DatetimeIndex
if len(bar_dates) > 0:
    placeholders = ','.join(['%s'] * len(bar_dates))
    ret_rows = fetch_all(f"""
        SELECT bar_time::date as d, 
               (close / LAG(close) OVER (ORDER BY bar_time)) - 1 as ret
        FROM bars 
        WHERE instrument = 'EUR_USD' AND granularity = 'D' 
          AND bar_time::date IN ({placeholders})
    """, tuple(str(d) for d in bar_dates))
    ret_map = {row['d']: float(row['ret']) for row in ret_rows}
    actual_returns = np.array([ret_map.get(d, 0.0) for d in bar_dates])
else:
    actual_returns = np.zeros(len(y_test))

signals = (probs > 0.55).astype(float)
# Direction: 1 if y_test=1 (up), -1 if y_test=0 (down)
direction = 2 * y_test - 1
daily_returns = signals * direction * actual_returns
```

**NOTE:** This assumes your `bars` table has a `close` column and you're working with EUR_USD. Adjust query as needed.

---

### 6. `models/meta_model.py` — Fix in-sample metrics

**Lines ~95-105 (in `train()` method, after `self.model.fit(X, y)`):**

```python
# BEFORE:
y_pred = self.model.predict(X)
metrics = {
    "accuracy": round(accuracy_score(y, y_pred), 4),
    ...
}

# AFTER:
# Remove in-sample metrics entirely, or clearly label them:
metrics_in_sample = {
    "note": "These are IN-SAMPLE training metrics — not for validation!",
    "accuracy_train": round(accuracy_score(y, self.model.predict(X)), 4),
    "samples": len(X),
    "positive_rate": round(y.mean(), 4),
}
# Return only CPCV metrics as the real validation:
return {
    "model_version": self.model_version,
    "cpcv": cpcv,
    "top_features": self.shap_importance[:10],
    "train_in_sample": metrics_in_sample,  # Clearly labeled
}
```

---

### 7. `fetchers/fred_yields.py` — Store spread ROC columns

**Lines ~65-75 (in `fetch_yields()` after computing `change_1d`, `change_5d`, `change_20d`):**

```python
# ADD new calculations for SPREAD rate-of-change:
spread_change_5d = None
spread_change_20d = None

if de_2y is not None and not de_2y.empty and len(us_2y) >= 6:
    # Compute historical spread series
    # Align indices
    spread_series = (us_2y - de_2y.reindex(us_2y.index, method='ffill')) * 100
    spread_series = spread_series.dropna()
    
    if len(spread_series) >= 6:
        spread_change_5d = round(float(spread_series.iloc[-1] - spread_series.iloc[-6]), 2)
    if len(spread_series) >= 21:
        spread_change_20d = round(float(spread_series.iloc[-1] - spread_series.iloc[-21]), 2)

result = {
    ...
    "us_2y_change_5d_bps": change_5d,
    "us_2y_change_20d_bps": change_20d,
    "spread_change_5d_bps": spread_change_5d,      # ADD
    "spread_change_20d_bps": spread_change_20d,    # ADD
    ...
}
```

**Lines ~90-100 (in `store_yields()`):**

```python
# BEFORE:
cur.execute("""
    INSERT INTO yield_data (date, us_2y_yield, de_2y_yield, yield_spread_bps, source)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (date, source)
    DO UPDATE SET ...
""", (...))

# AFTER:
cur.execute("""
    INSERT INTO yield_data 
        (date, us_2y_yield, de_2y_yield, yield_spread_bps, 
         us_2y_change_5d_bps, us_2y_change_20d_bps,
         spread_change_5d_bps, spread_change_20d_bps, source)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (date, source)
    DO UPDATE SET
        us_2y_yield = EXCLUDED.us_2y_yield,
        de_2y_yield = EXCLUDED.de_2y_yield,
        yield_spread_bps = EXCLUDED.yield_spread_bps,
        us_2y_change_5d_bps = EXCLUDED.us_2y_change_5d_bps,
        us_2y_change_20d_bps = EXCLUDED.us_2y_change_20d_bps,
        spread_change_5d_bps = EXCLUDED.spread_change_5d_bps,
        spread_change_20d_bps = EXCLUDED.spread_change_20d_bps,
        fetched_at = NOW()
""", (
    data["date"], data["us_2y_yield"], data["de_2y_yield"],
    data["yield_spread_bps"],
    data.get("us_2y_change_5d_bps"), data.get("us_2y_change_20d_bps"),
    data.get("spread_change_5d_bps"), data.get("spread_change_20d_bps"),
    data["source"],
))
```

**Also update the SQL schema in `sql/schema.sql`:**
```sql
ALTER TABLE yield_data ADD COLUMN IF NOT EXISTS us_2y_change_5d_bps NUMERIC;
ALTER TABLE yield_data ADD COLUMN IF NOT EXISTS us_2y_change_20d_bps NUMERIC;
ALTER TABLE yield_data ADD COLUMN IF NOT EXISTS spread_change_5d_bps NUMERIC;
ALTER TABLE yield_data ADD COLUMN IF NOT EXISTS spread_change_20d_bps NUMERIC;
```

---

### 8. `features/vector.py` — Use spread change, not US-only

**Lines ~45-47:**

```python
# BEFORE:
"yield_spread_change_5d": macro.get("us_2y_change_5d_bps", 0.0),
"yield_spread_change_20d": macro.get("us_2y_change_20d_bps", 0.0),

# AFTER:
"yield_spread_change_5d": macro.get("spread_change_5d_bps", 0.0),
"yield_spread_change_20d": macro.get("spread_change_20d_bps", 0.0),
```

---

### 9. `signals/composite.py` — Fix base_strength divisor

**Lines ~40-50 (in `compute_composite()`):**

```python
# BEFORE:
if long_score > short_score:
    direction = "long"
    base_strength = long_score / active_count
elif short_score > long_score:
    direction = "short"
    base_strength = short_score / active_count

# AFTER:
if long_score > short_score:
    direction = "long"
    long_count = sum(1 for s in tier1_signals if s["direction"] == "long")
    base_strength = long_score / max(long_count, 1)
elif short_score > long_score:
    direction = "short"
    short_count = sum(1 for s in tier1_signals if s["direction"] == "short")
    base_strength = short_score / max(short_count, 1)
```

---

### 10. `signals/tier1.py` — Fix eod_event_reversal conflicts

**Lines ~165-175 (in `eod_event_reversal()`):**

```python
# BEFORE:
surprise_direction = None
for event in events:
    sd = event.get("surprise_direction", "")
    if sd and sd != "neutral":
        surprise_direction = sd
        break  # ← STOPS AT FIRST

# AFTER:
# Aggregate all surprises by counting USD-positive vs USD-negative
positive_usd_count = 0
negative_usd_count = 0
for event in events:
    sd = event.get("surprise_direction", "")
    if "positive_usd" in sd:
        positive_usd_count += 1
    elif "negative_usd" in sd:
        negative_usd_count += 1

# Determine net surprise direction
if positive_usd_count > negative_usd_count:
    surprise_direction = "positive_usd"
elif negative_usd_count > positive_usd_count:
    surprise_direction = "negative_usd"
else:
    surprise_direction = None  # conflicting or neutral
```

---

### 11. `daily_loop.py` — Fix cron time comment

**Line 6:**

```python
# BEFORE:
Triggered by cron at 4:30 PM EST, Monday–Friday.

# AFTER:
Triggered by cron at 5:15 PM EST, Monday–Friday (after Forex day close at 5 PM).
```

---

## 📝 Recommendations to Implement Later

1. **Add DE 2Y Bund yield fetcher** — Currently `fred_yields.py` has a fallback for missing DE data. Add a real FRED series or ECB API endpoint.

2. **Add 4H bar features** — `daily_loop.py` fetches 4H bars but `features/technical.py` never uses them. Add intraday momentum/divergence features.

3. **Calibrate sentiment thresholds** — 72%/28% for EUR/USD may be wrong. Run historical analysis to find actual extremes.

4. **Add feature normalization** — Before XGBoost training, normalize or standardize features so `yield_spread_bps` (50-200) and `is_friday` (0-1) are on similar scales.

5. **Create `backtest_loop.py`** — Run `daily_loop` logic over stored historical data to generate full equity curve before going live.

6. **Add `utils/__init__.py`** — Make `utils` a proper package:
```python
# utils/__init__.py
from .trading_calendar import next_trading_day, is_trading_day

__all__ = ["next_trading_day", "is_trading_day"]
```

---

## ✅ Testing Checklist

After applying fixes:

1. ☐ Run `python -m models.hmm_regime` — verify state-map logging
2. ☐ Run `python -m fetchers.fred_yields` — verify new spread_change columns
3. ☐ Run `python -m models.meta_model` — verify CPCV doesn't crash
4. ☐ Check `predictions` table — verify `prediction_for` is T+1, not T+0
5. ☐ Run full `python daily_loop.py` — verify no exceptions

---

**All fixes documented. Apply them sequentially and test each module independently before running the full pipeline.**
