# Quantitative Signal & Feature Engineering Audit Report

**Date:** May 29, 2026  
**Auditor:** Antigravity (Quantitative Signal Engineer)  
**Target:** Quant EOD Engine — Signal Generation & Feature Engineering  
**Status:** Completed  

---

## Executive Summary

This signal and feature engineering audit evaluates the `quant-eod-engine` for mathematical validity, feature-to-model consistency, signal leakage, and threshold calibration.

**Overall Assessment:** The signal layer contains a mixture of severe architectural bugs and indicator miscalibrations. Most critically, a mismatch in dictionary wrapping between the daily pipeline (`daily_loop.py`) and the feature vector assembler (`vector.py`) causes all signal-related features to be zeroed out in live trading. Additionally, the ATR indicator uses an incorrect smoothing factor that violates Wilder's smoothing methodology, and storing Tier 2 signal states causes database query execution failures due to a string length truncation bug.

A summary table of vulnerabilities, rated by severity, is provided at the end of this report.

---

## 1. Live Vector Assembly Signal Mismatch (CRITICAL)

### Findings & Analysis
In [vector.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/features/vector.py), the signal-related features (`primary_signal_direction`, `primary_signal_count`, `composite_strength`, `tier2_confirmation_count`, `eod_event_reversal`, `event_surprise_magnitude`) are fetched using nested gets:
```python
"eod_event_reversal": signals_summary.get("composite", {}).get("eod_event_reversal", 0),
"composite_strength": signals_summary.get("composite", {}).get("composite_strength", 0.0),
...
```
During training and backtesting, `signals_summary` is correctly packaged as a dictionary wrapping the composite dictionary under the `"composite"` key:
```python
signals_summary = {"tier1": tier1_signals, "tier2": tier2_signals, "composite": composite}
```
However, in the live production pipeline [daily_loop.py:L333-L335](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L333-L335), the direct output of `compute_composite()` (which is a flat dictionary) is passed as `signals_summary`:
```python
feature_vector = assemble_feature_vector(
    today, PRIMARY_INSTRUMENT, technical_result, regime_result, composite_result
)
```
Inside `assemble_feature_vector`, `signals_summary` points to `composite_result`. The expression `signals_summary.get("composite", {})` looks for a key named `"composite"` inside `composite_result`. Since `composite_result` is flat and has no such key, it returns the default `{}`.
As a result:
* All 6 signal-related features default to `0.0` or `0` in live trading.
* The meta-model predicts on a feature vector where the primary signal direction, strength, and confirmations are completely masked.
This makes live predictions garbage and completely invalidates the meta-model's filtering capability in production.

### Code References
* Broken nested gets: [vector.py:L91-L92](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/features/vector.py#L91-L92), [vector.py:L103-L106](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/features/vector.py#L103-L106)
* Flat composite passed in daily loop: [daily_loop.py:L333-L335](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py#L333-L335)

### Severity: CRITICAL
### Recommendations
* In [daily_loop.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/daily_loop.py), wrap `composite_result` in a dictionary under the `"composite"` key before passing it to `assemble_feature_vector`:
  ```python
  signals_summary = {
      "composite": composite_result,
      "tier1": tier1_signals,
      "tier2": tier2_signals
  }
  ```

---

## 2. Database Value Truncation on Signal Storage (HIGH)

### Findings & Analysis
In [tier2.py:L191](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/tier2.py#L191), the `direction` field for Tier 2 signals is explicitly set to `None` since Tier 2 signals only confirm or deny direction:
```python
"direction": None
```
In `store_signals()` in [composite.py:L134](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/composite.py#L134), the database query handles a missing `direction` value using a fall-through expression:
```python
sig.get("direction") or ("confirmed" if sig.get("confirmed") else "not_confirmed")
```
Since `sig.get("direction")` is `None`, this falls through to `"confirmed"` or `"not_confirmed"`.
However, the `signals` table in [schema_phase2.sql](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_phase2.sql#L33) defines `direction` as `VARCHAR(10)`:
```sql
direction VARCHAR(10),
```
The string `"not_confirmed"` has **13 characters**. Attempting to insert a 13-character string into a `VARCHAR(10)` column will cause PostgreSQL to abort the transaction with a `value too long for type character varying(10)` database error.
This will crash the daily loop execution during the signal storage step whenever a Tier 2 signal is not confirmed.

### Code References
* Direction fall-through: [composite.py:L134](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/composite.py#L134)
* Schema column constraint: [schema_phase2.sql:L33](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/sql/schema_phase2.sql#L33)

### Severity: HIGH
### Recommendations
* Change the default string values for Tier 2 confirmations to fit within the `VARCHAR(10)` constraint (e.g., use `"yes"` and `"no"`, or `"confirm"` and `"deny"`).
* Alternatively, migrate the database column to a larger type (e.g., `VARCHAR(20)`).

---

## 3. Incorrect Wilder's Smoothing in ATR Indicator (HIGH)

### Findings & Analysis
The ATR calculation in [technical.py:L35](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/features/technical.py#L35) uses a standard Exponential Moving Average (EMA) with `span=period`:
```python
return true_range.ewm(span=period, adjust=False).mean()
```
Wilder's smoothing (which is the industry standard for calculating ATR and RSI) uses a smoothing factor of $\alpha = 1 / N$, where $N$ is the period (14).
A standard pandas `ewm` with `span=period` uses a smoothing factor of:
$$\alpha_{ema} = \frac{2}{period + 1}$$
For a 14-period ATR, this results in $\alpha = 2/15 \approx 0.133$, which corresponds to a 7-period Wilder's smoothing!
To achieve a proper 14-period Wilder's smoothing ($\alpha = 1/14 \approx 0.0714$), the formula must use `com = period - 1` or `span = 2 * period - 1`:
$$\alpha_{wilder} = \frac{1}{period} = \frac{2}{(2 \times period - 1) + 1}$$
By using `span=14`, the calculated ATR is far too reactive to recent price changes, leading to unstable position sizing calculations in `RiskManager` and incorrect regime Z-clipping in the physics engine.

### Code References
* ATR calculation: [technical.py:L35](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/features/technical.py#L35)

### Severity: HIGH
### Recommendations
* Modify [technical.py](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/features/technical.py#L35) to use Wilder's smoothing via the `com` parameter:
  ```python
  return true_range.ewm(com=period - 1, adjust=False).mean()
  ```

---

## 4. German government bond yield curve mismatch (HIGH)

### Findings & Analysis
In [settings.py:L50](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/config/settings.py#L50), `FRED_DE_2Y_SERIES` is configured to use the FRED series `IRLTLT01DEM156N`:
```python
FRED_DE_2Y_SERIES = "IRLTLT01DEM156N"
```
`IRLTLT01DEM156N` represents the **10-Year (long-term) government bond yield** for Germany, whereas `DGS2` (the US series) represents the **2-Year Treasury yield**.
The yield spread calculation `yield_spread_bps` is computed as:
$$spread = \text{US 2Y Yield} - \text{German 10Y Yield}$$
This is a yield curve mismatch. Short-term (2Y) interest rates are highly sensitive to central bank monetary policy shifts, while long-term (10Y) rates are driven by structural growth and inflation expectations. Subtracting a 10-year yield from a 2-year yield produces a hybrid spread that distorts the monetary policy differential signal, leading to false momentum triggers in `yield_spread_momentum`.

### Code References
* FRED configuration: [settings.py:L49-L50](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/config/settings.py#L49-L50)

### Severity: HIGH
### Recommendations
* Source actual 2-year German government bond yields (such as the ECB Statistical Data Warehouse or Trading Economics API) instead of using the German 10Y yield as a proxy.

---

## 5. Asymmetric Composite Signal Bias (MEDIUM)

### Findings & Analysis
In [composite.py:L77-L86](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/composite.py#L77-L86), the Tier 2 confirmation adjustments are asymmetric:
* Each confirmation adds `+0.05` to the composite strength.
* Each non-confirmation subtracts `-0.02` from the composite strength.
With 4 Tier 2 signals, the maximum boost is `+0.20`, while the maximum penalty is only `-0.08`.
This creates a structural bias toward inflating signal strength, encouraging trade-taking. For instance, if a primary signal has a weak base strength of `0.11` (below the `0.15` entry gate) and receives 2 confirmations and 2 non-confirmations, the net strength is boosted to:
$$strength = 0.11 + 2(0.05) - 2(0.02) = 0.17$$
This pushes the signal over the `0.15` trading threshold. The asymmetry systematically overweights confirmations, making the entry filter less selective.

### Code References
* Asymmetric adjustment: [composite.py:L77-L86](file:///C:/Users/angel/.antigravity-ide/Repos/quant-eod-engine/signals/composite.py#L77-L86)

### Severity: MEDIUM
### Recommendations
* Standardize the adjustments to be symmetric (e.g., `+0.05` for confirmation, `-0.05` for non-confirmation), or calibrate the offsets based on historical backtesting to ensure they do not introduce an arbitrary upward drift.

---

## Summary of Findings

| ID | Finding Title | Severity | Files Affected | Status |
|---|---|---|---|---|
| F-1 | Live Vector Assembly Signal Mismatch | **CRITICAL** | `daily_loop.py`, `features/vector.py` | Open |
| F-2 | Database Value Truncation on Signal Storage | **HIGH** | `signals/composite.py`, `sql/schema_phase2.sql` | Open |
| F-3 | Incorrect Wilder's Smoothing in ATR Indicator | **HIGH** | `features/technical.py` | Open |
| F-4 | German government bond yield curve mismatch | **HIGH** | `config/settings.py` | Open |
| F-5 | Asymmetric Composite Signal Bias | **MEDIUM** | `signals/composite.py` | Open |
