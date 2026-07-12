# Quantitative Strategy Audit Report — Quant EOD Engine

**Auditor:** Quantitative Finance Auditor
**Audit Date:** May 29, 2026
**Target Codebase:** Quant EOD Engine (EUR/USD)
**Scope of Audit:**
- [models/meta_model.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py) (XGBoost meta-model, CPCV, PSR, label generation)
- [models/hmm_regime.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/hmm_regime.py) (HMM regime detection)
- [walkforward_test.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/walkforward_test.py) (walk-forward backtest methodology)
- [backtest_loop.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/backtest_loop.py) (backtesting engine)
- [generate_historical_features.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/generate_historical_features.py) (historical feature generation)
- [docs/math_model_spec.md](file:///C:/.antigravity-ide/Repos/quant-eod-engine/docs/math_model_spec.md) and [docs/backtest_report.md](file:///C:/.antigravity-ide/Repos/quant-eod-engine/docs/backtest_report.md) (prior documentation)

---

## Executive Summary of Findings

This audit has identified several critical statistical and methodological flaws in the **Quant EOD Engine**'s strategy design, cross-validation, and backtest implementations. Most notably, a mismatch between training labels (open-to-close) and simulated returns (close-to-close) creates a severe **Look-Ahead Return Signing Bias** in CPCV validation, artificially inflating performance. Additionally, the **Probabilistic Sharpe Ratio (PSR)** implementation contains a fundamental mathematical error in its variance calculation due to incorrect scaling of the annualized Sharpe ratio, rendering the PSR metric invalid.

Below is a summary of findings categorized by severity:

| ID | Finding Title | Severity | Files Affected | Status |
|---|---|---|---|---|
| F-1 | Look-Ahead Return Signing Bias in CPCV Validation | **CRITICAL** | `models/meta_model.py` | Open |
| F-2 | Mathematical Scaling Error in Probabilistic Sharpe Ratio (PSR) | **CRITICAL** | `models/meta_model.py` | Open |
| F-3 | Look-Ahead Feature Leakage via Future H4 Bars in Backfill | **HIGH** | `generate_historical_features.py` | Open |
| F-4 | Overfitting via In-Sample Loading / Static Backtesting | **HIGH** | `walkforward_test.py` | Open |
| F-5 | Violations of IID Assumptions in CPCV Path Significance Testing | **MEDIUM** | `models/meta_model.py` | Open |
| F-6 | HMM Regime Instability and Downstream Feature Corruption | **MEDIUM** | `models/hmm_regime.py` | Open |
| F-7 | Performance Bottleneck via Repeated GPU Probing in CPCV Loop | **LOW** | `models/meta_model.py` | Open |

---

## Detailed Findings

### F-1: Look-Ahead Return Signing Bias in CPCV Validation (CRITICAL)

#### Description
In [models/meta_model.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py), there is a critical misalignment between how training labels are generated and how returns are simulated during Purged Combinatorial Cross-Validation (CPCV):
1. **Label Generation (Open-to-Close):** In `train_from_db()` (lines 159-161), the target label is generated based on the **Open-to-Close return of T+1**:
   $$ret = \frac{Close_{T+1}}{Open_{T+1}} - 1.0$$
   The binary label is $1$ if this intraday return is positive (for a long signal) and $0$ if it is negative.
2. **CPCV Simulated Return (Close-to-Close):** In `_run_cpcv()`, the return simulation fetches Close-to-Close returns using `_next_trading_day_pct_returns()` (lines 644-672):
   $$r\_next_T = \frac{Close_{T+1}}{Close_T} - 1.0$$
3. **Simulation Return Formulation:** On line 469, daily returns for a fold are computed as:
   ```python
   daily_returns = signals * (2 * y_test - 1) * np.abs(r_next)
   ```
   Because `y_test` is generated from the **Open-to-Close** return series but signed onto the magnitude of the **Close-to-Close** return series, the simulation creates synthetic returns that never occurred.
   For example, if the primary signal is Long:
   - On day T+1, the market gaps down significantly but rallies slightly intraday: Open-to-Close is $+0.05\%$, but Close-to-Close is $-2.0\%$.
   - The label `y_test` will be `1` (profitable trade based on Open-to-Close).
   - The CPCV simulation computes `daily_returns = 1 * (2*1 - 1) * |-2.0%| = +2.0%`.
   - The strategy records a $+2.0\%$ profit, when in reality entering at Open and exiting at Close would have yielded $+0.05\%$, and entering at Close of T would have yielded $-2.0\%$.

This methodology leaks the correctness of the intraday direction to sign the magnitude of the Close-to-Close return, generating massive artificial profits and invalidating the CPCV validation metrics.

#### Line References
- [models/meta_model.py:L159-L161](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L159-L161)
- [models/meta_model.py:L465-L470](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L465-L470)
- [models/meta_model.py:L644-L672](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L644-L672)

#### Concrete Fix Suggestion
Align the returns used in CPCV simulation with the execution logic of the strategy. If the strategy executes Open-to-Close on T+1, pre-calculate a vector of actual realized Open-to-Close trade returns instead of mixing labels and absolute returns:
1. Modify `_next_trading_day_pct_returns` to return the realized Open-to-Close return on T+1 for each sample date:
   ```python
   out[i] = (open_prices[nd] / close_prices[nd]) - 1.0 # Or close / open depending on asset convention
   ```
2. Simplify the CPCV return calculation to use the actual trade returns directly:
   ```python
   # Pre-calculate actual trade returns: primary_signal_direction * realized_return_t+1
   trade_returns = primary_signals * realized_open_to_close_returns
   ...
   # Inside CPCV fold loop:
   daily_returns = signals * trade_returns[test_idx]
   ```

---

### F-2: Mathematical Scaling Error in Probabilistic Sharpe Ratio (PSR) (CRITICAL)

#### Description
The implementation of `_probabilistic_sharpe_ratio_from_returns()` in [models/meta_model.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py) (lines 675-701) contains a fundamental scaling error:
1. **Annualized Sharpe Input:** The function computes `sr` as the **annualized** Sharpe ratio (line 689):
   ```python
   sr = (m / s) * np.sqrt(252.0)
   ```
2. **Variance of Sharpe Formula:** It then plugs `sr` directly into the Bailey & López de Prado variance formula (lines 692-697):
   ```python
   var_sr = (
       1.0
       + 0.5 * sr ** 2
       - skew * sr
       + (kurt_excess / 4.0) * sr ** 2
   ) / max(T - 1, 1)
   ```
The Bailey & López de Prado variance formula is strictly derived for the **non-annualized (daily) Sharpe ratio** ($SR_d$). If the annualized Sharpe ratio ($SR_a = SR_d \sqrt{252}$) is plugged in directly, the quadratic terms $0.5 SR_a^2$ and $\frac{\gamma_4}{4} SR_a^2$ scale up by $252 \approx 252$ times, while the constant term remains $1.0$ and the skewness term scales by $\sqrt{252}$. 

This dimensional inconsistency causes the variance to be highly over-estimated. For large values of $SR_a$, the variance becomes dominated by the quadratic terms:
$$Var(SR_a) \approx \frac{SR_a^2}{T-1} (0.5 + 0.25 \gamma_4^{excess})$$
Which leads to the z-statistic:
$$z = \frac{SR_a}{\sqrt{Var(SR_a)}} \approx \frac{\sqrt{T-1}}{\sqrt{0.5 + 0.25 \gamma_4^{excess}}}$$
Consequently, the z-statistic and the resulting PSR become entirely independent of the Sharpe ratio itself, functioning merely as a lookup of sample size and kurtosis. This renders the reported PSR metric mathematically invalid.

#### Line References
- [models/meta_model.py:L689-L700](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L689-L700)

#### Concrete Fix Suggestion
Calculate the daily Sharpe ratio, compute the daily variance, compute the z-statistic, and only scale to annual metrics if necessary:
```python
def _probabilistic_sharpe_ratio_from_returns(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3:
        return 0.5
    m = np.mean(r)
    s = np.std(r, ddof=1)
    if s <= 0:
        return 0.5
    
    # 1. Use DAILY (non-annualized) Sharpe ratio in variance calculation
    sr_daily = m / s
    
    # 2. Compute skewness and excess kurtosis
    skew = float(stats.skew(r, bias=False))
    kurt_excess = float(stats.kurtosis(r, fisher=True, bias=False))
    
    # 3. Correct variance of daily Sharpe ratio (incorporating excess kurtosis)
    var_sr_daily = (
        1.0
        + 1.0 * sr_daily ** 2  # Adjusted coefficient when using excess kurtosis
        - skew * sr_daily
        + (kurt_excess / 4.0) * sr_daily ** 2
    ) / (T - 1)
    
    var_sr_daily = max(var_sr_daily, 1e-12)
    
    # 4. Compute z-statistic
    z = sr_daily / np.sqrt(var_sr_daily)
    return float(stats.norm.cdf(z))
```

---

### F-3: Look-Ahead Feature Leakage via Future H4 Bars in Backfill (HIGH)

#### Description
In [generate_historical_features.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/generate_historical_features.py), the function `_load_h4_bars()` loads H4 bars for feature vector assembly using the following query (lines 120-136):
```sql
SELECT bar_time, open, high, low, close, volume
FROM bars
WHERE instrument = %s
  AND granularity = 'H4'
  AND complete = TRUE
  AND date(bar_time) BETWEEN %s AND %s
```
Where the start and end dates are `start_date` and `run_date` (which is trade date T).
- In OANDA, H4 bars are timestamped in UTC.
- For a trade date `run_date` $T$, the H4 bar starting at 5:00 PM NY time corresponds to 9:00 PM UTC, so its `date(bar_time)` is $T$.
- This bar ends at 9:00 PM NY time (which is in the future relative to the 5:15 PM NY execution cutoff).
- When running `generate_historical_features.py` historically, this bar is already complete in the database. Because `date(bar_time)` is $T$, the query fetches this bar and includes it in the technical indicator calculations (e.g., computing H4-based features for the EOD vector).
This means the historical feature vectors are trained on price data that occurred between 5:00 PM and 9:00 PM NY time, which would not be available in live trading at 5:15 PM NY. This look-ahead leakage is absent in live prediction, causing a distribution shift and overstating backtest performance.

#### Line References
- [generate_historical_features.py:L120-L136](file:///C:/.antigravity-ide/Repos/quant-eod-engine/generate_historical_features.py#L120-L136)

#### Concrete Fix Suggestion
Implement a strict datetime cutoff at 5:00 PM NY time for the H4 query, matching the logic used in the walk-forward script:
```python
def _load_h4_bars(instrument: str, run_date, lookback_days: int = 30) -> pd.DataFrame:
    start_date = run_date - timedelta(days=lookback_days)
    # Cutoff at exactly 5:00 PM NY time on the run date
    cutoff_dt = datetime.combine(run_date, time(17, 0), tzinfo=ZoneInfo("America/New_York"))
    rows = fetch_all(
        """
        SELECT bar_time, open, high, low, close, volume
        FROM bars
        WHERE instrument = %s
          AND granularity = 'H4'
          AND complete = TRUE
          AND bar_time BETWEEN %s AND %s
        ORDER BY bar_time ASC
        """,
        (instrument, start_date, cutoff_dt),
    )
    df = pd.DataFrame(rows, columns=["bar_time", "open", "high", "low", "close", "volume"])
    return _decimals_to_floats(df)
```

---

### F-4: Overfitting via In-Sample Loading / Static Backtesting (HIGH)

#### Description
In [walkforward_test.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/walkforward_test.py), the backtest logic loaded to simulate walk-forward performance has two major issues:
1. **In-Sample Loading:** By default, if `force_train` is `False`, the script attempts to load a pre-existing model from the database using `model._load_model()` (lines 420-435). If the model stored in the database was trained on the entire dataset (which is the case after running `train_from_db` without date parameters), the backtest will run **in-sample** over the evaluation period, inflating metrics.
2. **Static Model Evaluation:** If `force_train` is `True` or a model is not found, the script trains a model *once* up to `start_date - 1` (line 430), and then evaluates it statically over the entire out-of-sample period (e.g. 1 year). In quantitative finance, a walk-forward test must dynamically retrain the model on a rolling or expanding window (e.g., every 30 days) to reflect how the model would be updated in production. Keeping the XGBoost model static over 1 year does not represent a true walk-forward backtest.

#### Line References
- [walkforward_test.py:L415-L436](file:///C:/.antigravity-ide/Repos/quant-eod-engine/walkforward_test.py#L415-L436)

#### Concrete Fix Suggestion
1. **Prevent In-Sample Loading:** Force training in the backtest script by default or restrict database model loading to models whose training window ends before the backtest start date.
2. **Implement Retraining Loop:** Add periodic retraining (e.g., every 30 days) of the XGBoost meta-model inside the walk-forward loop to update the model with the latest historical feature vectors, similar to the HMM refitting interval.

---

### F-5: Violations of IID Assumptions in CPCV Path Significance Testing (MEDIUM)

#### Description
In `_run_cpcv()` in [models/meta_model.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py), a one-sample t-test is performed on the Sharpe ratios of the 15 CPCV paths to check for statistical significance (lines 488-498):
```python
t_res = stats.ttest_1samp(sharpe_array, 0.0, alternative="greater")
```
The one-sample t-test assumes that the observations are Independent and Identically Distributed (IID). However, the 15 paths in CPCV are constructed by combining overlapping test sets from the same historical series. Because the paths share a high degree of mutual information and are highly dependent, the standard error of the mean is severely underestimated, and the t-statistic is artificially inflated. This yields an artificially low p-value (often $< 0.0001$, as reported in `backtest_report.md` line 399), giving a false indication of statistical significance.

#### Line References
- [models/meta_model.py:L488-L498](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L488-L498)

#### Concrete Fix Suggestion
1. Replace the parametric t-test on Sharpe ratios with a non-parametric bootstrap test.
2. Alternatively, remove the t-test from the path validation and instead evaluate significance based on the pooled returns using a block-bootstrap or the PSR metric.

---

### F-6: HMM Regime Instability and Downstream Feature Corruption (MEDIUM)

#### Description
The HMM in [models/hmm_regime.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/hmm_regime.py) is re-fitted from scratch every 30 days on a rolling 504-day window. Hidden Markov Models fitted via the Baum-Welch (EM) algorithm are highly sensitive to initialization and minor changes in data. Re-fitting the HMM from scratch causes the estimated parameters and the resulting Viterbi state path to shift, meaning the classification of past days can change retroactively.
Furthermore, although the states are mapped to semantic labels (0, 1, 2) by sorting on mean volatility, if the volatility profiles of the states are close, the ordering of the HMM states can swap (a label flip). If a flip warning is logged but ignored, the semantic features (`regime_state` and `days_in_regime`) fed into the XGBoost meta-model will change meanings overnight, corrupting the feature representation and degrading downstream model performance.

#### Line References
- [models/hmm_regime.py:L92-L129](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/hmm_regime.py#L92-L129)

#### Concrete Fix Suggestion
1. **Fixed Parameter Model:** Rather than re-fitting the HMM every 30 days, train the HMM once on a long historical period, freeze its parameters, and use this fixed model to infer states. Only retrain the model offline under strict validation (e.g. quarterly).
2. **Anchor-Based Alignment:** If dynamic refitting is necessary, align the new HMM states with the previous model's states using KL divergence of their emission distributions against the old parameters, rather than just sorting by mean volatility.

---

### F-7: Performance Bottleneck via Repeated GPU Probing in CPCV Loop (LOW)

#### Description
In [models/meta_model.py](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py), the function `_get_xgb_device()` is called inside the CPCV fold loop for every single combination (line 458):
```python
device=_get_xgb_device(),
```
`_get_xgb_device()` performs a full GPU probe by instantiating a dummy `XGBClassifier`, transferring data to CUDA, and training it on a 1x1 array. For 15 folds, this probe is executed 15 separate times, causing substantial overhead due to repeated CUDA context initialization, memory allocation, and CPU-GPU serialization.

#### Line References
- [models/meta_model.py:L458](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L458)
- [models/meta_model.py:L66-L81](file:///C:/.antigravity-ide/Repos/quant-eod-engine/models/meta_model.py#L66-L81)

#### Concrete Fix Suggestion
Cache the detected device at the module level or decorate `_get_xgb_device` with `@functools.lru_cache()` so that the GPU is only probed once:
```python
import functools

@functools.lru_cache(maxsize=1)
def _get_xgb_device() -> str:
    # Existing GPU probing logic...
```

---

## Conclusion

The quantitative strategy contains structural errors in return calculation (F-1) and statistical formulas (F-2) that compromise the validity of the reported backtest results. Correcting these flaws will likely lower the reported Sharpe and PSR statistics, but will provide a methodologically sound framework for strategy validation and live execution. Implementing the suggested fixes is highly recommended before deploying capital to the live environment.
