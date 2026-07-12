# Quant EOD Engine — Compliance & Ethics Report

> **Scope:** Data sourcing compliance, bias audit, and ethical framework
> **Last Updated:** 2026-04-03
> **Version:** v1.0.0 (aligned with Change Log)

---

## Table of Contents

1. [Data Source Inventory & Compliance](#1-data-source-inventory--compliance)
2. [Bias Audit](#2-bias-audit)
3. [Ethical Framework](#3-ethical-framework)
4. [Remediation Recommendations](#4-remediation-recommendations)
5. [Compliance Checklist](#5-compliance-checklist)

---

## 1. Data Source Inventory & Compliance

### 1.1 Complete Data Source Registry

| # | Source | API | Data Type | License / Terms | Rate Limits | Code Reference |
|:-:|--------|-----|-----------|-----------------|:-----------:|:--------------:|
| 1 | **OANDA V20** | REST `/v3/instruments/{}/candles` | D1 + H4 OHLCV bars | OANDA API Terms of Service; requires funded/demo account | 120 req/sec | [oanda_bars.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/oanda_bars.py) |
| 2 | **OANDA V20** | REST `/v3/instruments/{}/positionBook` | Retail sentiment (% long/short) | Same as above | Same | [oanda_sentiment.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/oanda_sentiment.py) |
| 3 | **OANDA V20** | REST (swap endpoint) | Overnight financing rates | Same as above | Same | [swap_rates.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/swap_rates.py) |
| 4 | **FRED** (Federal Reserve Economic Data) | REST via `fredapi` library | US 2Y (`DGS2`) + DE 2Y (`IRLTLT01DEM156N`) yields | Public domain (US Gov data); API key required; no redistribution restrictions | 120 req/min | [fred_yields.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/fred_yields.py) |
| 5 | **Perplexity AI** | REST chat completion | Macro sentiment analysis, central bank stance, risk assessment | Perplexity API Terms; outputs are AI-generated (non-deterministic) | Varies by plan | [perplexity_sentiment.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/perplexity_sentiment.py) |
| 6 | **Perplexity AI** | REST chat completion (via calendar stub) | Economic calendar events | Same as above | Same | [calendar.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/calendar.py) |
| 7 | **PostgreSQL** | Local/remote DB | Persistent storage for all pipeline data | Self-hosted; no third-party data licensing concerns | N/A | [database.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/database.py) |
| 8 | **Discord** | Webhook POST | Signal delivery & error alerts | Discord Developer Terms; webhook-only (no bot) | 30 req/min per webhook | [discord_notify.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/discord_notify.py) |

### 1.2 Data Integrity Characteristics

| Source | Point-in-Time? | Revised After Publication? | Survivorship-Free? | Complete History? |
|--------|:--------------:|:--------------------------:|:-------------------:|:-----------------:|
| OANDA bars | ✅ Yes — bars are timestamped and immutable once `complete=True` | ❌ No — OANDA does not revise mid prices after publication | ✅ Yes — EUR/USD has no listing/delisting events | ✅ Continuous since 2005 |
| OANDA sentiment | ⚠️ Partial — snapshot at query time, not end-of-day | N/A — real-time snapshot | ✅ N/A (aggregate ratio) | ❌ No historical backfill available |
| FRED yields | ⚠️ Caution — FRED revises some economic series; bond yields are generally not revised, but publication time is T+1 (next business day) | ⚠️ Possible for DE series | ✅ Yes | ✅ Continuous since 1976 (US), limited for DE |
| Perplexity AI | ❌ No — LLM outputs are non-reproducible; same prompt may produce different results on different days | N/A | N/A | ❌ No historical data; cannot be backtested |

> [!WARNING]
> **FRED T+1 Publication Lag:** FRED publishes daily yield data with a 1-business-day delay. The system queries `date.today()`, but the most recent observation may be from yesterday. The code handles this by using `iloc[-1]` (the latest available), but this means the yield spread features lag the current market by ~1 day. This is **not** look-ahead bias (it's information the system would have had in real-time), but it does reduce the signal's timeliness. The code correctly uses the lagged-but-available data.

> [!CAUTION]
> **Perplexity AI Non-Reproducibility:** The AI macro sentiment signal cannot be backtested because LLM outputs are non-deterministic. In the backtest, this feature would need to be either excluded or populated from stored historical API responses. Any backtest results that include non-zero `macro_sentiment_score` or `ai_confidence` values represent **a different signal** than what would have been generated in real-time.

---

## 2. Bias Audit

### 2.1 Bias Summary Matrix

| Bias Type | Severity | Status | Detail |
|-----------|:--------:|:------:|--------|
| [Look-Ahead Bias](#22-look-ahead-bias) | 🟡 Medium | ⚠️ 2 instances found | Scaler fit, HMM training window edge |
| [Survivorship Bias](#23-survivorship-bias) | 🟢 Low | ✅ Minimal risk | Single continuously-traded instrument |
| [Selection Bias](#24-selection-bias) | 🟡 Medium | ⚠️ Present | Strategy designed and tuned on EUR/USD only |
| [Overfitting Bias](#25-overfitting-bias) | 🟢 Low | ✅ Mitigated | CPCV + PSR + regularization |
| [Data Snooping Bias](#26-data-snooping-bias) | 🟡 Medium | ⚠️ Unaddressed | Multiple signals tested, no correction |
| [Recency Bias](#27-recency-bias) | 🟢 Low | ✅ Controlled | HMM window is 504 days (~2 years) |
| [Execution Bias](#28-execution-bias) | 🟡 Medium | ⚠️ Partially addressed | Fixed spread assumption |
| [Confirmation Bias](#29-confirmation-bias) | 🟡 Medium | ⚠️ Structural | Asymmetric T2 scoring |

---

### 2.2 Look-Ahead Bias

Look-ahead bias occurs when a model uses information at time $t$ that would not have been available until time $t + k$.

#### Finding LA-01: StandardScaler Fit on Full Dataset (MEDIUM)

**Location:** [meta_model.py:L123–L124](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L123-L124)

```python
self.scaler = StandardScaler()
X = self.scaler.fit_transform(X_raw)  # Fit on ALL training data
```

**Issue:** The production StandardScaler is fit on the **entire** training dataset before the model is trained. This means the z-score normalization for any historical sample includes information from the sample's future (the mean and standard deviation of all features computed across the full time series).

**Mitigation already in place:** Inside `_run_cpcv()` (line 325–329), each fold fits its **own** StandardScaler on only the training split:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)  # Fit only on train fold
X_test = scaler.transform(X_test_raw)         # Transform test with train stats
```

**Verdict:** The CPCV validation is **correctly purged** — no look-ahead in the validation metrics. But the final production model's scaler has seen all data. This is **standard practice** for production models (since the production model will only predict future data, not historical), but it means in-sample training metrics (accuracy, F1) are slightly optimistic. The CPCV metrics are the trustworthy ones.

**Risk:** 🟡 **LOW-MEDIUM** — does not affect production predictions, only diagnostic metrics.

---

#### Finding LA-02: HMM Rolling Window Boundary (LOW)

**Location:** [hmm_regime.py:L79](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L79)

```python
df["vol_5d"] = df["log_return"].rolling(5).std()
```

**Issue:** The 5-day rolling standard deviation at any observation $t$ uses data from $\{t-4, t-3, t-2, t-1, t\}$. This is not look-ahead (it only uses past and present data), but the **HMM model itself** is fit on the entire 504-day window including all these vol estimates. The HMM's EM algorithm sees the full sequence when estimating transition probabilities and emission distributions.

**Mitigation:** HMM regime detection uses only historical bars that are already `complete=True` in the database. The HMM is re-fit daily using a rolling window, and the prediction uses forward-backward probabilities conditional on the full observed sequence. This is standard practice for HMMs — the model is fit on "known past" and used to classify the **current** state.

**Verdict:** The HMM's access to the full 504-day sequence during fitting means it "knows" the full trajectory of volatility when classifying any interior point. However, in production, we only care about the **last observation's classification**, which is analogous to online filtering. Early observations in the window contribute to parameter estimation but are not being predicted.

**Risk:** 🟢 **LOW** — standard HMM usage, negligible bias.

---

#### Finding LA-03: Feature Engineering Timing (CONFIRMED CLEAN)

**Location:** [oanda_bars.py:L48–L49](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/fetchers/oanda_bars.py#L48-L49)

```python
if not c.get("complete", False):
    continue  # skip incomplete (current) bar
```

**Verification:** The system explicitly filters for `complete=True` bars only. OANDA marks a daily bar as complete after the 5 PM ET rollover. The pipeline runs at 5:15–5:20 PM ET (per `daily_loop.py` docstring), ensuring:

1. The daily bar for Day T is complete at signal generation time
2. Features are computed from the completed Day T bar
3. The prediction targets Day T+1 (entry at T+1 Open, exit at T+1 Close)

**In the backtest** ([backtest_loop.py:L146–L149](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L146-L149)):

```python
entry_price = prices_nd["open"]   # T+1 Open
exit_price  = prices_nd["close"]  # T+1 Close
raw_ret = (exit_price / entry_price) - 1.0
```

The backtest correctly uses **next-day** Open→Close prices, not same-day Close prices.

**Verdict:** ✅ **CLEAN** — No look-ahead bias in feature engineering or backtest execution.

---

### 2.3 Survivorship Bias

Survivorship bias occurs when a backtest only includes assets that still exist today, ignoring delisted/failed assets.

**Assessment:**

- **Instrument:** EUR/USD — a single, continuously traded Forex pair since the Euro's adoption in 1999
- **No universe selection:** The system does not select from a pool of instruments; it trades only EUR/USD
- **No delisting risk:** Major Forex pairs do not delist (unlike individual stocks)
- **OANDA data:** Provides continuous mid-price data for EUR/USD without gaps

**Verdict:** 🟢 **LOW RISK** — Survivorship bias is structurally absent for a single-instrument Forex strategy on a major pair.

> [!NOTE]
> If the system is extended to trade **multiple currency pairs** (per KI-004 in the backlog), survivorship bias must be re-evaluated. Exotic pairs can become untradeable, and some crosses have been discontinued by brokers.

---

### 2.4 Selection Bias

Selection bias occurs when the strategy, thresholds, or features are designed based on knowledge of which instrument "works."

**Assessment:**

| Concern | Status | Detail |
|---------|:------:|--------|
| Instrument selection | 🟡 MEDIUM | EUR/USD was chosen — the most liquid Forex pair. This is a reasonable default, but no evidence of parallel testing on other pairs (GBP/USD, USD/JPY, etc.). |
| Threshold tuning | 🟡 MEDIUM | Yield spread thresholds (8/15/20 bps), sentiment extremes (72/28%), and probability gates (0.55/0.70) were set without documented cross-validation against alternative instruments. |
| Feature selection | 🟢 LOW | Features are macro-economically motivated (yields, sentiment, regime) rather than mined from data. |
| Time period selection | 🟡 MEDIUM | No documentation of which historical period the thresholds were optimized on. Risk that parameters fit a specific market regime. |

**Verdict:** 🟡 **MEDIUM RISK** — The strategy is architecturally sound, but thresholds lack multi-instrument or multi-period validation.

---

### 2.5 Overfitting Bias

Overfitting occurs when a model captures noise rather than signal, performing well in-sample but poorly out-of-sample.

**Mitigations in place:**

| Defense | Implementation | Effectiveness |
|---------|----------------|:-------------:|
| **CPCV validation** | 15 purged paths (N=6, k=2), 5-day purge, 2-day embargo | ✅ Strong — gold standard per de Prado |
| **PSR** | Bailey & López de Prado Probabilistic Sharpe Ratio with skew/kurtosis adjustment | ✅ Strong — accounts for non-normal returns |
| **XGBoost regularization** | L1=0.1, L2=1.0, min_child_weight=5, subsample=0.8, colsample=0.8, max_depth=4 | ✅ Strong — conservative tree complexity |
| **Primary signal veto** | Meta-model cannot override a flat primary signal | ✅ Strong — prevents ML-generated phantom trades |
| **Feature count** | 28 features, limited to theory-motivated inputs | ✅ Moderate — no kitchen-sink feature mining |

**Remaining concerns:**

| Concern | Severity | Detail |
|---------|:--------:|--------|
| CPCV uses weaker model than production | 🟡 Low | CPCV folds use 100 trees with default regularization; production uses 200 trees with stronger regularization. This makes CPCV estimates conservative — acceptable. |
| No holdout test set | 🟡 Medium | The entire dataset is used for CPCV + final model training. There is no strictly unseen holdout period. However, CPCV's purged combinatorial design is specifically meant to replace traditional holdout with a more statistically rigorous approach. |

**Verdict:** 🟢 **LOW RISK** — Overfitting defenses are state-of-the-art for this strategy class.

---

### 2.6 Data Snooping Bias

Data snooping (also called "multiple comparisons" or "p-hacking") occurs when many strategies or parameters are tested on the same data without adjusting for the number of tests.

**Assessment:**

| Concern | Severity | Detail |
|---------|:--------:|--------|
| **4 Tier 1 signals tested** | 🟡 Medium | Four alpha signals were developed and combined. Even though each is economically motivated, the selection of **these four** (and not others) introduces an implicit multiple-comparisons problem. None of the code includes a Bonferroni, Šidák, or FDR correction. |
| **Threshold optimization** | 🟡 Medium | Thresholds (yield: 8/15/20; sentiment: 72/28; composite floor: 0.15) were presumably selected via experimentation. No documentation of how many threshold values were tested before settling on these. |
| **PSR partially addresses this** | 🟢 Mitigating | The Probabilistic Sharpe Ratio accounts for the non-normality of returns, which partially hedges against overstated Sharpe ratios from data-mined strategies. However, PSR does not explicitly adjust for the number of strategies tested. |

**Verdict:** 🟡 **MEDIUM RISK** — Standard for research-stage quant systems, but lacks formal multiple-testing correction.

**Recommended correction:** Apply the Deflated Sharpe Ratio (DSR) from Bailey & López de Prado (2014), which adjusts the PSR for the total number of independent backtests conducted:

$$DSR = \Phi\left(\frac{(\widehat{SR} - SR_0) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4}{4}\widehat{SR}^2}}\right)$$

Where $SR_0 = \sqrt{\frac{V}{T-1}} \cdot \left[(1 - \gamma)\Phi^{-1}(1 - \frac{1}{N}) + \gamma\Phi^{-1}(1 - \frac{1}{N}\cdot e^{-1})\right]$ estimates the expected best Sharpe from $N$ independent trials under the null hypothesis.

---

### 2.7 Recency Bias

Recency bias gives disproportionate weight to recent data over historical patterns.

**Assessment:**

| Component | Window | Concern |
|-----------|:------:|---------|
| HMM | 504 trading days (~2 years) | Moderate window — captures recent regime but may miss structural shifts (e.g., post-GFC zero-rate era vs current rate cycle). Regime labeling is based on relative vol within this window, not absolute historical vol. |
| XGBoost | Trained on all available feature vectors | No explicit time-decay; older samples have equal weight to recent ones. This is appropriate for structural patterns but may be inappropriate if the alpha sources are evolving (e.g., retail sentiment behavior may differ in 2015 vs 2025). |
| Technical indicators | Standard lookbacks (14, 50, 200) | No recency bias — standard periods applied uniformly. |

**Verdict:** 🟢 **LOW RISK** — The 504-day HMM window is a reasonable compromise between stability and adaptiveness.

---

### 2.8 Execution Bias

Execution bias occurs when the backtest assumes trading conditions that don't reflect real-world execution.

**Assessment:**

| Assumption | Backtest Value | Real-World Condition | Bias Direction |
|------------|:--------------:|----------------------|:--------------:|
| **Spread** | Fixed 1.5 pips | Variable: 0.5–50+ pips depending on session, news, and liquidity | 🟡 Optimistic during stress |
| **Slippage** | None modeled | Can be 0.5–5 pips on market orders during volatility | 🟡 Optimistic |
| **Fill assumption** | 100% fill at Open price | Gaps and liquidity may prevent exact fill | 🟡 Slightly optimistic |
| **Execution timing** | T+1 Open | Achievable for EOD strategy with limit orders | 🟢 Realistic |
| **Leverage** | 10× fixed | Available from most Forex brokers for EUR/USD | 🟢 Realistic |
| **Margin / liquidation** | Not modeled | At 10× with no stop, a 10% adverse move triggers margin call | 🔴 Dangerous omission |

**Verdict:** 🟡 **MEDIUM RISK** — The fixed-spread assumption is the primary concern. During the historical stress events this system aims to trade through, actual spreads can be 10–50× wider than assumed.

---

### 2.9 Confirmation Bias (Structural)

Confirmation bias in system design occurs when the architecture systematically favors one outcome over the other.

**Finding CB-01: Asymmetric Tier 2 Scoring**

**Location:** [composite.py:L82–L85](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/composite.py#L82-L85)

```python
if conf.get("confirmed"):
    t2_adjustment += 0.05   # +0.05 for YES
else:
    t2_adjustment -= 0.02   # −0.02 for NO
```

**Issue:** A confirmation adds +0.05 but a non-confirmation only subtracts −0.02. With 4 Tier 2 checks:
- All 4 confirm: $+0.20$ boost
- All 4 non-confirm: $-0.08$ penalty
- 2 confirm, 2 don't: $+0.06$ (net positive)

This means the system is **structurally biased toward trading**. A signal with 50/50 Tier 2 confirmation still gets a net positive boost, making it more likely to pass the 0.15 composite floor.

**Verdict:** 🟡 **MEDIUM** — This is a design choice, not a bug, but it does encode a pro-trade bias. In practice, this means the system will trade more frequently than a symmetric design would suggest.

---

## 3. Ethical Framework

### 3.1 Market Manipulation Risk

| Question | Assessment |
|----------|------------|
| Does the system place orders that could manipulate prices? | ❌ No — the system generates **signals only** (delivered via Discord). It does not execute trades directly. |
| Does the system front-run public information? | ❌ No — all data sources are publicly available (OANDA, FRED, Perplexity). No access to private or privileged information. |
| Does the system engage in spoofing or layering? | ❌ No — no order management system is integrated. |
| Does the system trade on material non-public information (MNPI)? | ❌ No — all inputs are public data or AI-generated analysis. |

**Verdict:** 🟢 **NO MARKET MANIPULATION RISK** — signal-only system with no direct market access.

### 3.2 Data Usage Ethics

| Principle | Implementation |
|-----------|----------------|
| **Data sourced legally** | All APIs accessed with valid credentials under published terms of service. |
| **No personal data** | No individual trader data is collected — OANDA sentiment is aggregate ratios only. |
| **No scraping** | All data accessed through documented REST APIs, not web scraping. |
| **AI outputs disclosed** | The Perplexity AI sentiment is explicitly labeled as AI-generated in the feature vector (`macro_sentiment_score`, `ai_confidence`). |

### 3.3 Model Transparency

| Principle | Implementation |
|-----------|----------------|
| **Explainability** | SHAP values computed for every prediction; top 5 features reported with each signal. |
| **Audit trail** | Predictions stored in `predictions` table with probability, model version, and SHAP output. |
| **Version tracking** | Model version string (`xgb_v1_YYYY-MM-DD`) embedded in every prediction record. |
| **Reproducibility** | `random_state=42` fixed across XGBoost and HMM. Not fully reproducible due to LLM non-determinism (Perplexity). |

### 3.4 Human Oversight

| Control | Implementation |
|---------|----------------|
| **Human-in-the-loop** | Signals delivered to Discord — a human must decide whether to execute. No automated order execution. |
| **Primary signal veto** | The meta-model cannot create trades that the human-designed signal layer did not propose. |
| **Flat default** | When no model is available, the system defaults to `direction=flat, size=0.0`. |
| **Error alerting** | Pipeline failures trigger Discord error alerts for human review. |

---

## 4. Remediation Recommendations

### 4.1 Critical Remediations

| ID | Finding | Remediation | Effort |
|:--:|---------|-------------|:------:|
| R-01 | **No margin/liquidation modeling** (Execution Bias) | Implement equity drawdown circuit breaker in `backtest_loop.py`. If equity drops below 70% of peak, force flat. | 🟡 Medium |
| R-02 | **Fixed spread assumption** (Execution Bias) | Replace `SPREAD_BPS = 1.5` with regime-conditional spread: `{low_vol: 1.0, choppy: 2.5, crash: 10.0}`. | 🟢 Easy |
| R-03 | **No slippage model** (Execution Bias) | Add a random slippage component: `slippage = np.random.uniform(0, 0.5 * SPREAD_BPS) / 10000`. | 🟢 Easy |

### 4.2 Recommended Enhancements

| ID | Finding | Remediation | Effort |
|:--:|---------|-------------|:------:|
| R-04 | **No multiple-testing correction** (Data Snooping) | Implement Deflated Sharpe Ratio (DSR) alongside PSR. Document the total number of strategy variants tested. | 🟡 Medium |
| R-05 | **Asymmetric T2 scoring** (Confirmation Bias) | Consider symmetric adjustments (+0.03/−0.03) or adjust the composite floor upward to compensate. Test via OAT sweep. | 🟢 Easy |
| R-06 | **No out-of-sample instrument validation** (Selection Bias) | Run the strategy on GBP/USD and USD/JPY with identical parameters. If Sharpe degrades >50%, the strategy is EUR/USD-specific and thresholds should not be generalized. | 🔴 Hard |
| R-07 | **Perplexity non-reproducibility** (Data Integrity) | Store raw API responses in a `perplexity_responses` table with timestamp, prompt hash, and raw JSON. This creates a reproducible audit trail even if the LLM outputs change. | 🟡 Medium |

---

## 5. Compliance Checklist

### 5.1 Pre-Deployment Compliance Gate

```
DATA SOURCING
  [x] All API credentials stored in environment variables (not in code)
  [x] API terms of service reviewed for each data provider
  [x] No personal or private data collected
  [x] No web scraping; all data via documented APIs
  [x] OANDA bars filtered for complete=True only
  [x] FRED data accessed via official API with valid key

BIAS PREVENTION
  [x] Look-ahead bias: CPCV uses per-fold scaler (verified)
  [x] Look-ahead bias: Backtest uses T+1 Open→Close (verified)
  [x] Look-ahead bias: Only complete bars used in features (verified)
  [x] Survivorship bias: Single instrument, no universe selection (verified)
  [ ] Selection bias: Multi-instrument validation NOT YET DONE
  [x] Overfitting: CPCV + PSR implemented
  [ ] Data snooping: DSR correction NOT YET IMPLEMENTED
  [ ] Execution bias: Variable spread model NOT YET IMPLEMENTED

MODEL GOVERNANCE
  [x] SHAP explainability for every prediction
  [x] Predictions stored with model version in database
  [x] Human-in-the-loop via Discord (no auto-execution)
  [x] Primary signal veto prevents ML-only trades
  [x] Default behavior is flat (safe) when model unavailable
  [x] Change Log initialized with versioning convention

ETHICAL STANDARDS
  [x] No market manipulation capability (signal-only)
  [x] No front-running of private information
  [x] AI-generated content clearly labeled
  [x] Audit trail maintained in database
```

### 5.2 Ongoing Compliance Obligations

| Frequency | Action | Owner |
|:---------:|--------|-------|
| **Per-change** | Update Change Log with backtest impact | Developer |
| **Per-change** | If model parameters changed, re-run CPCV | Developer |
| **Weekly** | Verify OANDA API terms haven't changed | Ops |
| **Monthly** | Review prediction distribution for drift | Quant |
| **Quarterly** | Re-run full bias audit against latest code | Quant |
| **Annually** | Review Perplexity AI terms of service | Legal |
| **On regime change** | Verify HMM correctly detected the shift | Quant |
