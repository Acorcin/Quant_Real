# Quant EOD Engine — Mathematical & Model Specification

> **Scope:** Every formula, hyperparameter, loss function, and optimization algorithm in the codebase
> **Last Updated:** 2026-04-03
> **Notation:** Formulas match the exact implementation; code references link to source lines

---

## Table of Contents

1. [Technical Indicators](#1-technical-indicators)
2. [Candlestick Analysis](#2-candlestick-analysis)
3. [Hidden Markov Model — Regime Detection](#3-hidden-markov-model--regime-detection)
4. [Signal Scoring & Composite Assembly](#4-signal-scoring--composite-assembly)
5. [Feature Vector Normalization](#5-feature-vector-normalization)
6. [XGBoost Meta-Model](#6-xgboost-meta-model)
7. [Cross-Validation: Purged CPCV](#7-cross-validation-purged-cpcv)
8. [Probabilistic Sharpe Ratio](#8-probabilistic-sharpe-ratio)
9. [Backtest Performance Metrics](#9-backtest-performance-metrics)
10. [Execution Model & Friction](#10-execution-model--friction)
11. [Complete Hyperparameter Reference](#11-complete-hyperparameter-reference)

---

## 1. Technical Indicators

### 1.1 Average True Range (ATR-14)

**Source:** [technical.py:L15–L35](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L15-L35)

The ATR measures volatility as the smoothed average of the true range.

**True Range:**

$$TR_t = \max\bigl(H_t - L_t,\;\lvert H_t - C_{t-1}\rvert,\;\lvert L_t - C_{t-1}\rvert\bigr)$$

**Smoothing (EMA, span=14):**

$$ATR_t = \text{EWM}(TR,\;\text{span}=14,\;\text{adjust}=\text{False})$$

The pandas EWM with `span=14` uses decay factor:

$$\alpha = \frac{2}{14 + 1} = \frac{2}{15} \approx 0.1333$$

$$ATR_t = \alpha \cdot TR_t + (1 - \alpha) \cdot ATR_{t-1}$$

> [!NOTE]
> This is **EMA-based ATR**, not the original Wilder smoothing (which uses `com=13`). The difference is minor but means this ATR will react slightly faster to volatility spikes than a classic Wilder ATR.

---

### 1.2 Relative Strength Index (RSI-14)

**Source:** [technical.py:L38–L53](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L38-L53)

**Price changes:**

$$\Delta_t = C_t - C_{t-1}$$

**Gain/Loss separation:**

$$G_t = \max(\Delta_t, 0) \qquad L_t = \max(-\Delta_t, 0)$$

**Smoothing (Wilder's method via `com=13`):**

$$\bar{G}_t = \text{EWM}(G,\;\text{com}=13,\;\text{min\_periods}=14) \qquad \bar{L}_t = \text{EWM}(L,\;\text{com}=13,\;\text{min\_periods}=14)$$

With `com=13`:

$$\alpha = \frac{1}{13 + 1} = \frac{1}{14} \approx 0.0714$$

This is **exactly Wilder's original smoothing** factor:

$$\bar{G}_t = \frac{1}{14} \cdot G_t + \frac{13}{14} \cdot \bar{G}_{t-1}$$

**Relative Strength & RSI:**

$$RS_t = \frac{\bar{G}_t}{\bar{L}_t} \qquad RSI_t = 100 - \frac{100}{1 + RS_t}$$

**Null handling:** If $\bar{L}_t = 0$, $RS_t$ becomes NaN; the result is filled with $RSI = 50.0$ (neutral).

---

### 1.3 Simple Moving Average (SMA-50, SMA-200)

**Source:** [technical.py:L56–L58](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L56-L58)

$$SMA_{n,t} = \frac{1}{n}\sum_{i=0}^{n-1} C_{t-i}$$

For SMA-50: $n=50$, `min_periods=50` (returns NaN until 50 bars available).
For SMA-200: $n=200$, `min_periods=200`.

### 1.4 Exponential Moving Average (EMA-20)

**Source:** [technical.py:L61–L63](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L61-L63)

$$EMA_{20,t} = \alpha \cdot C_t + (1-\alpha) \cdot EMA_{20,t-1} \qquad \alpha = \frac{2}{20+1} \approx 0.0952$$

`adjust=False` — uses the recursive form from the first observation (no expanding-window correction).

> [!NOTE]
> EMA-20 is computed but **not directly included** in the 28-feature vector. It is available in the `technical` dict for internal use but is not one of the `FEATURE_COLS`.

---

### 1.5 Price vs Moving Average (% Distance)

**Source:** [technical.py:L191–L192](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L191-L192)

$$\text{price\_vs\_ma}_{n} = \left(\frac{C_t}{SMA_{n,t}} - 1\right) \times 100$$

Returns the percentage distance from the MA — positive means price is above, negative means below.

---

### 1.6 Rolling Volatility (5d, 20d)

**Source:** [technical.py:L211–L212](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L211-L212)

$$\sigma_{5d} = \text{std}\left(\frac{C_i}{C_{i-1}} - 1\right)_{i=t-4}^{t} \qquad \sigma_{20d} = \text{std}\left(\frac{C_i}{C_{i-1}} - 1\right)_{i=t-19}^{t}$$

Uses pandas default `ddof=1` (sample standard deviation). These are used as **inputs to the HMM**, not as direct features in the meta-model vector.

---

### 1.7 Daily Return

**Source:** [technical.py:L182](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L182)

$$r_t = \left(\frac{C_t}{C_{t-1}} - 1\right) \times 100 \quad (\text{simple return, in percent})$$

---

## 2. Candlestick Analysis

### 2.1 Body Analysis

**Source:** [technical.py:L66–L94](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L66-L94)

| Metric | Formula | Range |
|--------|---------|:-----:|
| Body | $B_t = C_t - O_t$ | ℝ |
| Total Range | $R_t = H_t - L_t$ | ≥ 0 |
| Body Direction | $\text{sign}(B_t) \in \{-1, 0, +1\}$ | Discrete |
| Body % of Range | $\frac{\lvert B_t \rvert}{R_t}$ | [0, 1] |
| Upper Wick % | $\frac{H_t - \max(O_t, C_t)}{R_t}$ | [0, 1] |
| Lower Wick % | $\frac{\min(O_t, C_t) - L_t}{R_t}$ | [0, 1] |

Division by zero guard: if $R_t = 0$ (doji with no range), all ratios are filled with 0.

### 2.2 Candlestick Pattern Detection

**Source:** [technical.py:L97–L139](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L97-L139)

| Pattern | Condition | Output |
|---------|-----------|:------:|
| **Bullish Engulfing** | $\text{dir}_t = +1$ AND $\text{dir}_{t-1} = -1$ AND $\lvert B_t \rvert > \lvert B_{t-1} \rvert$ | Boolean |
| **Bearish Engulfing** | $\text{dir}_t = -1$ AND $\text{dir}_{t-1} = +1$ AND $\lvert B_t \rvert > \lvert B_{t-1} \rvert$ | Boolean |
| **Bullish Pin Bar** | $\text{lower\_wick\_pct} > 0.60$ AND $\text{body\_ratio} < 0.25$ | Boolean |
| **Bearish Pin Bar** | $\text{upper\_wick\_pct} > 0.60$ AND $\text{body\_ratio} < 0.25$ | Boolean |
| **Inside Bar** | $H_t \leq H_{t-1}$ AND $L_t \geq L_{t-1}$ | Boolean |
| **Doji** | $\text{body\_ratio} < 0.10$ | Boolean |

> [!IMPORTANT]
> These patterns are **not** included as direct features in the 28-element meta-model vector. They are used by the **Tier 2 confirmation signals** to boost or penalize composite strength, and their influence reaches the meta-model indirectly through `tier2_confirmation_count`.

---

## 3. Hidden Markov Model — Regime Detection

### 3.1 Model Architecture

**Source:** [hmm_regime.py:L46–L137](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L46-L137)

| Parameter | Value | Source |
|-----------|-------|--------|
| Library | `hmmlearn.hmm.GaussianHMM` | — |
| States ($K$) | 3 | `n_components=3` |
| Covariance type | Diagonal | `covariance_type="diag"` |
| EM iterations | 200 | `n_iter=200` |
| Convergence tolerance | $10^{-4}$ | `tol=1e-4` |
| Random seed | 42 | `random_state=42` |
| Training window | 504 trading days (~2 years) | `lookback_days=504` |

### 3.2 Observation Space

The HMM observes a 2-dimensional feature vector at each time step:

$$\mathbf{x}_t = \begin{bmatrix} \ln\left(\frac{C_t}{C_{t-1}}\right) \\ \sigma_{5d,t} \end{bmatrix} = \begin{bmatrix} \text{log\_return}_t \\ \text{vol\_5d}_t \end{bmatrix}$$

Where:
- $\text{log\_return}_t = \ln(C_t / C_{t-1})$ — natural log return (used **only** in HMM, not elsewhere)
- $\sigma_{5d,t} = \text{std}(\text{log\_return}_{t-4:t})$ — 5-day rolling standard deviation of log returns

### 3.3 Gaussian Emission Model

For each state $k \in \{0, 1, 2\}$, the emission is:

$$p(\mathbf{x}_t \mid s_t = k) = \mathcal{N}(\mathbf{x}_t \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

With diagonal covariance:

$$\boldsymbol{\Sigma}_k = \text{diag}(\sigma_{k,1}^2, \sigma_{k,2}^2)$$

### 3.4 State-Map Construction

After fitting, states are **sorted by ascending mean vol_5d**:

```python
vol_means = model.means_[:, 1]       # Extract vol_5d means
sorted_states = np.argsort(vol_means) # Sort ascending
state_map = {
    sorted_states[0]: 0,   # Lowest vol → semantic "low_vol"
    sorted_states[1]: 1,   # Mid vol → semantic "high_vol_choppy"
    sorted_states[2]: 2,   # Highest vol → semantic "high_vol_crash"
}
```

This ensures semantic consistency: state 0 is always the calmest market, state 2 is always the most volatile, regardless of which raw integer the EM algorithm assigned.

### 3.5 Prediction & Confidence

**Source:** [hmm_regime.py:L139–L193](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L139-L193)

$$\hat{s}_T = \arg\max_k \; p(s_T = k \mid \mathbf{x}_{1:T}) \quad\text{(Viterbi / forward-backward)}$$

$$\text{confidence} = p(s_T = \hat{s}_T \mid \mathbf{x}_{1:T}) \quad\text{(posterior probability of most likely state)}$$

$$\text{days\_in\_regime} = \text{count of consecutive days ending at } T \text{ with state} = \hat{s}_T$$

Transition probabilities for reporting:

$$\text{transition\_prob}[j] = A_{\hat{s}_T, j} \quad\text{(row of transition matrix for current state)}$$

### 3.6 Serialization

| Artifact | Format | Contents |
|----------|--------|----------|
| `hmm_regime.joblib` | joblib | `{model: GaussianHMM, state_map: dict, version: str}` |

---

## 4. Signal Scoring & Composite Assembly

### 4.1 Tier 1 Strength Functions

#### Yield Spread Momentum

**Source:** [tier1.py:L28–L74](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py#L28-L74)

$$\text{strength} = \min\left(\frac{\lvert \Delta S_{5d} \rvert}{2 \times \tau_{\text{regime}}},\; 1.0\right)$$

Where:
- $\Delta S_{5d}$ = 5-day yield spread change in basis points
- $\tau_{\text{regime}}$ = regime-adaptive threshold:

| Regime State | $\tau$ (bps) |
|:---:|:---:|
| 0 (low_vol) | 8.0 |
| 1 (choppy) | 15.0 |
| 2 (crash) | 20.0 |

Direction: $\Delta S_{5d} > \tau \Rightarrow$ SHORT EUR/USD; $\Delta S_{5d} < -\tau \Rightarrow$ LONG EUR/USD.

#### Sentiment Extreme Fade

**Source:** [tier1.py:L77–L119](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py#L77-L119)

$$\text{strength}_{\text{long}} = \min\left(\frac{(1 - p_{\text{long}}) - (1 - \theta_H)}{0.18},\; 1.0\right) = \min\left(\frac{\theta_H - p_{\text{long}}}{0.18},\; 1.0\right)$$

$$\text{strength}_{\text{short}} = \min\left(\frac{p_{\text{long}} - \theta_H}{0.18},\; 1.0\right)$$

Where:
- $p_{\text{long}}$ = retail percent long (0.0 to 1.0)
- $\theta_H = 0.72$ (`SENTIMENT_EXTREME_HIGH`)
- $\theta_L = 0.28$ (`SENTIMENT_EXTREME_LOW`)
- $0.18$ = strength scaling span (maps $\theta_H$ → 0.0 strength, $\theta_H + 0.18 = 0.90$ → 1.0 strength)

Direction: $p_{\text{long}} > \theta_H \Rightarrow$ SHORT (fade the crowd); $p_{\text{long}} < \theta_L \Rightarrow$ LONG.

#### AI Macro Sentiment

**Source:** [tier1.py:L122–L169](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py#L122-L169)

$$\text{strength} = \min\left(\lvert s_{\text{AI}} \rvert \times c_{\text{AI}},\; 1.0\right)$$

Gate: Signal fires only if $\lvert s_{\text{AI}} \rvert > 0.5$ AND $c_{\text{AI}} > 0.6$.

Where:
- $s_{\text{AI}} \in [-1.0, +1.0]$ = LLM macro sentiment score
- $c_{\text{AI}} \in [0.0, 1.0]$ = LLM self-assessed confidence

#### EOD Event Reversal

**Source:** [tier1.py:L172–L279](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py#L172-L279)

$$\text{strength} = \min\left(0.85 + 0.02 \times (N_{\text{events}} - 1),\; 1.0\right)$$

Where $N_{\text{events}}$ = count of high-impact events with non-neutral surprise direction.

Reversal detection: if the aggregated surprise direction is opposite to the daily candle body direction.

### 4.2 Composite Signal Scoring

**Source:** [composite.py:L15–L111](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/composite.py#L15-L111)

**Step 1: Directional Voting (Tier 1 only)**

$$S_{\text{long}} = \sum_{i \in \text{T1}:\, d_i = \text{long}} w_i \qquad S_{\text{short}} = \sum_{i \in \text{T1}:\, d_i = \text{short}} w_i$$

$$\text{direction} = \begin{cases} \text{long} & \text{if } S_{\text{long}} > S_{\text{short}} \\ \text{short} & \text{if } S_{\text{short}} > S_{\text{long}} \\ \text{flat} & \text{if tied} \end{cases}$$

**Step 2: Base Strength (average strength of winning side)**

$$\text{base\_strength} = \frac{S_{\text{winner}}}{N_{\text{winner}}}$$

**Step 3: Tier 2 Adjustment**

$$\Delta_{\text{T2}} = \sum_{j \in \text{T2}} \begin{cases} +0.05 & \text{if confirmed} \\ -0.02 & \text{if not confirmed} \end{cases}$$

**Step 4: Final Composite Strength**

$$\text{composite\_strength} = \text{clamp}\bigl(\text{base\_strength} + \Delta_{\text{T2}},\; 0.0,\; 1.0\bigr)$$

**Step 5: Floor Gate**

$$\text{If } \text{composite\_strength} < 0.15 \Rightarrow \text{direction} = \text{flat},\; \text{strength} = 0.0$$

---

## 5. Feature Vector Normalization

### 5.1 Pre-Processing (Feature Assembly)

**Source:** [vector.py:L109–L112](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/vector.py#L109-L112)

Before entering the model, all `None` values are replaced with `0.0`:

```python
for k, v in vector.items():
    if v is None:
        vector[k] = 0.0
```

### 5.2 Categorical Encoding

**Source:** [vector.py:L17–L18](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/vector.py#L17-L18)

| Variable | hawkish / risk_on | neutral | dovish / risk_off |
|:--------:|:-:|:-:|:-:|
| `fed_stance_encoded` | +1 | 0 | −1 |
| `ecb_stance_encoded` | +1 | 0 | −1 |
| `risk_sentiment_encoded` | +1 | 0 | −1 |

### 5.3 Z-Score Normalization (StandardScaler)

**Source:** [meta_model.py:L123–L124](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L123-L124)

At training time, all 28 features are z-score normalized:

$$x'_{j} = \frac{x_j - \bar{x}_j}{\hat{\sigma}_j}$$

Where $\bar{x}_j$ and $\hat{\sigma}_j$ are the training-set mean and standard deviation of feature $j$.

The fitted `StandardScaler` is persisted in `meta_model.joblib` and applied to inference-time feature vectors. At training, the scaler is fit on **raw features** before CPCV; the final production model is trained on **scaler-transformed data**.

> [!WARNING]
> **CPCV inconsistency:** In `_run_cpcv()`, each fold fits its own `StandardScaler` on the training split (line 328), which is correct. But the final production model uses a scaler fit on **all data** (line 124). This means the production scaler has seen future data relative to any historical point, which is acceptable for a production-only model but means CPCV metrics are slightly more honest than production performance.

---

## 6. XGBoost Meta-Model

### 6.1 Architecture

**Source:** [meta_model.py:L140–L153](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L140-L153)

| Parameter | **Final Model** (Production) | **CPCV Fold Model** |
|-----------|:---:|:---:|
| `n_estimators` | **200** | **100** |
| `max_depth` | 4 | 4 |
| `learning_rate` (η) | 0.05 | 0.05 |
| `subsample` | 0.8 | 0.8 |
| `colsample_bytree` | 0.8 | 0.8 |
| `reg_alpha` (L1) | **0.1** | *default (0)* |
| `reg_lambda` (L2) | **1.0** | *default (1)* |
| `min_child_weight` | **5** | *default (1)* |
| `eval_metric` | logloss | logloss |
| `random_state` | 42 | 42 |

> [!IMPORTANT]
> The production model uses **200 trees** with stronger regularization (`reg_alpha=0.1`, `min_child_weight=5`), while the CPCV fold models use **100 trees** with default regularization. This means CPCV Sharpe estimates come from a **weaker model** than production — a conservative choice that avoids overstating out-of-sample performance.

### 6.2 Loss Function

**Binary cross-entropy (logloss):**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \ln(\hat{p}_i) + (1-y_i)\ln(1-\hat{p}_i)\right]$$

Where:
- $y_i \in \{0, 1\}$ = binary label (1 = signal was profitable, 0 = not)
- $\hat{p}_i$ = predicted probability that the trade is profitable

### 6.3 Optimization Algorithm

XGBoost uses **second-order gradient-boosted trees**:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(\mathbf{x}_i)$$

Each tree $f_t$ is fit to minimize:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{N}\left[g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t(\mathbf{x}_i)^2\right] + \Omega(f_t)$$

Where:
- $g_i = \partial \mathcal{L} / \partial \hat{y}_i^{(t-1)}$ = first-order gradient of log loss
- $h_i = \partial^2 \mathcal{L} / \partial (\hat{y}_i^{(t-1)})^2$ = second-order Hessian
- $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} \lvert w_j \rvert$ = tree regularization term

For logloss: $g_i = \hat{p}_i - y_i$ and $h_i = \hat{p}_i(1 - \hat{p}_i)$.

### 6.4 Position Sizing Decision Function

**Source:** [meta_model.py:L232–L245](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L232-L245)

$$(\text{direction},\; \text{size}) = \begin{cases} (\text{flat},\; 0.0\times) & \text{if } \hat{p} < 0.55 \\ (\text{signal\_dir},\; 0.5\times) & \text{if } 0.55 \leq \hat{p} < 0.70 \\ (\text{signal\_dir},\; 1.0\times) & \text{if } \hat{p} \geq 0.70 \end{cases}$$

**Constraint:** If `primary_signal_direction == 0` (flat), then output is always `(flat, 0.0×)` regardless of $\hat{p}$. This is the primary signal veto.

### 6.5 Feature Importance (SHAP)

**Source:** [meta_model.py:L396–L423](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L396-L423)

Primary: `shap.TreeExplainer` → mean absolute SHAP values across all training samples:

$$\text{importance}_j = \frac{1}{N}\sum_{i=1}^{N}\lvert\phi_j^{(i)}\rvert$$

Fallback (if SHAP fails): XGBoost native `feature_importances_` (gain-based).

### 6.6 Serialization

| Artifact | Format | Contents |
|----------|--------|----------|
| `meta_model_xgb.json` | XGBoost native JSON | Trained model (trees + parameters) |
| `meta_model.joblib` | joblib | `{scaler: StandardScaler, version: str, cpcv: dict, shap: list, feature_cols: list}` |

---

## 7. Cross-Validation: Purged CPCV

**Source:** [meta_model.py:L260–L394](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L260-L394)

### 7.1 Configuration

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| $N$ (groups) | 6 | Divides data into 6 contiguous temporal blocks |
| $k$ (test groups) | 2 | Each fold holds out 2 groups → $\binom{6}{2} = 15$ paths |
| Purge | 5 days | Removes 5 days before each test boundary |
| Embargo | 2 days | Removes 2 days after each test boundary |

### 7.2 Data Partitioning

```
Total samples: N
Group size: floor(N / 6)

Group 0: indices [0, group_size)
Group 1: indices [group_size, 2*group_size)
...
Group 5: indices [5*group_size, N)    ← absorbs remainder
```

For each combination $C = (g_a, g_b)$ where $g_a < g_b$:
- **Test set:** $\mathcal{I}_{\text{test}} = \text{Group}_{g_a} \cup \text{Group}_{g_b}$
- **Purge zone:** For each $i \in \mathcal{I}_{\text{test}}$, remove indices $\{i - 5, \ldots, i + 2\}$ from training
- **Train set:** All indices not in test and not in purge zone

### 7.3 Per-Path Evaluation

For each path, a fresh XGBClassifier (100 trees) + StandardScaler is trained. Then:

$$\text{signal}_i = \begin{cases} 1 & \text{if } \hat{p}_i > 0.55 \\ 0 & \text{otherwise} \end{cases}$$

**Daily returns per path (using realized returns):**

$$r_i^{\text{path}} = \text{signal}_i \times (2y_i - 1) \times \lvert r_i^{\text{realized}} \rvert$$

Where $r_i^{\text{realized}}$ is the actual next-trading-day close-to-close return from the `bars` table.

**Fallback (synthetic returns):** If realized returns are unavailable:

$$r_i^{\text{path}} = \text{signal}_i \times (2y_i - 1) \times 0.001$$

### 7.4 Per-Path Sharpe Ratio

$$\text{Sharpe}_{\text{path}} = \frac{\bar{r}^{\text{path}}}{\hat{\sigma}^{\text{path}}} \times \sqrt{252}$$

### 7.5 Statistical Significance

Across all 15 paths:

$$H_0: \mu_{\text{Sharpe}} = 0 \qquad H_1: \mu_{\text{Sharpe}} > 0$$

One-sample one-sided t-test (`scipy.stats.ttest_1samp`, `alternative="greater"`):

$$t = \frac{\bar{S} - 0}{s_S / \sqrt{n_{\text{paths}}}} \qquad p = P(T > t \mid T \sim t_{n-1})$$

Significance criterion: $p < 0.05$ AND $\bar{S} > 0$.

---

## 8. Probabilistic Sharpe Ratio (PSR)

**Source:** [meta_model.py:L561–L587](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L561-L587)

Following Bailey & López de Prado (*AFML*, Chapter 14):

**Input:** Pooled daily returns from all CPCV paths.

**Empirical Sharpe Ratio:**

$$\widehat{SR} = \frac{\bar{r}}{\hat{\sigma}} \times \sqrt{252}$$

**Variance of Sharpe Ratio (accounting for non-normality):**

$$\text{Var}(\widehat{SR}) = \frac{1 + \frac{1}{2}\widehat{SR}^2 - \hat{\gamma}_3 \cdot \widehat{SR} + \frac{\hat{\gamma}_4}{4}\widehat{SR}^2}{T - 1}$$

Where:
- $\hat{\gamma}_3$ = sample skewness (unbiased, `scipy.stats.skew(bias=False)`)
- $\hat{\gamma}_4$ = excess kurtosis (Fisher=True, `scipy.stats.kurtosis(fisher=True, bias=False)`)
- $T$ = number of observations

**PSR (probability that true Sharpe > 0):**

$$\text{PSR} = \Phi\left(\frac{\widehat{SR}}{\sqrt{\text{Var}(\widehat{SR})}}\right)$$

Where $\Phi$ is the standard normal CDF. PSR > 0.95 indicates statistical confidence that the strategy has positive risk-adjusted returns. The variance is floored at $10^{-12}$ to prevent division by zero.

---

## 9. Backtest Performance Metrics

**Source:** [backtest_loop.py:L63–L93](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L63-L93) and [L187–L216](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L187-L216)

### 9.1 Maximum Drawdown (MDD)

$$\text{MDD} = \min_{t}\left(\frac{E_t}{\max_{s \leq t} E_s} - 1\right)$$

Peak is tracked cumulatively. Returns a negative value (e.g., −0.12 = 12% drawdown).

### 9.2 Annualized Sharpe Ratio

$$\text{Sharpe}_{\text{annual}} = \frac{\bar{r}}{\hat{\sigma}(r)} \times \sqrt{252}$$

Where $\hat{\sigma}$ uses `ddof=1` (Bessel correction). If $\hat{\sigma} \leq 10^{-12}$ or $N < 2$, returns 0.0.

> [!NOTE]
> This is an **excess return Sharpe with risk-free rate = 0**. No risk-free rate subtraction is performed.

### 9.3 Annualized Sortino Ratio

$$\text{Sortino}_{\text{annual}} = \frac{\bar{r}}{\hat{\sigma}_{\text{down}}(r)} \times \sqrt{252}$$

Where $\hat{\sigma}_{\text{down}}$ is the standard deviation computed **only on negative returns**:

$$\hat{\sigma}_{\text{down}} = \text{std}\{r_i : r_i < 0\} \quad \text{(ddof=1)}$$

If fewer than 2 negative returns exist, returns 0.0.

### 9.4 CAGR (Compound Annual Growth Rate)

$$\text{CAGR} = \left(\frac{E_T}{E_0}\right)^{1/Y} - 1 \qquad Y = \frac{N_{\text{periods}}}{252}$$

### 9.5 Calmar Ratio

$$\text{Calmar} = \frac{\text{CAGR}}{\lvert\text{MDD}\rvert}$$

Returns 0.0 if MDD ≥ 0 (no drawdown).

### 9.6 Exposure & Turnover

$$\text{Exposure} = \frac{1}{N}\sum_{t=1}^{N}\mathbb{1}[\text{size}_t > 0]$$

$$\text{Turnover}_{\text{total}} = \sum_{t=1}^{N}\lvert\text{size}_t - \text{size}_{t-1}\rvert \qquad \text{Turnover/day} = \frac{\text{Turnover}_{\text{total}}}{N}$$

---

## 10. Execution Model & Friction

**Source:** [backtest_loop.py:L129–L160](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L129-L160)

### 10.1 Execution Timeline

| Event | Time | Price Used |
|-------|------|:---------:|
| Signal generated | Day T, 5:20 PM ET | — |
| Entry | Day T+1, Open | $O_{T+1}$ |
| Exit | Day T+1, Close | $C_{T+1}$ |

### 10.2 Raw Return

$$r_{\text{raw}} = \frac{C_{T+1}}{O_{T+1}} - 1$$

### 10.3 Trade PnL (with leverage & friction)

$$\text{PnL}_{\text{long}} = \text{size} \times L \times (r_{\text{raw}} - \text{spread\_cost})$$

$$\text{PnL}_{\text{short}} = \text{size} \times L \times (-r_{\text{raw}} - \text{spread\_cost})$$

$$\text{PnL}_{\text{flat}} = 0$$

Where:
- $\text{size} \in \{0.0, 0.5, 1.0\}$ = position size multiplier from meta-model
- $L = 10.0$ = fixed leverage
- $\text{spread\_cost} = \frac{1.5}{10{,}000} = 0.00015$ (1.5 pips, applied once)

### 10.4 Equity Evolution

$$E_t = E_{t-1} \times (1 + \text{PnL}_t)$$

Multiplicative (compounding) equity curve, not additive.

### 10.5 Friction Analysis

At 10× leverage with 1.5 pip spread:

$$\text{Effective spread cost per trade} = 10 \times 0.00015 = 0.0015 = 0.15\%$$

For a half-size trade (0.5×):

$$\text{Effective spread cost} = 0.5 \times 10 \times 0.00015 = 0.075\%$$

The strategy needs a **minimum raw alpha of 15 bps per full-size trade** just to break even.

---

## 11. Complete Hyperparameter Reference

### 11.1 Model Hyperparameters

| Model | Parameter | Value | Code Reference |
|-------|-----------|:-----:|:--------------:|
| **XGBoost (prod)** | n_estimators | 200 | [meta_model.py:L141](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L141) |
| | max_depth | 4 | [meta_model.py:L142](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L142) |
| | learning_rate | 0.05 | [meta_model.py:L143](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L143) |
| | subsample | 0.8 | [meta_model.py:L144](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L144) |
| | colsample_bytree | 0.8 | [meta_model.py:L145](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L145) |
| | reg_alpha (L1) | 0.1 | [meta_model.py:L146](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L146) |
| | reg_lambda (L2) | 1.0 | [meta_model.py:L147](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L147) |
| | min_child_weight | 5 | [meta_model.py:L148](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L148) |
| | eval_metric | logloss | [meta_model.py:L151](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L151) |
| **XGBoost (CPCV)** | n_estimators | 100 | [meta_model.py:L331](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L331) |
| | max_depth | 4 | [meta_model.py:L331](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L331) |
| | learning_rate | 0.05 | [meta_model.py:L331](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L331) |
| | subsample | 0.8 | [meta_model.py:L332](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L332) |
| | colsample_bytree | 0.8 | [meta_model.py:L332](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/meta_model.py#L332) |
| **HMM** | n_components | 3 | [hmm_regime.py:L84](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L84) |
| | covariance_type | diag | [hmm_regime.py:L85](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L85) |
| | n_iter | 200 | [hmm_regime.py:L86](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L86) |
| | tol | 1e-4 | [hmm_regime.py:L88](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L88) |
| | lookback_days | 504 | [hmm_regime.py:L49](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/models/hmm_regime.py#L49) |

### 11.2 Signal Thresholds

| Parameter | Value | Source |
|-----------|:-----:|--------|
| Yield threshold (low_vol) | 8.0 bps | [tier1.py:L21](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py#L21) |
| Yield threshold (choppy) | 15.0 bps | [tier1.py:L22](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py#L22) |
| Yield threshold (crash) | 20.0 bps | [tier1.py:L23](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py#L23) |
| Sentiment extreme (high) | 0.72 | [settings.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/config/settings.py) |
| Sentiment extreme (low) | 0.28 | [settings.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/config/settings.py) |
| Sentiment strength span | 0.18 | [tier1.py:L25](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py#L25) |
| AI sentiment score gate | ±0.5 | [tier1.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py) |
| AI confidence gate | 0.6 | [tier1.py](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier1.py) |
| Composite strength floor | 0.15 | [composite.py:L90](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/composite.py#L90) |
| T2 confirmation bonus | +0.05 | [composite.py:L82](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/composite.py#L82) |
| T2 non-confirmation penalty | −0.02 | [composite.py:L85](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/composite.py#L85) |
| RSI oversold | < 30 | [tier2.py:L74](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier2.py#L74) |
| RSI overbought | > 70 | [tier2.py:L76](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/signals/tier2.py#L76) |
| Pin bar wick threshold | > 60% of range | [technical.py:L123](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L123) |
| Pin bar body max | < 25% of range | [technical.py:L123](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L123) |
| Doji body max | < 10% of range | [technical.py:L130](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/features/technical.py#L130) |

### 11.3 Meta-Model Decision Thresholds

| Threshold | Value | Effect |
|-----------|:-----:|--------|
| Flat (no trade) | $\hat{p} < 0.55$ | size = 0.0× |
| Half position | $0.55 \leq \hat{p} < 0.70$ | size = 0.5× |
| Full position | $\hat{p} \geq 0.70$ | size = 1.0× |
| Primary signal veto | signal_direction = 0 | Always flat, regardless of $\hat{p}$ |
| CPCV significance | $p < 0.05$ AND $\bar{S} > 0$ | Model passes validation |
| Min training samples | 50 | Model refuses to train below this |
| CPCV min train per fold | 30 | Fold skipped if fewer |
| CPCV min test per fold | 10 | Fold skipped if fewer |

### 11.4 Backtest Parameters

| Parameter | Value | Source |
|-----------|:-----:|--------|
| Leverage | 10.0× | [backtest_loop.py:L130](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L130) |
| Spread | 1.5 pips | [backtest_loop.py:L131](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L131) |
| Default equity | $10,000 | [backtest_loop.py:L240](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L240) |
| Annualization factor | 252 | [backtest_loop.py:L75](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L75) |
| Execution | Open→Close T+1 | [backtest_loop.py:L146–L147](file:///c:/Users/angel/OneDrive/Documents/GitHub/quant-eod-engine/backtest_loop.py#L146-L147) |
