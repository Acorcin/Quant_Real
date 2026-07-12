# Quant EOD Engine — System Architecture Diagram

> **Version:** v1.0.0 | **Date:** 2026-04-03
> All diagrams render via Mermaid.js and map directly to source files.

---

## 1. End-to-End Pipeline Flow

The complete data-to-signal pipeline executed daily at 5:15 PM ET.

```mermaid
flowchart TB
    subgraph EXT["☁️ EXTERNAL DATA SOURCES"]
        direction LR
        OA["🏦 OANDA V20 API\n/v3/instruments/candles\n/v3/accounts/instruments"]
        FR["📊 FRED API\nDGS2 · IRLTLT01DEM156N"]
        PX["🤖 Perplexity Sonar API\nsonar-pro model"]
        CAL["📅 Calendar Stub\n(not yet integrated)"]
    end

    subgraph INGEST["📥 PHASE 1 — DATA INGESTION"]
        direction TB
        S1["Step 1: fetch_bars()\noanda_bars.py"]
        S2["Step 2: fetch_yields()\nfred_yields.py"]
        S3["Step 3: fetch_sentiment()\noanda_sentiment.py"]
        S4["Step 4: fetch_swaps()\nswap_rates.py"]
        S5["Step 5: fetch_calendar()\ncalendar.py"]
        S6["Step 6: fetch_ai_sentiment()\nperplexity_sentiment.py"]
    end

    subgraph DB["🗄️ POSTGRESQL DATABASE"]
        direction LR
        T1[("bars\nD + H4 OHLCV")]
        T2[("yield_data\nUS/DE spreads")]
        T3[("sentiment\n% long/short")]
        T4[("swap_rates\npips/day")]
        T5[("calendar_events\nhigh-impact")]
        T6[("ai_sentiment\nLLM scores")]
        T7[("feature_vectors\n28-dim JSON")]
        T8[("predictions\ndirection + prob")]
        T9[("signals\ncomposite + raw")]
        T10[("pipeline_runs\naudit log")]
        T11[("model_runs\ntraining history")]
    end

    subgraph ENGINE["⚙️ PHASE 2 — PREDICTION ENGINE"]
        direction TB
        E1["Step 7: RegimeDetector\nhmm_regime.py\n3-state Gaussian HMM"]
        E2["Step 8a: compute_all_features()\ntechnical.py\nATR · RSI · MA · Candles"]
        E3["Step 8b: generate_all_tier1()\ntier1.py\n4 alpha signals"]
        E4["Step 8c: generate_all_tier2()\ntier2.py\n4 confirmations"]
        E5["Step 9: compute_composite()\ncomposite.py\nWeighted voting + T2 adj"]
        E6["Step 10: assemble_feature_vector()\nvector.py\n28 features"]
        E7["Step 11: MetaModel.predict()\nmeta_model.py\nXGBoost probability"]
    end

    subgraph RISK["🛡️ RISK & SIZING LAYER"]
        direction TB
        R1{"Prob < 0.55?"}
        R2{"Primary signal\n== flat?"}
        R3["Half Position\n0.5× size"]
        R4["Full Position\n1.0× size"]
        R5["FLAT\n0.0× size"]
    end

    subgraph OUTPUT["📤 SIGNAL OUTPUT"]
        direction TB
        O1["Step 12: send_signal()\ndiscord_notify.py"]
        O2["Step 13: log_pipeline_run()\ndaily_loop.py"]
        DC["💬 Discord Webhook\nRich embed with\nprediction + regime + SHAP"]
        HU["👤 Human Trader\nManual execution"]
    end

    %% External → Ingestion
    OA --> S1 & S3 & S4
    FR --> S2
    PX --> S6
    CAL -.->|stub| S5

    %% Ingestion → Database
    S1 --> T1
    S2 --> T2
    S3 --> T3
    S4 --> T4
    S5 --> T5
    S6 --> T6

    %% Database → Engine
    T1 --> E1 & E2
    T2 --> E3
    T3 --> E3
    T6 --> E3
    T5 --> E3
    T1 --> E4

    %% Engine internal flow
    E1 --> E3
    E2 --> E3
    E2 --> E4
    E3 --> E5
    E4 --> E5
    E5 --> E6
    E1 --> E6
    T2 & T3 & T4 & T6 --> E6
    E6 --> T7
    E6 --> E7

    %% Engine → Risk Layer
    E7 --> R1
    R1 -->|Yes| R5
    R1 -->|No| R2
    R2 -->|Yes| R5
    R2 -->|"No, prob 0.55–0.70"| R3
    R2 -->|"No, prob ≥ 0.70"| R4

    %% Risk → Output
    R3 & R4 & R5 --> O1
    R3 & R4 & R5 --> T8
    E5 --> T9
    O1 --> DC
    DC --> HU
    O1 --> O2
    O2 --> T10

    %% Styling
    classDef ext fill:#1a1a2e,stroke:#e94560,color:#eee,stroke-width:2px
    classDef ingest fill:#16213e,stroke:#0f3460,color:#eee
    classDef db fill:#0f3460,stroke:#53a8b6,color:#eee
    classDef engine fill:#1b262c,stroke:#bbe1fa,color:#eee
    classDef risk fill:#2d132c,stroke:#ee4540,color:#eee
    classDef output fill:#0a3d62,stroke:#38ada9,color:#eee
    classDef decision fill:#2d132c,stroke:#ee4540,color:#eee

    class OA,FR,PX,CAL ext
    class S1,S2,S3,S4,S5,S6 ingest
    class T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11 db
    class E1,E2,E3,E4,E5,E6,E7 engine
    class R1,R2 decision
    class R3,R4,R5 risk
    class O1,O2,DC,HU output
```

---

## 2. Model Architecture Detail

Zoomed view of the XGBoost meta-labeling pipeline and its validation framework.

```mermaid
flowchart LR
    subgraph INPUT["Feature Vector (28-dim)"]
        direction TB
        F1["Regime: state, days_in_regime"]
        F2["Macro: yield_spread, Δ5d, Δ20d"]
        F3["Sentiment: pct_long, extreme"]
        F4["AI: score, confidence, stances"]
        F5["Technical: ATR, RSI, MAs, candles"]
        F6["Calendar: reversal, surprise_mag"]
        F7["Temporal: day_of_week, is_friday"]
        F8["Carry: long_swap, short_swap"]
        F9["Signal: direction, count, strength, T2"]
    end

    subgraph NORM["Normalization"]
        SC["StandardScaler\nz-score transform\nfit on training data"]
    end

    subgraph MODEL["XGBoost Classifier"]
        direction TB
        XG["XGBClassifier\n200 trees · depth=4\nη=0.05 · L1=0.1 · L2=1.0\nsubsample=0.8 · colsample=0.8\nmin_child_weight=5"]
    end

    subgraph GATE["3-Tier Probability Gate"]
        direction TB
        G1["P < 0.55 → FLAT (0.0×)"]
        G2["0.55 ≤ P < 0.70 → HALF (0.5×)"]
        G3["P ≥ 0.70 → FULL (1.0×)"]
    end

    subgraph VETO["Signal Veto"]
        V1{"primary_signal\n_direction == 0?"}
        V2["Override → FLAT"]
        V3["Pass through"]
    end

    subgraph EXPLAIN["Explainability"]
        SH["SHAP TreeExplainer\nTop 5 features per prediction"]
    end

    subgraph VALID["Validation (Training Only)"]
        direction TB
        CV["CPCV\nN=6, k=2 → 15 paths\npurge=5, embargo=2"]
        PSR["Probabilistic Sharpe Ratio\nBailey & López de Prado\nskew + kurtosis adjusted"]
    end

    INPUT --> SC
    SC --> XG
    XG --> GATE
    XG --> SH
    GATE --> V1
    V1 -->|Yes| V2
    V1 -->|No| V3

    XG -.->|training| CV
    CV -.-> PSR

    classDef input fill:#1b262c,stroke:#bbe1fa,color:#eee
    classDef proc fill:#16213e,stroke:#0f3460,color:#eee
    classDef gate fill:#2d132c,stroke:#ee4540,color:#eee
    classDef valid fill:#0a3d62,stroke:#38ada9,color:#eee

    class F1,F2,F3,F4,F5,F6,F7,F8,F9 input
    class SC,XG,SH proc
    class G1,G2,G3,V1,V2,V3 gate
    class CV,PSR valid
```

---

## 3. Signal Generation Cascade

How the 4 Tier 1 alpha signals and 4 Tier 2 confirmations combine into a composite score.

```mermaid
flowchart TB
    subgraph T1["TIER 1 — Alpha Signals (Direction + Strength)"]
        direction LR
        A1["🏦 Yield Spread\nMomentum\n\nΔ5d vs regime τ\nStrength: |Δ|/(2τ)"]
        A2["👥 Sentiment\nExtreme Fade\n\npct_long > 72% → SHORT\npct_long < 28% → LONG"]
        A3["🤖 AI Macro\nSentiment\n\n|score| > 0.5 +\nconfidence > 0.6"]
        A4["📅 EOD Event\nReversal\n\nSurprise opposes\ncandle body"]
    end

    subgraph VOTE["Directional Voting"]
        DV["Strength-weighted\nsum per side\n\nMajority side wins\nBase = avg strength"]
    end

    subgraph T2["TIER 2 — Confirmations (Yes/No per direction)"]
        direction LR
        B1["📊 Candle\nPattern\n\nEngulfing or\npin bar match"]
        B2["📈 RSI-14\nExtreme\n\n< 30 or > 70\nzone match"]
        B3["📉 MA\nAlignment\n\nMA50 vs MA200\ntrend match"]
        B4["🕐 Multi-TF\n4H Bars\n\n2/2 H4 bars\nagree"]
    end

    subgraph ADJ["T2 Adjustment"]
        TA["+0.05 per confirm\n−0.02 per non-confirm\n\n4 confirms: +0.20\n0 confirms: −0.08"]
    end

    subgraph COMP["Composite Output"]
        direction TB
        CF{"strength\n≥ 0.15?"}
        CO["composite_direction\ncomposite_strength\nprimary_signal_direction\ntier2_confirmation_count"]
        FLAT["FLAT\nstrength = 0"]
    end

    A1 & A2 & A3 & A4 --> DV
    DV --> T2
    B1 & B2 & B3 & B4 --> ADJ
    DV --> ADJ
    ADJ --> CF
    CF -->|"Yes"| CO
    CF -->|"No"| FLAT

    classDef t1 fill:#1a1a2e,stroke:#e94560,color:#eee
    classDef t2 fill:#16213e,stroke:#53a8b6,color:#eee
    classDef vote fill:#0f3460,stroke:#bbe1fa,color:#eee
    classDef comp fill:#2d132c,stroke:#ee4540,color:#eee

    class A1,A2,A3,A4 t1
    class B1,B2,B3,B4 t2
    class DV,TA vote
    class CF,CO,FLAT comp
```

---

## 4. Data Flow & Storage Architecture

How data moves between external sources, the PostgreSQL persistence layer, and the computation modules.

```mermaid
flowchart LR
    subgraph SOURCES["External APIs"]
        direction TB
        S1["OANDA V20"]
        S2["FRED"]
        S3["Perplexity AI"]
    end

    subgraph FETCHERS["Fetcher Layer\n(fetchers/)"]
        direction TB
        F1["oanda_bars.py"]
        F2["fred_yields.py"]
        F3["oanda_sentiment.py"]
        F4["swap_rates.py"]
        F5["calendar.py"]
        F6["perplexity_sentiment.py"]
    end

    subgraph PG["PostgreSQL (quant_eod)"]
        direction TB
        subgraph RAW["Raw Data Tables"]
            R1[("bars")]
            R2[("yield_data")]
            R3[("sentiment")]
            R4[("swap_rates")]
            R5[("calendar_events")]
            R6[("ai_sentiment")]
        end
        subgraph DERIVED["Derived/Output Tables"]
            D1[("feature_vectors")]
            D2[("predictions")]
            D3[("signals")]
            D4[("daily_snapshots")]
        end
        subgraph AUDIT["Audit Tables"]
            A1[("pipeline_runs")]
            A2[("model_runs")]
            A3[("model_artifacts\nBYTEA blobs")]
        end
    end

    subgraph COMPUTE["Computation Modules"]
        direction TB
        C1["technical.py\nIndicators"]
        C2["hmm_regime.py\nRegime"]
        C3["tier1.py\nSignals"]
        C4["tier2.py\nConfirm"]
        C5["composite.py\nScoring"]
        C6["vector.py\nAssembly"]
        C7["meta_model.py\nPrediction"]
    end

    subgraph DISK["Local Filesystem"]
        direction TB
        K1["/tmp/modelartifacts/\n(Ephemeral buffer)"]
        K2["logs/\ndaily_YYYY-MM-DD.log"]
    end

    subgraph NOTIFY["Delivery"]
        N1["discord_notify.py\n→ Discord Webhook"]
    end

    S1 --> F1 & F3 & F4
    S2 --> F2
    S3 --> F5 & F6

    F1 --> R1
    F2 --> R2
    F3 --> R3
    F4 --> R4
    F5 --> R5
    F6 --> R6

    R1 --> C1 & C2
    R2 & R3 & R5 & R6 --> C3
    R1 --> C4
    C1 --> C3
    C3 --> C5
    C4 --> C5
    C2 & C5 & R2 & R3 & R4 & R6 --> C6
    C6 --> D1
    C6 --> C7
    C5 --> D3
    C7 --> D2

    C7 -.->|buffer| K1
    C2 -.->|buffer| K1
    C7 <-->|read/write blob| A3
    C2 <-->|read/write blob| A3
    C7 --> N1
    N1 --> D4

    classDef source fill:#1a1a2e,stroke:#e94560,color:#eee
    classDef fetch fill:#16213e,stroke:#0f3460,color:#eee
    classDef raw fill:#0f3460,stroke:#53a8b6,color:#eee
    classDef derived fill:#0a3d62,stroke:#38ada9,color:#eee
    classDef audit fill:#2d132c,stroke:#c3073f,color:#eee
    classDef compute fill:#1b262c,stroke:#bbe1fa,color:#eee
    classDef disk fill:#2d132c,stroke:#ee4540,color:#eee

    class S1,S2,S3 source
    class F1,F2,F3,F4,F5,F6 fetch
    class R1,R2,R3,R4,R5,R6 raw
    class D1,D2,D3,D4 derived
    class A1,A2,A3 audit
    class C1,C2,C3,C4,C5,C6,C7 compute
    class K1,K2 disk
```

---

## 5. Backtest Execution Model

Timing and PnL computation in `backtest_loop.py`.

```mermaid
sequenceDiagram
    participant D as Day T (Signal)
    participant T1 as Day T+1 (Execution)
    participant DB as PostgreSQL
    participant MM as MetaModel

    Note over D: 5:15 PM ET — Daily bar complete

    D->>DB: Load feature_vector for Day T
    DB-->>D: 28-feature dict
    D->>MM: predict(feature_vector)
    MM-->>D: {direction, prob, size}

    Note over D,T1: Overnight — position determined

    T1->>DB: Load bars for Day T+1
    DB-->>T1: {open, close}

    Note over T1: Entry at T+1 Open

    rect rgb(45, 19, 44)
        Note over T1: raw_ret = (close / open) - 1.0
        Note over T1: spread_cost = 1.5 pips / open
        Note over T1: pnl = direction × size × leverage × raw_ret − spread_cost
        Note over T1: equity *= (1 + pnl)
    end

    Note over T1: Exit at T+1 Close
```

---

## 6. HMM Regime Detection

State machine for the 3-state Hidden Markov Model.

```mermaid
stateDiagram-v2
    [*] --> low_vol

    low_vol: 📈 State 0 — Low Volatility Trend
    low_vol: Lowest mean vol_5d
    low_vol: Yield threshold: 8 bps

    choppy: 🌀 State 1 — High Vol Choppy
    choppy: Mid-range vol_5d
    choppy: Yield threshold: 15 bps

    crash: ⚡ State 2 — High Vol Crash
    crash: Highest mean vol_5d
    crash: Yield threshold: 20 bps

    low_vol --> choppy: Volatility increases
    low_vol --> crash: Shock event
    choppy --> low_vol: Volatility normalizes
    choppy --> crash: Volatility spikes
    crash --> choppy: Partial recovery
    crash --> low_vol: Full recovery

    note right of low_vol
        Training: 504-day rolling window
        Observations: [log_return, vol_5d]
        State mapping: argsort by vol mean
    end note
```

---

## 7. Deployment Topology

Infrastructure layout for the production system.

```mermaid
flowchart TB
    subgraph CRON["⏰ Scheduler"]
        CR["cron / Task Scheduler\n5:15 PM ET Mon–Fri"]
    end

    subgraph HOST["🖥️ Heroku Eco Dyno (Ephemeral)"]
        PY["Python 3.11+\ndaily_loop.py"]
        MA["/tmp/modelartifacts\n(Memory Buffer)"]
        WEB["gunicorn health:app\n/health endpoint"]
        LG["logs/\nDaily rotated"]
    end

    subgraph DOCKER["🐳 Docker (Optional)"]
        DC["docker-compose.yml\npython + postgres"]
    end

    subgraph PGDB["🐘 PostgreSQL"]
        PG["quant_eod\n12 tables\n2 schema files"]
    end

    subgraph APIS["☁️ External"]
        direction LR
        A1["OANDA V20"]
        A2["FRED"]
        A3["Perplexity"]
        A4["Discord"]
    end

    CR -->|triggers| PY
    PY --> MA
    PY --> LG
    WEB --> PG
    CR -->|triggers| PY
    PY <-->|read/write| PG
    PY -->|REST calls| A1 & A2 & A3
    PY -->|webhook POST| A4
    DOCKER -.->|alternative| PY & PG

    classDef sched fill:#1a1a2e,stroke:#e94560,color:#eee
    classDef host fill:#16213e,stroke:#0f3460,color:#eee
    classDef db fill:#0f3460,stroke:#53a8b6,color:#eee
    classDef api fill:#0a3d62,stroke:#38ada9,color:#eee

    class CR sched
    class PY,MA,LG,WEB host
    class PG db
    class A1,A2,A3,A4 api
```

---

## 8. Live Stream & Execution Architecture

Decoupled, event-driven architecture for simulated and live trade execution utilizing Redis streams and dockerized services.

```mermaid
flowchart TB
    subgraph SIM["📼 LIVE STREAM SIMULATOR"]
        DB_FILES["Databento CME Ticks\n.csv.zst files"]
        WF_JSON["walkforward_results.json\nDaily predictions/regimes"]
        SIM_RUN["databento_stream_simulator.py\n(Replays trades & models)"]
    end

    subgraph REDIS["⚡ REDIS (IN-MEMORY STREAMS)"]
        R_REGIME["Key: EUR_USD:regime\n(Dynamic rollover)"]
        R_MODEL["Key: EUR_USD:metamodel\n(Dynamic rollover)"]
        R_ATR["Key: EUR_USD:daily_atr\n(Dynamic rollover)"]
        R_TICKS[("Stream: ticks:EUR_USD\n(Raw bid/ask/time ticks)")]
        R_COND_TICKS[("Stream: cond_ticks:EUR_USD\n(Filtered bid/ask + Kalman velocity)")]
    end

    subgraph WORKER["⚙️ PHYSICS ENGINE CONTAINER"]
        P_WORKER["physics/worker.py\n(Kalman filter worker)"]
        P_STATE["Redis Key: kalman_state:EUR_USD\n(Crash-resilience)"]
    end

    subgraph TRADER["🛡️ LIVE TRADING CONTAINER"]
        LT_RUN["execution/live_trader.py\n(Continuous event loop)"]
        RM["Risk & Sizing\nKelly Sizing + 2% Drawdown Guard"]
        OM["Order Manager\nDry-run/Live OANDA API REST"]
    end

    subgraph DB["🗄️ DATABASE"]
        LT_LOG[("live_trades\nEntries, exits, pips, and PnL")]
    end

    %% Simulator flow
    DB_FILES --> SIM_RUN
    WF_JSON --> SIM_RUN
    SIM_RUN -->|Updates daily models| R_REGIME & R_MODEL & R_ATR
    SIM_RUN -->|Streams ticks| R_TICKS

    %% Physics worker flow
    R_TICKS --> P_WORKER
    P_WORKER <-->|Persists state| P_STATE
    P_WORKER -->|Publishes conditioned ticks| R_COND_TICKS

    %% Live trader flow
    R_COND_TICKS --> LT_RUN
    LT_RUN <-->|Reads daily model state| R_REGIME & R_MODEL & R_ATR
    LT_RUN --> RM
    LT_RUN --> OM
    OM -->|Writes trade logs| LT_LOG
    RM -->|Queries realized PnL| LT_LOG

    %% Styling
    classDef sim fill:#1a1a2e,stroke:#e94560,color:#eee,stroke-width:2px
    classDef redis fill:#2d132c,stroke:#ee4540,color:#eee,stroke-width:2px
    classDef worker fill:#16213e,stroke:#0f3460,color:#eee
    classDef trader fill:#0a3d62,stroke:#38ada9,color:#eee
    classDef db fill:#0f3460,stroke:#53a8b6,color:#eee

    class SIM_RUN,DB_FILES,WF_JSON sim
    class R_REGIME,R_MODEL,R_ATR,R_TICKS,R_COND_TICKS redis
    class P_WORKER,P_STATE worker
    class LT_RUN,RM,OM trader
    class LT_LOG db
```

---

## Diagram Index

| # | Diagram | Shows |
|:-:|---------|-------|
| 1 | **End-to-End Pipeline** | Complete 13-step flow from APIs to Discord, through DB and engine |
| 2 | **Model Architecture** | XGBoost internals: features → scaler → classifier → probability gate → signal veto |
| 3 | **Signal Cascade** | 4 Tier 1 + 4 Tier 2 → composite voting → strength floor |
| 4 | **Data Flow & Storage** | All read/write relationships between modules and 12 DB tables |
| 5 | **Backtest Execution** | Sequence diagram of T/T+1 timing, PnL formula, and equity update |
| 6 | **HMM Regime States** | State machine with transitions and per-state parameters |
| 7 | **Deployment Topology** | Infrastructure: cron → Python → PostgreSQL → APIs → Discord |
| 8 | **Live Replay Architecture** | Decoupled streaming flow: simulator → Redis → physics engine → live trader → database |
