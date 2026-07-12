-- ============================================================
-- Quant EOD Engine — Phase 2 Schema (Prediction Engine)
-- Run after Phase 1 schema:
--   psql -U postgres -d quant_eod -f schema_phase2.sql
-- ============================================================

-- HMM regime classifications
CREATE TABLE IF NOT EXISTS regimes (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    instrument      VARCHAR(10) NOT NULL,
    state_id        INTEGER NOT NULL,          -- 0, 1, 2
    state_label     VARCHAR(30) NOT NULL,       -- low_vol_trend, high_vol_choppy, high_vol_crash
    confidence      NUMERIC(5, 4),              -- posterior probability of assigned state
    days_in_regime  INTEGER NOT NULL DEFAULT 1,
    transition_prob JSONB,                      -- row of transition matrix for current state
    model_version   VARCHAR(50),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, instrument)
);

CREATE INDEX IF NOT EXISTS idx_regimes_lookup
    ON regimes (instrument, date DESC);


-- Daily signal outputs (Tier 1 + Tier 2)
CREATE TABLE IF NOT EXISTS signals (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    instrument      VARCHAR(10) NOT NULL,
    tier            INTEGER NOT NULL,           -- 1 or 2
    detector        VARCHAR(50) NOT NULL,       -- yield_spread_momentum, sentiment_extreme_fade, etc.
    direction       VARCHAR(10),                -- long, short, flat
    strength        NUMERIC(5, 4),              -- 0.0 to 1.0
    detail          TEXT,
    metadata        JSONB,                      -- detector-specific data
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, instrument, detector)
);

CREATE INDEX IF NOT EXISTS idx_signals_lookup
    ON signals (instrument, date DESC, tier);


-- Daily feature vectors (input to meta-model)
CREATE TABLE IF NOT EXISTS feature_vectors (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    instrument      VARCHAR(10) NOT NULL,
    features        JSONB NOT NULL,             -- full feature dict
    label           INTEGER,                    -- 1=profitable, 0=not (filled next day)
    label_return_pips NUMERIC(8, 2),            -- actual next-day return in pips
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, instrument)
);

CREATE INDEX IF NOT EXISTS idx_feature_vectors_lookup
    ON feature_vectors (instrument, date DESC);


-- Meta-model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    prediction_for  DATE NOT NULL,              -- T+1 date
    instrument      VARCHAR(10) NOT NULL,
    direction       VARCHAR(10) NOT NULL,       -- long, short, flat
    probability     NUMERIC(5, 4),              -- 0.0 to 1.0
    size_multiplier NUMERIC(4, 3),              -- 0.0, 0.5, or 1.0
    model_version   VARCHAR(50),
    top_shap        JSONB,                      -- top 5 SHAP feature contributions
    regime_state    INTEGER,
    composite_strength NUMERIC(5, 4),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, instrument)
);

CREATE INDEX IF NOT EXISTS idx_predictions_lookup
    ON predictions (instrument, date DESC);


-- Model training runs (CPCV results, SHAP snapshots)
CREATE TABLE IF NOT EXISTS model_runs (
    id              SERIAL PRIMARY KEY,
    run_date        DATE NOT NULL,
    model_type      VARCHAR(30) NOT NULL,       -- xgboost, hmm
    model_version   VARCHAR(50) NOT NULL,
    training_samples INTEGER,
    cpcv_results    JSONB,                      -- sharpe_mean, sharpe_std, p-value, PSR, path stats
    shap_importance JSONB,                      -- ordered feature importance
    hyperparams     JSONB,
    metrics         JSONB,                      -- accuracy, precision, recall, f1
    model_path      VARCHAR(200),               -- path to serialized model file
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
