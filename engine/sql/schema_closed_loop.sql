-- ============================================================
-- Quant EOD Engine — Closed Loop Architecture Schema
-- ============================================================

-- Table for Layer 3 (Event Extraction)
CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    instrument      VARCHAR(10) NOT NULL,
    event_type      VARCHAR(50) NOT NULL,       -- e.g., 'composite_signal'
    direction       VARCHAR(10) NOT NULL,       -- long, short, flat
    magnitude       NUMERIC(5, 4) NOT NULL,     -- composite strength
    confidence      NUMERIC(5, 4) NOT NULL,     -- metamodel probability
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, instrument, event_type)
);

CREATE INDEX IF NOT EXISTS idx_events_lookup 
    ON events (instrument, date DESC);

-- Table for Layer 4 (Opportunity Measurement)
CREATE TABLE IF NOT EXISTS opportunity_measurements (
    id                  SERIAL PRIMARY KEY,
    event_id            INTEGER REFERENCES events(id) ON DELETE CASCADE,
    date                DATE NOT NULL,          -- Event date (T)
    trade_date          DATE NOT NULL,          -- Execution date (T+1)
    instrument          VARCHAR(10) NOT NULL,
    direction           VARCHAR(10) NOT NULL,
    entry_price         NUMERIC(10, 6) NOT NULL,
    exit_price          NUMERIC(10, 6) NOT NULL,
    mfe_pips            NUMERIC(8, 2) NOT NULL, -- Max favorable excursion
    mae_pips            NUMERIC(8, 2) NOT NULL, -- Max adverse excursion
    optimal_return_pips NUMERIC(8, 2) NOT NULL, -- Max potential profit (pips)
    close_return_pips   NUMERIC(8, 2) NOT NULL, -- Return at end of day (pips)
    path_ratio          NUMERIC(5, 4) NOT NULL, -- MFE / (MFE + |MAE|)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (event_id)
);

CREATE INDEX IF NOT EXISTS idx_opp_measurements_lookup 
    ON opportunity_measurements (instrument, date DESC);

-- Table for Layer 8 (System Learning)
CREATE TABLE IF NOT EXISTS learning_runs (
    id                  SERIAL PRIMARY KEY,
    date                DATE NOT NULL,
    instrument          VARCHAR(10) NOT NULL,
    metrics             JSONB NOT NULL,         -- win_rate_20d, avg_slippage, etc.
    adjusted_parameters JSONB NOT NULL,         -- new thresholds, sizing multipliers
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, instrument)
);

CREATE INDEX IF NOT EXISTS idx_learning_runs_lookup 
    ON learning_runs (instrument, date DESC);
