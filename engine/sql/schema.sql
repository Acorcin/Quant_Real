-- ============================================================
-- Quant EOD Engine — PostgreSQL Schema
-- Run once: psql -U postgres -d quant_eod -f schema.sql
-- ============================================================

-- Daily OHLCV bars (daily + 4H)
CREATE TABLE IF NOT EXISTS bars (
    id              SERIAL PRIMARY KEY,
    instrument      VARCHAR(10) NOT NULL,       -- EUR_USD, GBP_USD, USD_JPY
    granularity     VARCHAR(5) NOT NULL,         -- D, H4
    bar_time        TIMESTAMPTZ NOT NULL,
    open            NUMERIC(10, 6) NOT NULL,
    high            NUMERIC(10, 6) NOT NULL,
    low             NUMERIC(10, 6) NOT NULL,
    close           NUMERIC(10, 6) NOT NULL,
    volume          INTEGER NOT NULL DEFAULT 0,
    complete        BOOLEAN NOT NULL DEFAULT TRUE,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (instrument, granularity, bar_time)
);

CREATE INDEX IF NOT EXISTS idx_bars_lookup
    ON bars (instrument, granularity, bar_time DESC);


-- Yield spread data (FRED)
CREATE TABLE IF NOT EXISTS yield_data (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    us_2y_yield     NUMERIC(6, 4),               -- e.g. 3.8570
    de_2y_yield     NUMERIC(6, 4),               -- e.g. 2.6070
    yield_spread_bps NUMERIC(8, 2),              -- e.g. 125.00 (US 2Y - DE 2Y, bps)
    spread_change_5d_bps NUMERIC(8, 2),          -- change in spread vs ~5 obs ago
    spread_change_20d_bps NUMERIC(8, 2),           -- change in spread vs ~20 obs ago
    source          VARCHAR(20) NOT NULL DEFAULT 'fred',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, source)
);


-- OANDA retail sentiment / position ratios
CREATE TABLE IF NOT EXISTS sentiment (
    id              SERIAL PRIMARY KEY,
    instrument      VARCHAR(10) NOT NULL,
    date            DATE NOT NULL,
    pct_long        NUMERIC(5, 4),               -- e.g. 0.6800
    pct_short       NUMERIC(5, 4),               -- e.g. 0.3200
    long_short_ratio NUMERIC(6, 3),              -- e.g. 2.125
    source          VARCHAR(30) NOT NULL DEFAULT 'oanda_position_ratios',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (instrument, date, source)
);


-- Swap / rollover rates
CREATE TABLE IF NOT EXISTS swap_rates (
    id              SERIAL PRIMARY KEY,
    instrument      VARCHAR(10) NOT NULL,
    date            DATE NOT NULL,
    long_swap_pips  NUMERIC(8, 4),               -- e.g. -0.5200
    short_swap_pips NUMERIC(8, 4),               -- e.g.  0.3100
    source          VARCHAR(20) NOT NULL DEFAULT 'oanda',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (instrument, date, source)
);


-- Economic calendar events
CREATE TABLE IF NOT EXISTS calendar_events (
    id              SERIAL PRIMARY KEY,
    event_name      VARCHAR(200) NOT NULL,
    currency        VARCHAR(5) NOT NULL,          -- USD, EUR, GBP, JPY
    impact          VARCHAR(10) NOT NULL,         -- high, medium, low
    event_time      TIMESTAMPTZ NOT NULL,
    forecast        VARCHAR(50),
    previous        VARCHAR(50),
    actual          VARCHAR(50),
    surprise_direction VARCHAR(20),               -- positive_usd, negative_usd, neutral, etc.
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (event_name, event_time)
);


-- Perplexity AI macro sentiment scores
CREATE TABLE IF NOT EXISTS ai_sentiment (
    id                      SERIAL PRIMARY KEY,
    date                    DATE NOT NULL,
    macro_sentiment_score   NUMERIC(4, 3),       -- -1.000 to 1.000
    confidence              NUMERIC(4, 3),       -- 0.000 to 1.000
    dominant_driver         VARCHAR(50),
    key_events              JSONB,                -- array of event strings
    rationale               TEXT,
    fed_stance              VARCHAR(20),          -- hawkish, neutral, dovish
    ecb_stance              VARCHAR(20),
    risk_sentiment          VARCHAR(20),          -- risk_on, neutral, risk_off
    sources_consulted       INTEGER,
    model_used              VARCHAR(30),
    fallback_used           BOOLEAN DEFAULT FALSE,
    raw_response            JSONB,                -- full API response for audit
    fetched_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date)
);


-- Daily snapshots (the assembled feature vector per day)
CREATE TABLE IF NOT EXISTS daily_snapshots (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    instrument      VARCHAR(10) NOT NULL,
    snapshot_data   JSONB NOT NULL,               -- full daily_snapshot JSON
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, instrument)
);


-- Pipeline run log (track every daily execution)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              SERIAL PRIMARY KEY,
    run_date        DATE NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          VARCHAR(20) NOT NULL DEFAULT 'running',  -- running, success, partial, failed
    steps_completed JSONB,                        -- {"oanda_bars": true, "fred_yields": true, ...}
    errors          JSONB,                        -- any errors encountered
    UNIQUE (run_date)
);
