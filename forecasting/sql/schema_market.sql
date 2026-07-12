-- ============================================================================
-- Quant_Real market-data schema (md): durable home for everything the
-- forecasting plane currently keeps in files and process memory.
--
-- Applies to the engine's existing Postgres (docker-compose: quant_postgres,
-- db quant_eod). Idempotent — safe to re-run.
--
--   psql "$DATABASE_URL" -f forecasting/sql/schema_market.sql
--
-- Provenance model: md.runs is the root; every derived row carries run_id.
-- Bars and conditioned series are keyed by (instrument, bar_size[, vol_window])
-- instead — they are run-independent canonical data, built once, read by all.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS md;

DO $$ BEGIN
    CREATE TYPE md.structure_type AS ENUM
        ('efficient', 'linear', 'vol_only', 'nonlinear');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE md.gate_state AS ENUM ('GO', 'GO_REDUCED', 'VETO');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ── provenance root ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS md.runs (
    run_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    git_sha      TEXT,
    instrument   TEXT NOT NULL,
    config       JSONB NOT NULL,          -- bar_size, vol_window, horizon, step,
                                          -- backend, context_length, num_paths...
    source_file  TEXT,                    -- input trades parquet path
    source_hash  TEXT,                    -- sha256 of the input file
    notes        TEXT
);

-- ── canonical volume bars (run-independent) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS md.volume_bars (
    instrument   TEXT        NOT NULL,
    bar_size     INTEGER     NOT NULL,
    bar_seq      INTEGER     NOT NULL,     -- 0-based, contiguous
    ts_close     TIMESTAMPTZ NOT NULL,     -- strictly increasing per (instr, size)
    open         DOUBLE PRECISION NOT NULL,
    high         DOUBLE PRECISION NOT NULL,
    low          DOUBLE PRECISION NOT NULL,
    close        DOUBLE PRECISION NOT NULL,
    volume       BIGINT      NOT NULL,
    PRIMARY KEY (instrument, bar_size, bar_seq)
);
CREATE INDEX IF NOT EXISTS idx_md_bars_ts
    ON md.volume_bars (instrument, bar_size, ts_close);

-- ── conditioned series: the normalized plane, aligned 1:1 with returns ──────
-- Row (instrument, bar_size, vol_window, seq) holds return index `seq`
-- (realized by bar bar_seq = seq offset in the valid range; ts carried for
-- direct query). sigma is STRICTLY TRAILING (shift-1 rolling std).
CREATE TABLE IF NOT EXISTS md.conditioned_series (
    instrument    TEXT        NOT NULL,
    bar_size      INTEGER     NOT NULL,
    vol_window    INTEGER     NOT NULL,
    seq           INTEGER     NOT NULL,    -- return index, 0-based
    ts            TIMESTAMPTZ NOT NULL,
    raw_return    DOUBLE PRECISION NOT NULL,
    sigma         DOUBLE PRECISION NOT NULL,
    scaled_return DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (instrument, bar_size, vol_window, seq)
);

-- ── forecast archive: every (origin, model, step) with full quantile grid ───
CREATE TABLE IF NOT EXISTS md.forecast_archive (
    run_id      UUID    NOT NULL REFERENCES md.runs(run_id) ON DELETE CASCADE,
    model       TEXT    NOT NULL,
    rung        SMALLINT NOT NULL,
    origin_seq  INTEGER NOT NULL,
    step        SMALLINT NOT NULL,          -- 1..horizon
    target_seq  INTEGER NOT NULL,
    y_true      DOUBLE PRECISION NOT NULL,  -- scaled-return space
    mean        DOUBLE PRECISION NOT NULL,
    quantiles   JSONB   NOT NULL,           -- {"0.01": .., ..., "0.99": ..}
    latency_ms  REAL,
    PRIMARY KEY (run_id, model, origin_seq, step)
);
CREATE INDEX IF NOT EXISTS idx_md_archive_model
    ON md.forecast_archive (run_id, model, step, origin_seq);

-- ── characterization: one verdict per run ───────────────────────────────────
CREATE TABLE IF NOT EXISTS md.characterizations (
    run_id         UUID PRIMARY KEY REFERENCES md.runs(run_id) ON DELETE CASCADE,
    instrument     TEXT NOT NULL,
    data_asof      DATE,
    structure      md.structure_type NOT NULL,
    adf_p          DOUBLE PRECISION,
    kpss_p         DOUBLE PRECISION,
    hurst          DOUBLE PRECISION,
    perm_entropy   DOUBLE PRECISION,
    edge_horizon   INTEGER,
    ladder         JSONB,                  -- per-rung MASE/R2/DMp readings
    regimes        JSONB,                  -- names, current, unpredictable
    breaks         JSONB                   -- structural break indices
);

-- ── gate decisions: the auditable veto record ───────────────────────────────
CREATE TABLE IF NOT EXISTS md.gate_signals (
    run_id       UUID    NOT NULL REFERENCES md.runs(run_id) ON DELETE CASCADE,
    origin_seq   INTEGER NOT NULL,
    ts           TIMESTAMPTZ,
    state        md.gate_state NOT NULL,
    veto_reasons TEXT[]  NOT NULL DEFAULT '{}',
    direction    DOUBLE PRECISION,          -- tanh(mean/vol) in [-1, 1]
    conviction   DOUBLE PRECISION,
    vol_forecast DOUBLE PRECISION,
    regime       TEXT,
    PRIMARY KEY (run_id, origin_seq)
);
CREATE INDEX IF NOT EXISTS idx_md_gate_state
    ON md.gate_signals (run_id, state);

-- ── the bridge, landed: per-origin features the engine consumes ─────────────
CREATE TABLE IF NOT EXISTS md.kronos_features (
    run_id               UUID    NOT NULL REFERENCES md.runs(run_id) ON DELETE CASCADE,
    origin_seq           INTEGER NOT NULL,
    ts                   TIMESTAMPTZ,
    sigma_256            DOUBLE PRECISION,
    y_true_scaled        DOUBLE PRECISION,
    kronos_p50_scaled    DOUBLE PRECISION,
    kronos_spread_scaled DOUBLE PRECISION,
    structure            md.structure_type,
    gate_state           md.gate_state,
    veto_reasons         TEXT[],
    extra                JSONB,             -- per-model p50/spread columns etc.
    PRIMARY KEY (run_id, origin_seq)
);

-- Engine-facing view: latest run per instrument, engine feature names.
-- (engine reads kronos_p50_scaled / kronos_uncertainty / kronos_context_vol /
--  kronos_regime_encoded; efficient=0 drives the structural veto)
CREATE OR REPLACE VIEW md.v_engine_features AS
WITH latest AS (
    SELECT DISTINCT ON (instrument) run_id, instrument
    FROM md.runs ORDER BY instrument, started_at DESC
)
SELECT
    l.instrument,
    k.ts,
    k.kronos_p50_scaled,
    k.kronos_spread_scaled                    AS kronos_uncertainty,
    k.sigma_256                               AS kronos_context_vol,
    CASE k.structure
        WHEN 'efficient' THEN 0 WHEN 'linear' THEN 1
        WHEN 'vol_only'  THEN 2 WHEN 'nonlinear' THEN 3
        ELSE -1 END                           AS kronos_regime_encoded,
    k.gate_state,
    k.run_id,
    k.origin_seq
FROM md.kronos_features k
JOIN latest l USING (run_id);
