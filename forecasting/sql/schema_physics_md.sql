-- ============================================================================
-- Physics layer on the Databento path: per-tick conditioned trades.
--
-- The engine's PhysicsEngine (rolling-median spike rejection + 2D Kalman
-- price/velocity filter + scale-normalized, regime-clipped returns) was built
-- for OANDA FX ticks over Redis. This schema is its new home on the CME path:
-- ticks come from the same free Databento polling the other feeds use, and
-- conditioned output + filter state live in Postgres like everything else.
-- Idempotent; applied alongside the other md migrations.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS md;

-- ── conditioned ticks ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS md.cond_ticks (
    tick_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    instrument       TEXT             NOT NULL,   -- tradable, e.g. M6EU6
    ts               TIMESTAMPTZ      NOT NULL,   -- exchange ts_event
    price            DOUBLE PRECISION NOT NULL,   -- raw trade price
    size             INTEGER          NOT NULL,
    filtered_price   DOUBLE PRECISION NOT NULL,   -- after spike rejection
    is_spike         BOOLEAN          NOT NULL,
    kalman_price     DOUBLE PRECISION NOT NULL,
    kalman_velocity  DOUBLE PRECISION NOT NULL,   -- price units / second
    tick_return      DOUBLE PRECISION NOT NULL,   -- of the Kalman estimate
    normalized_return DOUBLE PRECISION NOT NULL,  -- / daily scale
    clipped_return   DOUBLE PRECISION NOT NULL    -- regime-aware z-clip
);
CREATE INDEX IF NOT EXISTS idx_md_cond_ticks
    ON md.cond_ticks (instrument, ts);

-- ── worker state: Kalman/spike checkpoint + watermark (replaces Redis) ───────
CREATE TABLE IF NOT EXISTS md.physics_state (
    instrument   TEXT PRIMARY KEY,
    state        JSONB       NOT NULL,   -- PhysicsEngine.get_state()
    watermark    TIMESTAMPTZ NOT NULL,   -- trades processed through here
    regime       TEXT,
    daily_scale  DOUBLE PRECISION,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── latest physics read for dashboards / the engine ─────────────────────────
-- One row per instrument: newest Kalman state + trailing 15-min spike rate.
CREATE OR REPLACE VIEW md.v_physics_latest AS
WITH latest AS (
    SELECT DISTINCT ON (instrument)
           instrument, ts, price, kalman_price, kalman_velocity,
           clipped_return
    FROM md.cond_ticks
    ORDER BY instrument, ts DESC
),
spikes AS (
    SELECT c.instrument,
           count(*) FILTER (WHERE c.is_spike)::float8
             / greatest(count(*), 1)                AS spike_rate_15m,
           count(*)                                 AS ticks_15m
    FROM md.cond_ticks c
    JOIN latest l USING (instrument)
    WHERE c.ts > l.ts - interval '15 minutes'
    GROUP BY c.instrument
)
SELECT l.instrument, l.ts, l.price, l.kalman_price, l.kalman_velocity,
       l.clipped_return, s.spike_rate_15m, s.ticks_15m,
       p.regime, p.daily_scale, p.watermark
FROM latest l
LEFT JOIN spikes s USING (instrument)
LEFT JOIN md.physics_state p USING (instrument);
