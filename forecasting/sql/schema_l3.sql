-- ============================================================================
-- L3 microstructure layer: periodic intraday MBO polls -> derived book state
-- and VPIN buckets. Raw MBO increments stay in the file lake (DBN); Postgres
-- holds only what downstream features query.
--
-- Feeds the meta-model's l3_order_book_imbalance / l3_vpin columns.
-- Idempotent; applied alongside schema_market.sql.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS md;

-- ── poll log: one row per Historical-API pull (operational audit) ───────────
CREATE TABLE IF NOT EXISTS md.l3_polls (
    poll_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    instrument  TEXT        NOT NULL,      -- parent symbol, e.g. ES.FUT
    ts_start    TIMESTAMPTZ NOT NULL,      -- window pulled [start, end)
    ts_end      TIMESTAMPTZ NOT NULL,
    pulled_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_events    BIGINT      NOT NULL,
    n_trades    BIGINT      NOT NULL,
    quoted_cost DOUBLE PRECISION NOT NULL, -- pre-pull get_cost (guarded ~0)
    raw_file    TEXT,                      -- lake path of the DBN increment
    status      TEXT        NOT NULL DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_md_l3_polls
    ON md.l3_polls (instrument, ts_end DESC);
-- schema the poll pulled: mbp-1 (live loop, ~10min lag) or mbo (deep
-- backfill; Databento's MBO processing runs ~hours behind for GLBX)
ALTER TABLE md.l3_polls ADD COLUMN IF NOT EXISTS schema TEXT NOT NULL DEFAULT 'mbo';

-- ── sampled book state (event-time sampled during replay) ───────────────────
CREATE TABLE IF NOT EXISTS md.l3_book_state (
    instrument   TEXT        NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,     -- event time of the sample
    bid_px       DOUBLE PRECISION,
    ask_px       DOUBLE PRECISION,
    bid_sz       BIGINT,
    ask_sz       BIGINT,
    imbalance_l1 DOUBLE PRECISION,         -- (bid-ask)/(bid+ask) at top level
    imbalance_d10 DOUBLE PRECISION,        -- depth-weighted, top 10 px levels/side
    n_bid_orders INTEGER,
    n_ask_orders INTEGER,
    PRIMARY KEY (instrument, ts)
);

-- ── VPIN volume buckets (volume-synchronized, aggressor-side classified) ────
CREATE TABLE IF NOT EXISTS md.l3_vpin_buckets (
    instrument  TEXT        NOT NULL,
    bucket_seq  BIGINT      NOT NULL,      -- monotone per instrument
    ts_close    TIMESTAMPTZ NOT NULL,      -- time the bucket filled
    bucket_vol  BIGINT      NOT NULL,      -- target volume per bucket
    buy_vol     BIGINT      NOT NULL,      -- aggressor-buy volume
    sell_vol    BIGINT      NOT NULL,
    PRIMARY KEY (instrument, bucket_seq)
);

-- ── latest features per instrument: what the live predict path joins ────────
-- l3_vpin = trailing 50-bucket mean of |buy-sell|/vol  (Easley et al.)
CREATE OR REPLACE VIEW md.v_l3_latest AS
WITH vpin AS (
    SELECT instrument,
           avg(abs(buy_vol - sell_vol)::float8
               / nullif(buy_vol + sell_vol, 0)) AS vpin
    FROM (
        SELECT instrument, buy_vol, sell_vol,
               row_number() OVER (PARTITION BY instrument
                                  ORDER BY bucket_seq DESC) AS rn
        FROM md.l3_vpin_buckets
    ) t
    WHERE rn <= 50
    GROUP BY instrument
),
book AS (
    SELECT DISTINCT ON (instrument)
           instrument, ts, imbalance_l1, imbalance_d10
    FROM md.l3_book_state
    ORDER BY instrument, ts DESC
)
SELECT b.instrument,
       b.ts                       AS book_ts,
       b.imbalance_l1             AS l3_order_book_imbalance,
       b.imbalance_d10            AS l3_depth_imbalance,
       v.vpin                     AS l3_vpin
FROM book b
LEFT JOIN vpin v USING (instrument);
