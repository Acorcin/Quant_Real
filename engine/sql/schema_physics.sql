-- ============================================================
-- Quant EOD Engine — Live Trading Execution Schema
-- ============================================================

CREATE TABLE IF NOT EXISTS live_trades (
    id              SERIAL PRIMARY KEY,
    ticket_id       VARCHAR(50) UNIQUE,
    instrument      VARCHAR(10) NOT NULL,
    direction       VARCHAR(10) NOT NULL,       -- long, short
    entry_time      TIMESTAMPTZ NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    position_size   INTEGER NOT NULL,           -- in units (e.g. 10000)
    exit_time       TIMESTAMPTZ,
    exit_price      DOUBLE PRECISION,
    pnl_pips        DOUBLE PRECISION,
    pnl_amount      DOUBLE PRECISION,
    exit_reason     VARCHAR(50),                -- take_profit, stop_loss, signal_reversal, risk_drawdown
    regime_state    INTEGER,
    model_version   VARCHAR(50),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_live_trades_lookup
    ON live_trades (instrument, entry_time DESC);
