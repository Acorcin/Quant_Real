-- Add EUR–USD spread momentum columns (existing databases only; fresh installs use schema.sql)
ALTER TABLE yield_data ADD COLUMN IF NOT EXISTS spread_change_5d_bps NUMERIC(8, 2);
ALTER TABLE yield_data ADD COLUMN IF NOT EXISTS spread_change_20d_bps NUMERIC(8, 2);
