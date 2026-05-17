-- Snapshot of the AI disease classifier result captured when a plant is
-- added (or last re-scanned). Used on the plant detail screen to show the
-- user the health verdict without re-running inference on every page open.

ALTER TABLE user_plants
ADD COLUMN IF NOT EXISTS last_health_label VARCHAR(20),
ADD COLUMN IF NOT EXISTS last_health_confidence DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS last_health_check_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS last_diseases JSONB;
