-- Multiple photos per user plant. Replaces the single-image model where each
-- user_plants row stored one fid; that column is kept for backward compat but
-- new uploads go through this table.

CREATE TABLE IF NOT EXISTS user_plant_images (
    id            SERIAL PRIMARY KEY,
    user_plant_id INTEGER NOT NULL REFERENCES user_plants(id) ON DELETE CASCADE,
    fid           VARCHAR(255) NOT NULL,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_plant_images_user_plant_id
    ON user_plant_images(user_plant_id, uploaded_at DESC);
