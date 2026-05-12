-- Gamification tables required by backend_gamification (routers/gamification.py).
-- Mirrors the SQLModel definitions in backend/app/models/base.py.

CREATE TABLE IF NOT EXISTS gamification_data (
    id                       SERIAL PRIMARY KEY,
    user_id                  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    xp                       INTEGER NOT NULL DEFAULT 0,
    current_streak           INTEGER NOT NULL DEFAULT 0,
    longest_streak           INTEGER NOT NULL DEFAULT 0,
    last_activity            TIMESTAMPTZ,
    last_login_at            TIMESTAMPTZ,

    plants_added             INTEGER NOT NULL DEFAULT 0,
    plants_scanned           INTEGER NOT NULL DEFAULT 0,
    plants_scanned_not_added INTEGER NOT NULL DEFAULT 0,
    plants_watered           INTEGER NOT NULL DEFAULT 0,
    care_tasks_completed     INTEGER NOT NULL DEFAULT 0,
    species_owned            INTEGER NOT NULL DEFAULT 0,
    species_scanned          INTEGER NOT NULL DEFAULT 0,
    waters_before_9am        INTEGER NOT NULL DEFAULT 0,

    flags                    JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_gamification_data_user_id ON gamification_data(user_id);

CREATE TABLE IF NOT EXISTS achievement_data (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    achievement_name VARCHAR(100) NOT NULL,
    UNIQUE (user_id, achievement_name)
);

CREATE INDEX IF NOT EXISTS idx_achievement_data_user_id ON achievement_data(user_id);
