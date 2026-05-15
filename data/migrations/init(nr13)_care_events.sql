-- Generalize watering_data into a multi-type care event log. Adds a care_type
-- column with a default of 'water' so existing rows keep their meaning, and
-- introduces an enum covering the activities the UI lets users log.
--
-- The table name stays as watering_data for backward compat; new code reads it
-- as a generic care-event table.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'care_type') THEN
        CREATE TYPE care_type AS ENUM (
            'water',
            'mist',
            'fertilize',
            'prune',
            'rotate',
            'inspect',
            'other'
        );
    END IF;
END $$;

ALTER TABLE watering_data
    ADD COLUMN IF NOT EXISTS care_type care_type NOT NULL DEFAULT 'water';

-- Existing rows are already 'water' via the DEFAULT, but make it explicit for
-- rows that may have been inserted by a partial earlier run.
UPDATE watering_data SET care_type = 'water' WHERE care_type IS NULL;

CREATE INDEX IF NOT EXISTS idx_watering_plant_type_timestamp
    ON watering_data(plant_id, care_type, timestamp_of_watering DESC);
