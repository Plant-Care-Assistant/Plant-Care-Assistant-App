CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL PRIMARY KEY,
    temperature NUMERIC(5, 2),          -- Obsłuży np. -20.55
    humidity NUMERIC(5, 2),             -- Obsłuży np. 99.99
    humidity_lvl humidity_level,        -- Typ ENUM: 'low', 'medium', 'high'
    light_lvl light_level,              -- Typ ENUM: 'low', 'medium', 'high'
    measured_at TIMESTAMPTZ NOT NULL    -- Czas z Azure Event Hub
);

-- Indeks dla wydajności przy pobieraniu danych historycznych
CREATE INDEX IF NOT EXISTS idx_measured_at ON sensor_data (measured_at DESC);