--ale najpierw usuwanie, ostroznie!!!!!
DROP TABLE IF EXISTS watering_data CASCADE;
DROP TABLE IF EXISTS user_plants CASCADE;
DROP TABLE IF EXISTS levels_xp_ranges CASCADE;
DROP TABLE IF EXISTS plants_catalog CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 2. Usuwamy typy (CASCADE wyczyści powiązania, jeśli jakieś zostały)
DROP TYPE IF EXISTS light_level CASCADE;
DROP TYPE IF EXISTS humidity_level CASCADE;






---------tworzenie-----------

CREATE TYPE light_level AS ENUM ('low', 'medium', 'high');
CREATE TYPE humidity_level AS ENUM ('low', 'medium', 'high');


-- 1. TABELA UŻYTKOWNIKÓW
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,
    --
    -- Statystyki konta
    --level INTEGER DEFAULT 1,
    --level zrobi sie potem xd
    xp INTEGER DEFAULT 0,
    day_streak INTEGER DEFAULT 0,
    last_login_at TIMESTAMP WITH TIME ZONE,
    --
    -- Lokalizacja i ustawienia
    location_city VARCHAR(100),
    --
    -- JSONB przechowuje: dark_mode, water_reminders, weather_tips (bool)
    preferences JSONB DEFAULT '{"dark_mode": false, "care_reminders": true, "weather_tips": true}'::jsonb,
    --
    -- Metadane i Soft Delete
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
);


-- 2. TABELA KATALOGU ROŚLIN (Zdefiniowane gatunki)
CREATE TABLE plants_catalog (
    id SERIAL PRIMARY KEY,
    common_name VARCHAR(150) NOT NULL, --popolskiemu
    scientific_name VARCHAR(150), --lacina
    --
    -- Parametry optymalne
    preferred_sunlight light_level NOT NULL,
    preferred_temp_min INTEGER, -- w stopniach Celsjusza
    preferred_temp_max INTEGER,
    air_humidity_req humidity_level,
    soil_humidity_req humidity_level,
    --
    --prefered_watering_interval_days INTEGER
    prefered_watering_interval_days INTEGER CHECK (prefered_watering_interval_days > 0)
);


-- 3. TABELA ROŚLIN UŻYTKOWNIKÓW (Konkretne egzemplarze/ biblioteka roslin uzytkownikow)
CREATE TABLE user_plants (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    plant_catalog_id INTEGER REFERENCES plants_catalog(id),
    --
    custom_name VARCHAR(100), -- np. "Paprotka Zenka, chyba martyniuka"
    note TEXT,
    photo_url TEXT,
    --
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    age DATE
);


--to do trigger do usuwania starszych niz 7 dni, bha bhahahaha
CREATE TABLE watering_data (
    plant_id INTEGER REFERENCES user_plants(id),
    timestamp_of_watering TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


--tabela z poziomami
CREATE TABLE levels_xp_ranges (
    level_val INTEGER,
    req_xp INTEGER
);

/*
INSERT INTO levels_xp_ranges (level_val, req_xp) VALUES
(1, 0),      -- Poziom 1 zaczyna się od 0 XP
(2, 100),    -- Poziom 2: +100 XP (Suma: 100)
(3, 300),    -- Poziom 3: +200 XP (Suma: 300)
(4, 600),    -- Poziom 4: +300 XP (Suma: 600)
(5, 1000),   -- Poziom 5: +400 XP (Suma: 1000)
(6, 1500),   -- Poziom 6: +500 XP (Suma: 1500)
(7, 2100),   -- Poziom 7: +600 XP (Suma: 2100)
(8, 2800),   -- Poziom 8: +700 XP (Suma: 2800)
(9, 3600),   -- Poziom 9: +800 XP (Suma: 3600)
(10, 1000000);  -- Poziom 10: +duzo XP (Suma: miliun)
*/
