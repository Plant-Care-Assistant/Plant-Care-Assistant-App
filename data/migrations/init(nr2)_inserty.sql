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


-- ==========================================
-- 1. WYPEŁNIANIE KATALOGU ROŚLIN (PLANTS_CATALOG)
-- ==========================================
INSERT INTO plants_catalog
(common_name, scientific_name, preferred_sunlight, preferred_temp_min, preferred_temp_max, air_humidity_req, soil_humidity_req, prefered_watering_interval_days)
VALUES
('Monstera Dziurawa', 'Monstera deliciosa', 'medium', 18, 30, 'high', 'medium', 7),
('Sansewieria (Wężownica)', 'Sansevieria trifasciata', 'low', 15, 28, 'low', 'low', 14),
('Skrzydłokwiat', 'Spathiphyllum', 'low', 18, 25, 'high', 'high', 4),
('Sukulenty Mix', 'Succulentus', 'high', 20, 35, 'low', 'low', 20),
('Paproć Domowa', 'Nephrolepis exaltata', 'medium', 18, 24, 'high', 'high', 3),
('Zamiokulkas', 'Zamioculcas zamiifolia', 'low', 16, 26, 'low', 'low', 12);

-- ==========================================
-- 2. TWORZENIE UŻYTKOWNIKÓW (USERS)
-- ==========================================
INSERT INTO users (email, password_hash, username, xp, day_streak, location_city, preferences) VALUES
('jan.kowalski@example.com', 'hash_haslo_123', 'JanuszOgrodnik', 150, 5, 'Warszawa', '{"dark_mode": true, "care_reminders": true, "weather_tips": false}'::jsonb),
('anna.nowak@example.com', 'hash_haslo_456', 'AniaZZielonegoWzg', 1200, 45, 'Kraków', '{"dark_mode": false, "care_reminders": true, "weather_tips": true}'::jsonb),
('test.user@example.com', 'hash_haslo_789', 'TesterBazy', 0, 0, 'Gdańsk', '{"dark_mode": true, "care_reminders": false, "weather_tips": false}'::jsonb);

-- ==========================================
-- 3. PRZYPISYWANIE ROŚLIN DO UŻYTKOWNIKÓW (USER_PLANTS)
-- ==========================================
-- Zakładamy, że ID z SERIAL pójdą po kolei (1, 2, 3...)

-- Użytkownik 1 (Janusz) ma Monsterę i Paprotkę
INSERT INTO user_plants (user_id, plant_catalog_id, custom_name, note, created_at, age) VALUES
(1, 1, 'Potwór w salonie', 'Stoi przy oknie, szybko rośnie', NOW() - INTERVAL '2 months', '2023-05-01'),
(1, 5, 'Paprotka Zenka', 'Ta co zawsze usycha', NOW() - INTERVAL '1 year', '2022-01-01');

-- Użytkownik 2 (Ania) ma Sansewierię i Zamiokulkasa (dużo XP, zadbane rośliny)
INSERT INTO user_plants (user_id, plant_catalog_id, custom_name, note, created_at, age) VALUES
(2, 2, 'Żelazna Dama', 'Nie podlewać za często!', NOW() - INTERVAL '6 months', '2023-01-15'),
(2, 6, 'Zamek', 'Prezent od cioci', NOW() - INTERVAL '3 months', '2023-08-10');

-- Użytkownik 3 (Tester) ma Sukulenta
INSERT INTO user_plants (user_id, plant_catalog_id, custom_name, note, created_at, age) VALUES
(3, 4, 'Kaktus Testowy', 'Brak uwag', NOW(), '2024-01-01');

-- ==========================================
-- 4. HISTORIA PODLEWANIA (WATERING_DATA)
-- ==========================================
-- Tutaj wrzucamy dane do testowania triggera usuwającego stare wpisy.

-- Roślina ID 1 (Monstera Janusza):
INSERT INTO watering_data (plant_id, timestamp_of_watering) VALUES
(1, NOW() - INTERVAL '1 day'),   -- Wczoraj (powinno zostać)
(1, NOW() - INTERVAL '4 days'),  -- 4 dni temu (powinno zostać)
(1, NOW() - INTERVAL '8 days'),  -- 8 dni temu (DO USUNIĘCIA przez Twój przyszły trigger)
(1, NOW() - INTERVAL '15 days'); -- 15 dni temu (DO USUNIĘCIA)

-- Roślina ID 3 (Sansewieria Ani):
INSERT INTO watering_data (plant_id, timestamp_of_watering) VALUES
(3, NOW() - INTERVAL '10 hours'), -- Dzisiaj
(3, NOW() - INTERVAL '14 days');  -- Dawno temu (DO USUNIĘCIA)

-- Roślina ID 5 (Kaktus Testera) - jeszcze nie podlewana (brak wpisu)
