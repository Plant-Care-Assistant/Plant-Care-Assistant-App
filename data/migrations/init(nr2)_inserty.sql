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
(common_name, scientific_name, preferred_sunlight, preferred_temp_min, preferred_temp_max, air_humidity_req, soil_humidity_req, preferred_watering_interval_days)
VALUES
-- Pelargonie
('Pelargonia Pachnąca', 'Pelargonium graveolens', 'high', 15, 25, 'low', 'medium', 7),
('Pelargonia Strefowa', 'Pelargonium zonale', 'high', 15, 25, 'low', 'medium', 5),
('Pelargonia Purpurowa', 'Pelargonium inquinans', 'high', 15, 25, 'low', 'medium', 6),
-- Osty
('Ostrożeń Polny', 'Cirsium arvense', 'high', 10, 25, 'medium', 'medium', 7),
('Ostrożeń Zwyczajny', 'Cirsium vulgare', 'high', 10, 25, 'medium', 'medium', 7),
('Ostrożeń Błotny', 'Cirsium palustre', 'medium', 10, 22, 'high', 'high', 5),
('Ostrożeń Wełnisty', 'Cirsium eriophorum', 'high', 10, 24, 'medium', 'medium', 7),
('Ostrożeń Warzywny', 'Cirsium oleraceum', 'medium', 10, 23, 'medium', 'high', 6),
-- Szczawiki i Przywrotek
('Szczawik Roczny', 'Mercurialis annua', 'medium', 12, 25, 'medium', 'medium', 5),
('Szczawik Trwały', 'Mercurialis perennis', 'low', 10, 22, 'medium', 'high', 7),
-- Dziurawce
('Dziurawiec Zwyczajny', 'Hypericum perforatum', 'high', 10, 25, 'medium', 'medium', 7),
('Dziurawiec Kielichowy', 'Hypericum calycinum', 'medium', 12, 24, 'medium', 'medium', 7),
('Dziurawiec Czyszczykowy', 'Hypericum androsaemum', 'medium', 10, 23, 'medium', 'medium', 6),
-- Trzykrotki
('Trzykrotka Rzeczna', 'Tradescantia fluminensis', 'medium', 18, 27, 'medium', 'medium', 5),
('Trzykrotka Różowa', 'Tradescantia spathacea', 'medium', 18, 28, 'medium', 'medium', 6),
('Trzykrotka Wirginijska', 'Tradescantia virginiana', 'medium', 15, 25, 'medium', 'medium', 5),
('Trzykrotka Pręgowana', 'Tradescantia zebrina', 'medium', 18, 27, 'medium', 'medium', 5),
('Trzykrotka Purpurowa', 'Tradescantia pallida', 'high', 18, 28, 'low', 'medium', 6),
-- Jasnoty
('Jasnota Różowa', 'Lamium amplexicaule', 'medium', 10, 24, 'medium', 'medium', 6),
('Jasnota Mieszańcowa', 'Lamium hybridum', 'medium', 10, 23, 'medium', 'medium', 6),
('Jasnota Purpurowa', 'Lamium purpureum', 'medium', 10, 24, 'medium', 'medium', 6),
('Jasnota Żółta', 'Lamium galeobdolon', 'low', 10, 22, 'medium', 'high', 7),
('Jasnota Biała', 'Lamium album', 'medium', 10, 23, 'medium', 'medium', 6),
('Jasnota Plamista', 'Lamium maculatum', 'medium', 10, 23, 'medium', 'medium', 6),
-- Lawendy
('Lawenda Zębata', 'Lavandula dentata', 'high', 15, 28, 'low', 'low', 8),
('Lawenda Francuska', 'Lavandula stoechas', 'high', 15, 28, 'low', 'low', 8),
('Lawenda Wąskolistna', 'Lavandula angustifolia', 'high', 15, 28, 'low', 'low', 9),
-- Nostrzyki
('Nostrzyk Biały', 'Melilotus albus', 'high', 12, 26, 'medium', 'medium', 7),
('Nostrzyk Indyjski', 'Melilotus indicus', 'high', 15, 28, 'low', 'medium', 7),
('Nostrzyk Lekarski', 'Melilotus officinalis', 'high', 12, 26, 'medium', 'medium', 7),
-- Paprocie
('Nerecznica Samcza', 'Dryopteris filix-mas', 'low', 10, 22, 'high', 'high', 4),
-- Chmiel
('Chmiel Zwyczajny', 'Humulus lupulus', 'high', 12, 25, 'medium', 'medium', 6),
-- Nagietek
('Nagietek Lekarski', 'Calendula officinalis', 'high', 15, 25, 'medium', 'medium', 5),
('Nagietek Polny', 'Calendula arvensis', 'high', 15, 26, 'medium', 'medium', 6),
-- Mlecze
('Szczeć Kolczasty', 'Helminthotheca echioides', 'high', 12, 25, 'medium', 'medium', 7),
('Sałata Murowa', 'Lactuca muralis', 'medium', 12, 23, 'medium', 'medium', 6),
('Sałata Uprawna', 'Lactuca sativa', 'high', 15, 25, 'medium', 'high', 3),
('Sałata Kompasowa', 'Lactuca serriola', 'high', 15, 27, 'low', 'medium', 7),
-- Rozchodniki
('Rozchodnik Ostry', 'Sedum acre', 'high', 15, 30, 'low', 'low', 14),
('Rozchodnik Biały', 'Sedum album', 'high', 15, 30, 'low', 'low', 14),
('Rozchodnik Kudłaty', 'Sedum dasyphyllum', 'high', 15, 28, 'low', 'low', 15),
('Rozchodnik Skałowy', 'Sedum sediforme', 'high', 15, 30, 'low', 'low', 14),
('Rozchodnik Skalny', 'Sedum rupestre', 'high', 15, 30, 'low', 'low', 14),
('Rozchodnik Palmera', 'Sedum palmeri', 'high', 15, 28, 'low', 'low', 12),
-- Koniczyny
('Koniczyna Polna', 'Trifolium arvense', 'high', 12, 25, 'medium', 'medium', 6),
('Koniczyna Zwyczajna', 'Trifolium campestre', 'high', 12, 25, 'medium', 'medium', 6),
('Koniczyna Czerwona', 'Trifolium incarnatum', 'high', 12, 25, 'medium', 'medium', 5),
('Koniczyna Łąkowa', 'Trifolium pratense', 'high', 12, 25, 'medium', 'medium', 6),
('Koniczyna Wątpliwa', 'Trifolium dubium', 'high', 12, 25, 'medium', 'medium', 6),
('Koniczyna Biała', 'Trifolium repens', 'high', 12, 25, 'medium', 'medium', 5),
-- Granat
('Granat Właściwy', 'Punica granatum', 'high', 18, 30, 'low', 'medium', 7),
-- Malwy
('Malwa Ogrodowa', 'Alcea rosea', 'high', 15, 28, 'medium', 'medium', 5),
('Prawoślaz Lekarski', 'Althaea officinalis', 'high', 15, 26, 'medium', 'high', 6),
-- Grzybieniem
('Grzybienie Białe', 'Nymphaea alba', 'high', 18, 28, 'high', 'high', 1),
-- Zawilce
('Zawilec Koronkowy', 'Anemone coronaria', 'high', 15, 25, 'medium', 'medium', 5),
('Zawilec Alpejski', 'Anemone alpina', 'medium', 10, 20, 'medium', 'medium', 6),
('Zawilec Wątrobowiec', 'Anemone hepatica', 'low', 10, 22, 'medium', 'high', 7),
('Zawilec Gajowy', 'Anemone nemorosa', 'low', 10, 22, 'medium', 'high', 6),
('Zawilec Zwyczajny', 'Anemone pulsatilla', 'high', 12, 24, 'low', 'medium', 7),
('Zawilec Chiński', 'Anemone hupehensis', 'medium', 15, 25, 'medium', 'medium', 5),
-- Wawrzynek
('Wawrzynek Smukły', 'Daphne gnidium', 'high', 15, 28, 'low', 'medium', 8),
('Wawrzynek Wilczełyko', 'Daphne laureola', 'medium', 10, 24, 'medium', 'medium', 7),
('Wawrzynek Wilczełyko Zwyczajne', 'Daphne mezereum', 'medium', 10, 23, 'medium', 'medium', 7),
-- Trybula
('Trybula Ogrodowa', 'Chaerophyllum temulum', 'medium', 12, 23, 'medium', 'medium', 6),
-- Pancratium
('Lilia Morska', 'Pancratium maritimum', 'high', 18, 30, 'low', 'medium', 10),
-- Złoć
('Złoć Liliowata', 'Anthericum liliago', 'high', 15, 28, 'low', 'medium', 8),
-- Storczyki
('Storczyk Pszczołowaty', 'Ophrys apifera', 'medium', 15, 25, 'medium', 'medium', 7),
('Kruszczyk Szerokolistny', 'Epipactis helleborine', 'medium', 12, 24, 'medium', 'medium', 6),
-- Maki
('Mak Polny', 'Papaver rhoeas', 'high', 15, 28, 'low', 'medium', 7),
('Mak Lekarski', 'Papaver somniferum', 'high', 15, 28, 'low', 'medium', 7),
('Mak Wschodni', 'Papaver orientale', 'high', 15, 28, 'medium', 'medium', 6),
-- Marchew
('Marchew Zwyczajna', 'Daucus carota', 'high', 15, 25, 'medium', 'medium', 4),
-- Kolcorośl
('Kolcorośl Szorstkliwy', 'Smilax aspera', 'high', 15, 28, 'low', 'medium', 8),
-- Akacja
('Akacja Srebrzysta', 'Acacia dealbata', 'high', 15, 28, 'low', 'medium', 7),
-- Poziomka
('Poziomka Pospolita', 'Fragaria vesca', 'medium', 15, 25, 'medium', 'medium', 4),
-- Ostróżka
('Ostróżka Czerwona', 'Centranthus ruber', 'high', 15, 28, 'low', 'medium', 8),
-- Aksamitki
('Aksamitka Rozpierzchła', 'Tagetes patula', 'high', 18, 28, 'medium', 'medium', 4),
('Aksamitka Wzniesiona', 'Tagetes erecta', 'high', 18, 28, 'medium', 'medium', 4),
-- Mlecz
('Mlecz Zwyczajny', 'Lapsana communis', 'medium', 12, 25, 'medium', 'medium', 6),
-- Łubin
('Łubin Wielolistny', 'Lupinus polyphyllus', 'high', 12, 25, 'medium', 'medium', 6),
-- Jaśminowiec
('Jaśminowiec Gwiaździsty', 'Trachelospermum jasminoides', 'high', 15, 28, 'medium', 'medium', 5),
-- Dynie
('Dynia Zwyczajna', 'Cucurbita pepo', 'high', 18, 30, 'medium', 'high', 3),
('Dynia Olbrzymia', 'Cucurbita maxima', 'high', 18, 30, 'medium', 'high', 3),
-- Zamiokulkas
('Zamiokulkas', 'Zamioculcas zamiifolia', 'low', 16, 26, 'low', 'low', 12),
-- Podagrycznik
('Podagrycznik Pospolity', 'Aegopodium podagraria', 'medium', 10, 24, 'medium', 'medium', 5),
-- Rzeżucha
('Rzeżucha Pospolita', 'Alliaria petiolata', 'medium', 12, 24, 'medium', 'medium', 6),
-- Kniphofia
('Kniphofia Zwyczajna', 'Kniphofia uvaria', 'high', 15, 28, 'low', 'medium', 7),
-- Tulipanowiec
('Tulipanowiec Amerykański', 'Liriodendron tulipifera', 'high', 15, 28, 'medium', 'medium', 7),
-- Anioł
('Dziewanna Leśna', 'Angelica sylvestris', 'medium', 12, 24, 'medium', 'high', 5),
-- Ognik
('Ognik Szkarłatny', 'Pyracantha coccinea', 'high', 12, 28, 'low', 'medium', 7),
-- Cymbalaria
('Cymbalaria Murowa', 'Cymbalaria muralis', 'medium', 12, 24, 'medium', 'medium', 5),
-- Perowskia
('Perowskia Łobodolistna', 'Perovskia atriplicifolia', 'high', 15, 30, 'low', 'low', 10),
-- Gorczyca
('Gorczyca Zwyczajna', 'Barbarea vulgaris', 'high', 10, 25, 'medium', 'medium', 6),
-- Schefflera
('Schefflera Drzewkowata', 'Schefflera arboricola', 'medium', 18, 27, 'medium', 'medium', 7),
-- Anturium
('Anturium', 'Anthurium andraeanum', 'medium', 18, 28, 'high', 'high', 5),
-- Nandina
('Nandina Domowa', 'Nandina domestica', 'medium', 12, 26, 'medium', 'medium', 6),
-- Fittonia
('Fittonia Srebrzysta', 'Fittonia albivenis', 'low', 18, 26, 'high', 'high', 4);


-- ==========================================
-- 2. TWORZENIE UŻYTKOWNIKÓW (USERS)
-- ==========================================
INSERT INTO users (email, password_hash, username, xp, day_streak, location_city, preferences) VALUES
('jan.kowalski@example.com', '$argon2id$v=19$m=16,t=2,p=1$ZVBlbDMyaFJzSUVRVGhTaQ$2CApfmG+774nfaR0hi4HxQ', 'JanuszOgrodnik', 150, 5, 'Warszawa', '{"dark_mode": true, "care_reminders": true, "weather_tips": false}'::jsonb),
('anna.nowak@example.com', '$argon2id$v=19$m=16,t=2,p=1$ZVBlbDMyaFJzSUVRVGhTaQ$rPGnR6wN8P7aJuzsHyrnBg', 'AniaZZielonegoWzg', 1200, 45, 'Kraków', '{"dark_mode": false, "care_reminders": true, "weather_tips": true}'::jsonb),
('test.user@example.com', '$argon2id$v=19$m=16,t=2,p=1$ZVBlbDMyaFJzSUVRVGhTaQ$OZlVSPAT4AxSy0gF9wNvlA', 'TesterBazy', 0, 0, 'Gdańsk', '{"dark_mode": true, "care_reminders": false, "weather_tips": false}'::jsonb);


-- ==========================================
-- PRZYPISYWANIE ROŚLIN DO UŻYTKOWNIKÓW
-- =========================================
-- Użytkownik 1 (Janusz) ma Pelargonię Pachnącą i Nerecznicę Samczą (paproć)
INSERT INTO user_plants (user_id, plant_catalog_id, custom_name, note, created_at, sprouted_at)
VALUES
(
    1,
    (SELECT id FROM plants_catalog WHERE scientific_name = 'Pelargonium graveolens' LIMIT 1),
    'Pelcia Pachnąca',
    'Pięknie pachnie przy oknie',
    NOW() - INTERVAL '2 months',
    '2023-05-01'
),
(
    1,
    (SELECT id FROM plants_catalog WHERE scientific_name = 'Dryopteris filix-mas' LIMIT 1),
    'Paprotka Zenka',
    'Ta co zawsze usycha',
    NOW() - INTERVAL '1 year',
    '2022-01-01'
);

-- Użytkownik 2 (Ania) ma Lawendę Wąskolistną i Zamiokulkasa (dużo XP, zadbane rośliny)
INSERT INTO user_plants (user_id, plant_catalog_id, custom_name, note, created_at, sprouted_at)
VALUES
(
    2,
    (SELECT id FROM plants_catalog WHERE scientific_name = 'Lavandula angustifolia' LIMIT 1),
    'Lawenda Prowansalska',
    'Uwielbiam jej zapach!',
    NOW() - INTERVAL '6 months',
    '2023-01-15'
),
(
    2,
    (SELECT id FROM plants_catalog WHERE scientific_name = 'Zamioculcas zamiifolia' LIMIT 1),
    'Zamek',
    'Prezent od cioci',
    NOW() - INTERVAL '3 months',
    '2023-08-10'
);

-- Użytkownik 3 (Tester) ma Rozchodnik Ostry (sukulent z listy)
INSERT INTO user_plants (user_id, plant_catalog_id, custom_name, note, created_at, sprouted_at)
VALUES
(
    3,
    (SELECT id FROM plants_catalog WHERE scientific_name = 'Sedum acre' LIMIT 1),
    'Rozchodnik Testowy',
    'Brak uwag',
    NOW(),
    '2024-01-01'
);

-- ==========================================
-- HISTORIA PODLEWANIA (WATERING_DATA)
-- ==========================================
-- WAŻNE: Te ID odnoszą się do user_plants.id, NIE plants_catalog.id!
-- Musimy najpierw znaleźć ID z tabeli user_plants

-- Roślina użytkownika 1 - Pelargonia (user_plants.id)
INSERT INTO watering_data (plant_id, timestamp_of_watering)
VALUES
(
    (SELECT id FROM user_plants WHERE user_id = 1 AND custom_name = 'Pelcia Pachnąca' LIMIT 1),
    NOW() - INTERVAL '1 day'
),
(
    (SELECT id FROM user_plants WHERE user_id = 1 AND custom_name = 'Pelcia Pachnąca' LIMIT 1),
    NOW() - INTERVAL '4 days'
),
(
    (SELECT id FROM user_plants WHERE user_id = 1 AND custom_name = 'Pelcia Pachnąca' LIMIT 1),
    NOW() - INTERVAL '8 days'  -- DO USUNIĘCIA przez trigger (>7 dni)
),
(
    (SELECT id FROM user_plants WHERE user_id = 1 AND custom_name = 'Pelcia Pachnąca' LIMIT 1),
    NOW() - INTERVAL '15 days'  -- DO USUNIĘCIA przez trigger (>7 dni)
);

-- Roślina użytkownika 2 - Lawenda (user_plants.id)
INSERT INTO watering_data (plant_id, timestamp_of_watering)
VALUES
(
    (SELECT id FROM user_plants WHERE user_id = 2 AND custom_name = 'Lawenda Prowansalska' LIMIT 1),
    NOW() - INTERVAL '10 hours'
),
(
    (SELECT id FROM user_plants WHERE user_id = 2 AND custom_name = 'Lawenda Prowansalska' LIMIT 1),
    NOW() - INTERVAL '14 days'  -- DO USUNIĘCIA przez trigger (>7 dni)
);
