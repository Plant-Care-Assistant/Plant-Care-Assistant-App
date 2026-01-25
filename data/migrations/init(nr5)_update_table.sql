-- user_plants wymaga wielu pól
ALTER TABLE user_plants
ALTER COLUMN custom_name TYPE varchar(150); -- parity z plants_catalog.common_name

ALTER TABLE user_plants
ADD COLUMN IF NOT EXISTS scientific_name VARCHAR(150), 
ADD COLUMN IF NOT EXISTS preferred_sunlight light_level,
ADD COLUMN IF NOT EXISTS preferred_temp_min INTEGER, 
ADD COLUMN IF NOT EXISTS preferred_temp_max INTEGER,
ADD COLUMN IF NOT EXISTS air_humidity_req humidity_level,
ADD COLUMN IF NOT EXISTS soil_humidity_req humidity_level,
ADD COLUMN IF NOT EXISTS preferred_watering_interval_days INTEGER CHECK (preferred_watering_interval_days > 0);
-- to wszystko musi być, jeśli mają być wartości nadpisywane własne