-- watering_data.plant_id FK was missing ON DELETE CASCADE, so any plant with
-- recorded care events (water/mist/fertilize/...) couldn't be deleted — DELETE
-- /my-plants/{id} blew up with a ForeignKeyViolation. user_plant_images already
-- has cascade; this aligns watering_data with it.

ALTER TABLE watering_data
    DROP CONSTRAINT IF EXISTS watering_data_plant_id_fkey;

ALTER TABLE watering_data
    ADD CONSTRAINT watering_data_plant_id_fkey
        FOREIGN KEY (plant_id) REFERENCES user_plants(id) ON DELETE CASCADE;
