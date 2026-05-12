ALTER TABLE plants_catalog
    ADD COLUMN plantsnet_id VARCHAR(20);

CREATE INDEX idx_plants_catalog_plantsnet_id
    ON plants_catalog(plantsnet_id);
