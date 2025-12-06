CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
    uid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    login VARCHAR(255) UNIQUE NOT NULL,
    salt CHAR(32) NOT NULL, -- this is the password salt..
    password CHAR(128) NOT NULL, -- and this is the salted + hashed password!!! not plaintext, ever!!!
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE plant_info (
    plant_id SERIAL PRIMARY KEY,
    common_name VARCHAR(255) NOT NULL,
    species_name VARCHAR(255) UNIQUE NOT NULL,
    image_reference VARCHAR(512), -- this is a file path or URL.
    description TEXT -- this could be really long, so it's a TEXT field
);

COMMENT ON TABLE users IS 'Stores user authentication and profile data.';
COMMENT ON TABLE plant_info IS 'Stores detailed information about plants.';
