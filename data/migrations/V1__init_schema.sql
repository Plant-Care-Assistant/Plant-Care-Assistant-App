CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    login VARCHAR(255) UNIQUE NOT NULL,
    password CHAR(97) NOT NULL,
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE plant_info (
    id SERIAL PRIMARY KEY,
    common_name VARCHAR(255) NOT NULL,
    species_name VARCHAR(255) UNIQUE NOT NULL,
    image_reference VARCHAR(512), -- this is a file path or URL.
    description TEXT -- this could be really long, so it's a TEXT field
);

CREATE TABLE users_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    special_role TEXT
);

CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    replaced_by_token_id UUID REFERENCES refresh_tokens(id),
    device_name TEXT,
    issued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE users IS 'Stores user authentication and profile data.';
COMMENT ON TABLE plant_info IS 'Stores detailed information about plants.';
