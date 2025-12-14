-- seed users
INSERT INTO users (
    id,
    name,
    login,
    password,
    email
)
VALUES
(
    -- User 1
    'ab01cd23-5050-44ff-bb66-1234567890fe',  -- Example UUID
    'Alice Smith',
    'alice.s',
    '$argon2id$v=19$m=65536,t=3,p=4$4BEnZHvunIIwaOvx3M4mgA$QZXj+v1O+a2tM/lshkpFFAq/LXgSVGyGN/V3q8O7qko', -- Example 128-char hash (hex)
    'alice@example.com'
),
(
    -- User 2
    'ef45ab67-0404-44ff-bb66-1234567890fe',  -- Example UUID
    'Bob Johnson',
    'bob.j',
    '$argon2id$v=19$m=65536,t=3,p=4$4BEnZHvunIIwaOvx3M4mgA$QZXj+v1O+a2tM/lshkpFFAq/LXgSVGyGN/V3q8O7qko',
    'bob@example.com'
);

-- seed plan info
INSERT INTO plant_info (
    -- plant_id is already SERIAL
    common_name,
    species_name,
    image_reference,
    description
)
VALUES
(
    'Snake Plant',
    'Dracaena trifasciata',
    '/images/plants/snake_plant.jpg',
    'Until 2017, it was known under the synonym Sansevieria trifasciata. This plant is often kept as a houseplant due to its non-demanding maintenance; they can survive with very little water and sun.'
),
(
    'Peppermint',
    'Mentha × piperita',
    '/images/plants/peppermint.png',
    'A hybrid species of mint, a cross between watermint and spearmint. Indigenous to Europe and the Middle East, the plant is now widely spread and cultivated in many regions of the world. It is occasionally found in the wild with its parent species.'
);
