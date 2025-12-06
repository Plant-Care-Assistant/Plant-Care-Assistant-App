-- seed users
INSERT INTO users (
    uid, -- UID is provided explicitly for testing. realistically it's automatically generated
    name,
    login,
    salt,
    password
)
VALUES
(
    -- User 1
    'ab01cd23-5050-44ff-bb66-1234567890fe',  -- Example UUID
    'Alice Smith',
    'alice.s',
    '2e8a94b015adaa4ec9c1a951848070f9',  -- example, not actual salt
    '9e5295b0956d5e5f937c477cc59339c2a07af9777a3d68265c74844ab745393f5faec1bcff1fbdefa6310779b2048cba29a6b893869e55ea84662552a4306bb9' -- Example 128-char hash (hex)
),
(
    -- User 2
    'ef45ab67-0404-44ff-bb66-1234567890fe',  -- Example UUID
    'Bob Johnson',
    'bob.j',
    'fd4844ec987d93aadf55300a32a0b4cf',
    '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd'
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
