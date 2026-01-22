-- Dodawanie fid do katalogu - każda roślina z założenia ma jakieś "default" zdjęcie
ALTER TABLE plants_catalog 
ADD COLUMN IF NOT EXISTS fid varchar(64) NOT NULL
CONSTRAINT D_plants_catalog_fid DEFAULT '1,01e49b6671';


-- Dodawanie fid do roślin użytkownika - jeśli jest w katalogu, bierzemy stamtąd, inaczej też zdjęcie "brak zdjęcia"
ALTER TABLE user_plants DROP COLUMN IF EXISTS photo_url; -- już nie używamy tego
ALTER TABLE user_plants
ADD COLUMN IF NOT EXISTS fid varchar(64) NOT NULL
CONSTRAINT D_user_plants_fid DEFAULT '1,01e49b6671'; -- Tutaj jest opcja przypisania przez użytkownika!