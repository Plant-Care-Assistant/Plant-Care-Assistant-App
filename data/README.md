## Wymagania
- **Docker**
- Albo włączony **Docker Desktop** albo **Docker Engine** jeśli używana jest komenda `compose` z CLI
- ~1 GB RAM, aby było na pewno miejsce na kontener
## Sposób użycia
- Przejdź w CLI do folderu `infra/`.
- Wywołaj komendę `docker compose up -d db` - powinno włączyć kontener z bazą danych.
- - Aby potwierdzić jego istnienie, wywołaj `docker compose ps` i sprawdź istnienie `plant_care_dev_db`.
- Aby sprawdzić bazę danych wewnątrz konteneru, wywołaj `docker compose exec db psql -U pca_devuser -d pca_localdb` - włącza to `psql` i shell DB.
- - wywołanie wewnątrz `psql` `\dt` da listę tablic, których powinny być po pierwszych migracjach dwie.
- - wywołanie wewnątrz `psql` kwerendy, np. `SELECT * FROM users;`, powinny pojawić się dwa wpisy.
- - wywołanie wewnątrz `psql` `\q` opuszcza shell DB.
- Aby zatrzymać kontener, wywołaj komendę `docker compose stop db`.
- - Ważne jest to, że wtedy ponowne włączanie kontenera nie wykona znów migracji!
- - Jeśli jest to potrzebne, lub wymagane jest całkowite wyczyszczenie kontenera, wywołaj `docker compose down --volumes`.
## Struktura
Aktualnie istnieją dwie bazy danych wykonane do wstępnego testowania kontenera:
> users
> - `uid` - UUID użytkownika
> - `login` - Login użytkownika
> - `name` - Imię użytkownika
> - `salt` - Salt do hasła użytkownika
> - `password` - hasło użytkownika po przejściu salting + hashing
> - `created_at` - data stworzenia konta

> plant_info
> - `plant_id` - unikatowy ID dla każdej rośliny
> - `common_name` - Nazwa pospolita rośliny
> - `species_name` - Poprawna nazwa gatunku rośliny
> - `image_reference` - odnośnik do miejsca ze zdjęciem rośliny
> - `description` - opis rośliny
