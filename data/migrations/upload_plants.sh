#!/bin/bash

# Adres Twojego SeaweedFS
WEED_URL="http://adres_ip:9333/submit"
OUTPUT_SQL="update_fids.sql"

# Wyczyszczenie starego pliku SQL
> $OUTPUT_SQL

echo "Rozpoczynam przesyłanie plików..."

for file in *.jpg; do
    # 1. Pobieramy nazwę bez rozszerzenia (np. "Acacia dealbata")
    # Zamieniamy na małe litery, aby ułatwić dopasowanie w bazie
    scientific_name="${file%.jpg}"

    echo -n "Przesyłam: $scientific_name... "

    # 2. Wysyłamy plik do SeaweedFS i wyciągamy 'fid' z JSON-a
    # SeaweedFS zwraca format: {"fileName":"...","fileSize":...,"fid":"3,01e49b6671"}
    response=$(curl -s -F "file=@$file" "$WEED_URL")
    fid=$(echo $response | jq -r '.fid')

    if [ "$fid" != "null" ] && [ -n "$fid" ]; then
        # 3. Tworzymy kwerendę SQL (UPDATE, bo dane roślin już prawdopodobnie masz)
        # Używamy LOWER() dla pewności dopasowania
        echo "UPDATE plants_catalog SET fid = '$fid' WHERE LOWER(scientific_name) = LOWER('$scientific_name');" >> $OUTPUT_SQL
        echo "OK (fid: $fid)"
    else
        echo "BŁĄD (nie udało się pobrać fid)"
    fi
done

echo "--------------------------------------"
echo "Gotowe! Kwerendy SQL zostały zapisane w: $OUTPUT_SQL"
