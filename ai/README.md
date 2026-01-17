# Moduł AI Plant Care

| | |
| --- | --- |
| Wersja | ![Version](https://img.shields.io/badge/version-0.0.2-blue)
| Pokrycie testami | ![Coverage](./assets/unit-coverage.svg)
| Python | 3.11+
| Framework | PyTorch 2.9.1

Moduł inferencji AI do identyfikacji gatunków roślin z wykorzystaniem modeli głębokiego uczenia.

## Szybki start

### Usługa API (Produkcja)

Uruchom usługę AI z FastAPI:

```bash
cd ai/

# Zbuduj obraz Docker
docker build -f docker/api.Dockerfile -t plant-care-ai:api .

# Uruchom usługę API
docker run --name plant-care-ai-api \
  -p 8001:8001 \
  -v ${PWD}/models:/app/models:ro \
  -e MODEL_NAME=resnet18 \
  -e DEVICE=cpu \
  plant-care-ai:api

# API dostępne pod adresem: http://localhost:8001
# Dokumentacja: http://localhost:8001/docs
```

### Tryb deweloperski

Do lokalnego rozwoju i trenowania:

```bash
cd ai/

# Zbuduj obraz deweloperski CPU
docker build -f docker/cpu.Dockerfile -t plant-care-ai:cpu .

# Uruchom interaktywnie
docker run --name plant-care-ai --rm -it \
  -v ${PWD}:/home/plant_user/work \
  plant-care-ai:cpu bash
```

### Wsparcie GPU

Do trenowania/inferencji z akceleracją GPU:

```bash
# Zbuduj obraz GPU (wymaga NVIDIA Container Toolkit)
docker build -f docker/gpu.Dockerfile -t plant-care-ai:gpu .

# Uruchom z GPU
docker run --name plant-care-ai --rm -it \
  -v ${PWD}:/home/plant_user/work \
  --gpus all \
  plant-care-ai:gpu bash
```

---

##  Endpointy API

### Sprawdzenie stanu
```bash
GET /health

# Odpowiedź:
{
  "status": "healthy",
  "model": "resnet18",
  "device": "cpu",
  "num_classes": 1081,
  "has_weights": true,
  "num_class_mappings": 1081
}
```

### Identyfikacja rośliny
```bash
POST /predict
Content-Type: multipart/form-data

Parametry:
  - file: Plik obrazu (JPEG/PNG)
  - top_k: Liczba predykcji (domyślnie: 5, max: 20)

# Przykład:
curl -X POST http://localhost:8001/predict \
  -F "file=@zdjecie_rosliny.jpg" \
  -F "top_k=3"

# Odpowiedź:
{
  "predictions": [
    {
      "class_id": 123,
      "class_name": "Rosa canina (Dzika róża)",
      "confidence": 0.87
    },
    {
      "class_id": 456,
      "class_name": "Tulipa gesneriana (Tulipan)",
      "confidence": 0.05
    }
  ],
  "processing_time_ms": 52.3
}
```

### Interaktywna dokumentacja
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

---

##  Struktura projektu

```
ai/
├── src/plant_care_ai/
│   ├── models/              # Architektury modeli
│   │   ├── resnet18.py
│   │   ├── efficientnetv2.py
│   │   └── load_models.py
│   ├── data/                # Przetwarzanie danych
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── preprocessing.py
│   └── api/                 # Usługa FastAPI
│       ├── __init__.py
│       └── main.py          # Endpointy API
├── models/                  # Wytrenowane wagi (nie w git)
│   ├── best_model.pth       # Checkpoint modelu
│   ├── class_id_to_name.json
│   └── README.md
├── tests/
│   ├── unit/                # Testy jednostkowe
│   └── integration/         # Testy integracyjne
│       └── test_api.py      # Testy API
├── docker/
│   ├── cpu.Dockerfile       # Obraz deweloperski
│   ├── gpu.Dockerfile       # Obraz GPU do trenowania
│   └── api.Dockerfile       # Obraz API produkcyjny
└── pyproject.toml
```

---

##  Testowanie

### Uruchom testy jednostkowe
```bash
cd ai/
pytest tests/unit/ -v
```

### Uruchom testy integracyjne (w tym testy API)
```bash
cd ai/
pytest tests/integration/ -v
```

### Uruchom wszystkie testy z pokryciem
```bash
cd ai/
pytest --cov=plant_care_ai --cov-report=html
```

### Testuj usługę API
```bash
# Najpierw uruchom usługę API
uvicorn plant_care_ai.api.main:app --reload

# W innym terminalu:
pytest tests/integration/test_api.py -v
```

---

## 🔧 Konfiguracja

### Zmienne środowiskowe

| Zmienna | Opis | Domyślna wartość |
|---------|------|------------------|
| `MODEL_NAME` | Architektura modelu (`resnet18` lub `efficientnetv2`) | `resnet18` |
| `MODEL_WEIGHTS_PATH` | Ścieżka do checkpointu modelu | `/app/models/best_model.pth` |
| `DEVICE` | Urządzenie obliczeniowe (`cpu` lub `cuda`) | `cpu` |
| `NUM_CLASSES` | Liczba klas roślin | `1081` |
| `CLASS_MAPPING_PATH` | Ścieżka do JSON z mapowaniem ID klasy → nazwa | `/app/models/class_id_to_name.json` |

### Wagi modelu

Umieść wytrenowany model w `ai/models/`:

```bash
# Przykład: Skopiuj wytrenowany model
cp /sciezka/do/twojego/checkpoint.pth ai/models/best_model.pth

# Upewnij się, że mapowanie klas istnieje
cat ai/models/class_id_to_name.json
```

Zobacz [models/README.md](models/README.md) po szczegóły.

---

##  Instalacja

### Lokalne środowisko deweloperskie (bez Dockera)

```bash
cd ai/

# Zainstaluj PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Zainstaluj pakiet z zależnościami API
pip install -e ".[api,dev]"

# Uruchom serwer API
uvicorn plant_care_ai.api.main:app --host 0.0.0.0 --port 8001 --reload
```

### Docker Compose (z Backendem)


---

##  Rozwiązywanie problemów

### API zwraca "Model weights not found"

**Rozwiązanie:**
```bash
# Sprawdź czy plik wag istnieje
ls -lh ai/models/best_model.pth

# Jeśli brakuje, skopiuj wytrenowany model:
cp /sciezka/do/checkpoint.pth ai/models/best_model.pth
```

### Ostrzeżenie "Class mapping file not found"

**Rozwiązanie:**
```bash
# Sprawdź czy mapowanie istnieje
cat ai/models/class_id_to_name.json

# Jeśli brakuje, utwórz z danych treningowych
# Zobacz models/README.md po format
```

### Brak pamięci na GPU

**Rozwiązanie:**
```bash
# Użyj CPU zamiast GPU
docker run -e DEVICE=cpu ...

# Lub użyj mniejszego modelu
docker run -e MODEL_NAME=resnet18 ...
```
