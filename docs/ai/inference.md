# Inference Module

Moduł inferencji dla klasyfikacji roślin w projekcie Plant Care AI.

## Przegląd

Moduł `plant_care_ai.inference` zapewnia wysokopoziomowy interfejs do wykonywania predykcji na wytrenowanych modelach. Głównym komponentem jest klasa `PlantClassifier`, która obsługuje ładowanie checkpointów, preprocessing obrazów i wykonywanie inferencji.

## Komponenty

### PlantClassifier

Klasa do inferencji modeli klasyfikacji roślin.

```python
from plant_care_ai.inference import PlantClassifier

# Ładowanie z checkpointu
classifier = PlantClassifier.from_checkpoint(
    checkpoint_path="checkpoints/best.pth",
    device="cuda",
    verbose=True,
)

# Predykcja
result = classifier.predict("image.jpg", top_k=5)
print(result["predictions"])
```

## Ładowanie modelu

### from_checkpoint (zalecane)

Ładuje classifier z checkpointu treningowego. Automatycznie wykrywa typ modelu i konfigurację.

```python
classifier = PlantClassifier.from_checkpoint(
    checkpoint_path="checkpoints/best.pth",
    device="cuda",  # lub "cpu"
    verbose=True,
)
```

#### Parametry

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `checkpoint_path` | str \| Path | - | Ścieżka do pliku .pth |
| `device` | str \| None | auto | Urządzenie: `"cuda"`, `"cpu"` lub None (auto) |
| `verbose` | bool | True | Czy drukować informacje o ładowaniu |

#### Obsługiwane modele

- `resnet18` - ResNet18 z custom head
- `efficientnetv2` - EfficientNetV2 (warianty b0, b1, b2, s, m, l)

### Inicjalizacja ręczna

Dla zaawansowanych przypadków użycia:

```python
from plant_care_ai.models.resnet18 import Resnet18

model = Resnet18(num_classes=100)
model.load_state_dict(torch.load("weights.pth"))

classifier = PlantClassifier(
    model=model,
    idx_to_class={0: "1355936", 1: "1355932", ...},
    img_size=224,
    device="cuda",
)
```

## Predykcja

### predict

Wykonuje inferencję na pojedynczym obrazie.

```python
result = classifier.predict(
    image="path/to/image.jpg",  # lub PIL.Image
    top_k=5,
)
```

#### Parametry

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `image` | str \| Path \| PIL.Image | - | Obraz wejściowy |
| `top_k` | int | 5 | Liczba top predykcji do zwrócenia |

#### Format odpowiedzi

```python
{
    "predictions": [
        {
            "class_id": "1355936",
            "class_name": "Rosa canina",  # jeśli ustawiono name mapping
            "confidence": 0.87
        },
        {
            "class_id": "1355932",
            "class_name": "Bellis perennis",
            "confidence": 0.08
        },
        # ...
    ],
    "processing_time_ms": 52.3
}
```

#### Typy wejścia

```python
# Ścieżka jako string
result = classifier.predict("photos/rose.jpg")

# Ścieżka jako Path
from pathlib import Path
result = classifier.predict(Path("photos/rose.jpg"))

# PIL Image
from PIL import Image
image = Image.open("photos/rose.jpg")
result = classifier.predict(image)
```

## Name Mapping

Mapowanie ID klas na nazwy roślin dla czytelniejszych wyników.

### set_name_mapping

```python
# Załaduj mapowanie z pliku JSON
import json
with open("class_id_to_name.json") as f:
    name_mapping = json.load(f)

classifier.set_name_mapping(name_mapping)

# Teraz predykcje zawierają class_name
result = classifier.predict("image.jpg")
for pred in result["predictions"]:
    print(f"{pred['class_name']}: {pred['confidence']:.2%}")
```

#### Format mapowania

```json
{
    "1355936": "Rosa canina (Dzika róża)",
    "1355932": "Bellis perennis (Stokrotka)",
    "1355868": "Taraxacum officinale (Mniszek lekarski)"
}
```

## Atrybuty

| Atrybut | Typ | Opis |
|---------|-----|------|
| `model` | nn.Module | Wytrenowany model PyTorch |
| `device` | str | Urządzenie (cuda/cpu) |
| `num_classes` | int | Liczba klas |
| `img_size` | int | Rozmiar obrazu wejściowego |
| `idx_to_class` | dict | Mapowanie indeks → plant_id |
| `id_to_name` | dict \| None | Mapowanie plant_id → nazwa |
| `transform` | Compose | Pipeline preprocessingu |

## Przykłady użycia

### Podstawowa inferencja

```python
from plant_care_ai.inference import PlantClassifier

classifier = PlantClassifier.from_checkpoint("best.pth")
result = classifier.predict("rosa_canina.jpg", top_k=3)

for pred in result["predictions"]:
    print(f"ID: {pred['class_id']}, Pewność: {pred['confidence']:.2%}")
```

### Z name mappingiem

```python
import json
from plant_care_ai.inference import PlantClassifier

# Załaduj classifier
classifier = PlantClassifier.from_checkpoint("best.pth", device="cpu")

# Załaduj nazwy
with open("class_id_to_name.json") as f:
    names = json.load(f)
classifier.set_name_mapping(names)

# Predykcja
result = classifier.predict("unknown_plant.jpg", top_k=5)

print(f"Czas przetwarzania: {result['processing_time_ms']:.1f}ms\n")
for i, pred in enumerate(result["predictions"], 1):
    name = pred.get("class_name", pred["class_id"])
    print(f"{i}. {name}: {pred['confidence']:.1%}")
```

### Batch processing

```python
from pathlib import Path
from plant_care_ai.inference import PlantClassifier

classifier = PlantClassifier.from_checkpoint("best.pth")

image_dir = Path("photos")
results = []

for image_path in image_dir.glob("*.jpg"):
    result = classifier.predict(image_path, top_k=1)
    top_pred = result["predictions"][0]
    results.append({
        "file": image_path.name,
        "class_id": top_pred["class_id"],
        "confidence": top_pred["confidence"],
    })

# Zapisz wyniki
import json
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Integracja z FastAPI

```python
from fastapi import FastAPI, UploadFile
from PIL import Image
import io

from plant_care_ai.inference import PlantClassifier

app = FastAPI()
classifier = PlantClassifier.from_checkpoint("best.pth")

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = classifier.predict(image, top_k=5)
    return result
```

## Preprocessing

Classifier automatycznie stosuje odpowiedni preprocessing:

1. **Resize** - przeskalowanie do `img_size + margin`
2. **CenterCrop** - wycinanie centralnej części `img_size × img_size`
3. **ToTensor** - konwersja do tensora
4. **Normalize** - normalizacja ImageNet (mean, std)

Nie musisz ręcznie preprocessować obrazów - classifier zrobi to automatycznie.

## Wydajność

### GPU vs CPU

| Urządzenie | Czas na obraz | Throughput |
|------------|---------------|------------|
| CUDA (RTX 3080) | ~15ms | ~65 img/s |
| CPU (i7-10700) | ~150ms | ~6 img/s |

### Optymalizacje

1. **Batch processing**: Dla wielu obrazów rozważ własny DataLoader
2. **Model caching**: Classifier ładuje model raz, predykcje są szybkie
3. **GPU**: Użyj CUDA dla produkcji
4. **Half precision**: Dla jeszcze szybszej inferencji (zaawansowane)

## Obsługa błędów

```python
try:
    classifier = PlantClassifier.from_checkpoint("nonexistent.pth")
except FileNotFoundError:
    print("Checkpoint nie znaleziony")

try:
    result = classifier.predict("invalid_image.txt")
except Exception as e:
    print(f"Błąd predykcji: {e}")
```

## API Reference

### PlantClassifier

```python
class PlantClassifier:
    def __init__(
        self,
        model: nn.Module,
        idx_to_class: dict[int, str],
        img_size: int = 224,
        device: str | None = None,
    ) -> None: ...

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | None = None,
        *,
        verbose: bool = True,
    ) -> "PlantClassifier": ...

    def set_name_mapping(
        self,
        id_to_name: dict[str, str],
    ) -> None: ...

    def predict(
        self,
        image: str | Path | Image.Image,
        top_k: int = 5,
    ) -> dict[str, Any]: ...
```
