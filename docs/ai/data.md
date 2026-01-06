# Moduł Data - Plant Care AI

Moduł `data` odpowiada za ładowanie, przetwarzanie i augmentację danych obrazowych roślin z datasetu PlantNet-300K.

## Spis treści

- [Przegląd](#przegląd)
- [Struktura katalogów](#struktura-katalogów)
- [Klasy](#klasy)
  - [PlantNetDataset](#plantnetdataset)
  - [PlantNetDataLoader](#plantnetdataloader)
  - [PlantNetPreprocessor](#plantnetpreprocessor)
- [Funkcje pomocnicze](#funkcje-pomocnicze)
- [Przykłady użycia](#przykłady-użycia)

---

## Przegląd

Moduł składa się z trzech głównych komponentów:

| Plik | Opis |
|------|------|
| `dataset.py` | Klasa `PlantNetDataset` - ładowanie obrazów i mapowanie etykiet |
| `dataloader.py` | Klasa `PlantNetDataLoader` - wrapper dla PyTorch DataLoader |
| `preprocessing.py` | Klasa `PlantNetPreprocessor` - pipeline transformacji i augmentacji |

---

## Struktura katalogów

Moduł oczekuje następującej struktury danych:

```
data_dir/
└── images/
    ├── train/
    │   ├── {species_id_1}/
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   └── {species_id_2}/
    │       └── ...
    ├── val/
    │   └── {species_id}/
    │       └── ...
    └── test/
        └── {species_id}/
            └── ...
```

---

## Klasy

### PlantNetDataset

Klasa dziedzicząca po `torch.utils.data.Dataset` do ładowania obrazów roślin.

#### Importowanie

```python
from plant_care_ai.data.dataset import PlantNetDataset
```

#### Konstruktor

```python
PlantNetDataset(
    data_dir: str,
    split: str = "train",
    transform: transforms.Compose | None = None
)
```

**Parametry:**

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `data_dir` | `str` | - | Ścieżka do głównego katalogu z danymi |
| `split` | `str` | `"train"` | Podział danych: `"train"`, `"val"` lub `"test"` |
| `transform` | `transforms.Compose \| None` | `None` | Pipeline transformacji obrazów |

#### Atrybuty

| Atrybut | Typ | Opis |
|---------|-----|------|
| `paths` | `list[tuple[Path, str]]` | Lista par (ścieżka_obrazu, species_id) |
| `classes` | `list[str]` | Posortowana lista unikalnych klas (gatunków) |
| `class_to_idx` | `dict[str, int]` | Mapowanie nazwy klasy na indeks numeryczny |

#### Metody

| Metoda | Zwraca | Opis |
|--------|--------|------|
| `__len__()` | `int` | Liczba próbek w datasecie |
| `__getitem__(idx)` | `tuple[Tensor, int]` | Zwraca parę (tensor_obrazu, indeks_klasy) |

---

### PlantNetDataLoader

Wrapper ułatwiający tworzenie DataLoaderów dla wszystkich splitów.

#### Importowanie

```python
from plant_care_ai.data.dataloader import PlantNetDataLoader
```

#### Konstruktor

```python
PlantNetDataLoader(
    data_dir: str,
    batch_size: int = 32,
    train_transform: transforms.Compose | None = None,
    val_transform: transforms.Compose | None = None
)
```

**Parametry:**

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `data_dir` | `str` | - | Ścieżka do głównego katalogu z danymi |
| `batch_size` | `int` | `32` | Rozmiar batcha dla wszystkich loaderów |
| `train_transform` | `transforms.Compose \| None` | `None` | Transformacje dla danych treningowych |
| `val_transform` | `transforms.Compose \| None` | `None` | Transformacje dla danych walidacyjnych/testowych |

#### Atrybuty

| Atrybut | Typ | Opis |
|---------|-----|------|
| `num_classes` | `int` | Liczba unikalnych klas w zbiorze treningowym |
| `train_dataset` | `PlantNetDataset` | Dataset treningowy |
| `val_dataset` | `PlantNetDataset` | Dataset walidacyjny |
| `test_dataset` | `PlantNetDataset` | Dataset testowy |

#### Metody

| Metoda | Zwraca | Opis |
|--------|--------|------|
| `get_train_loader()` | `DataLoader` | Loader treningowy (z shufflingiem) |
| `get_val_loader()` | `DataLoader` | Loader walidacyjny (bez shufflingu) |
| `get_test_loader()` | `DataLoader` | Loader testowy (bez shufflingu) |

---

### PlantNetPreprocessor

Klasa do tworzenia pipeline'ów przetwarzania obrazów z opcjonalną augmentacją.

#### Importowanie

```python
from plant_care_ai.data.preprocessing import PlantNetPreprocessor
```

#### Konstruktor

```python
PlantNetPreprocessor(
    img_size: int = 224,
    *,
    normalize: bool = True,
    augm_strength: float = 0.0
)
```

**Parametry:**

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `img_size` | `int` | `224` | Docelowy rozmiar obrazu (kwadratowy) |
| `normalize` | `bool` | `True` | Czy stosować normalizację ImageNet |
| `augm_strength` | `float` | `0.0` | Intensywność augmentacji (0.0 - 1.0) |

#### Stałe klasowe

```python
ROTATION_THRESHOLD = 0.3      # Próg dla losowych rotacji
COLOR_JITTER_THRESHOLD = 0.5  # Próg dla jittera kolorów
AFFINE_THRESHOLD = 0.7        # Próg dla transformacji afinicznych
```

#### Normalizacja ImageNet

Preprocessor używa standardowych wartości normalizacji ImageNet:

- **Mean:** `[0.485, 0.456, 0.406]`
- **Std:** `[0.229, 0.224, 0.225]`

#### Metody

| Metoda | Zwraca | Opis |
|--------|--------|------|
| `get_full_transform()` | `transforms.Compose` | Pipeline treningowy z augmentacją |
| `get_inference_transform()` | `transforms.Compose` | Pipeline inferencyjny (bez augmentacji) |
| `get_transform(train=False)` | `transforms.Compose` | Wybór pipeline'u na podstawie trybu |

#### Augmentacje w zależności od `augm_strength`

| Zakres | Augmentacje |
|--------|-------------|
| `> 0.0` | Losowe odbicie poziome (p=0.5) |
| `≥ 0.3` | Losowa rotacja (do 30° × strength) |
| `≥ 0.5` | Color jitter (brightness, contrast, saturation, hue) |
| `≥ 0.7` | Losowa translacja i skalowanie |

---

## Funkcje pomocnicze

### `get_training_pipeline`

```python
def get_training_pipeline(
    img_size: int = 224,
    augm_strength: float = 0.5
) -> transforms.Compose
```

Szybki dostęp do pipeline'u treningowego z augmentacją.

### `get_inference_pipeline`

```python
def get_inference_pipeline(img_size: int = 224) -> transforms.Compose
```

Szybki dostęp do pipeline'u inferencyjnego.

### `preprocess_single_image`

```python
def preprocess_single_image(
    image: str | Path,
    img_size: int = 224
) -> torch.Tensor
```

Przetwarza pojedynczy obraz do formatu gotowego dla modelu.

**Zwraca:** Tensor o kształcie `(1, C, H, W)`.

---

## Przykłady użycia

### Podstawowe ładowanie danych

```python
from plant_care_ai.data import PlantNetDataLoader, PlantNetPreprocessor

# Utworzenie preprocessora
preprocessor = PlantNetPreprocessor(img_size=224, augm_strength=0.5)

# Utworzenie data loadera
data_loader = PlantNetDataLoader(
    data_dir="/path/to/plantnet",
    batch_size=32,
    train_transform=preprocessor.get_full_transform(),
    val_transform=preprocessor.get_inference_transform()
)

# Pobranie loaderów
train_loader = data_loader.get_train_loader()
val_loader = data_loader.get_val_loader()

print(f"Liczba klas: {data_loader.num_classes}")
```

### Iterowanie po danych

```python
for images, labels in train_loader:
    # images: Tensor[B, 3, 224, 224]
    # labels: Tensor[B]

    outputs = model(images)
    loss = criterion(outputs, labels)
    # ...
```

### Przetwarzanie pojedynczego obrazu do inferencji

```python
from plant_care_ai.data.preprocessing import preprocess_single_image

# Przetworzenie obrazu
tensor = preprocess_single_image("/path/to/plant_image.jpg")

# Predykcja
with torch.no_grad():
    output = model(tensor)
    predicted_class = output.argmax(dim=1).item()
```

### Używanie różnych poziomów augmentacji

```python
# Lekka augmentacja (tylko flip)
light_prep = PlantNetPreprocessor(augm_strength=0.2)

# Średnia augmentacja (flip + rotacja + color jitter)
medium_prep = PlantNetPreprocessor(augm_strength=0.6)

# Silna augmentacja (wszystkie transformacje)
strong_prep = PlantNetPreprocessor(augm_strength=1.0)
```

---

## Zależności

- `torch`
- `torchvision`
- `PIL` (Pillow)
- `pathlib` (stdlib)
