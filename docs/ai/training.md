# Training Module

Moduł treningowy dla klasyfikacji roślin w projekcie Plant Care AI.

## Przegląd

Moduł `plant_care_ai.training` zapewnia kompletny pipeline do trenowania modeli klasyfikacji roślin. Zawiera narzędzia do selekcji klas, przygotowania danych oraz pełnego procesu treningowego z walidacją i zapisem checkpointów.

## Komponenty

### PlantTrainer

Główna klasa treningowa zarządzająca całym procesem treningu modelu.

```python
from plant_care_ai.training import PlantTrainer

config = {
    "data_dir": "/path/to/plantnet",
    "checkpoint_dir": "/path/to/checkpoints",
    "subset_classes": ["1355936", "1355932", "1355868"],
    "train_samples_per_class": 500,
    "val_samples_per_class": 100,
    "model": "efficientnetv2",
    "variant": "b0",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 50,
    "lr": 0.001,
    "weight_decay": 0.01,
    "augm_strength": 0.5,
    "experiment_name": "exp_v1",
}

trainer = PlantTrainer(config, verbose=True)
trainer.prepare_data()
trainer.build_model()
trainer.setup_training()
history = trainer.train()
```

#### Wymagane klucze konfiguracji

| Klucz | Typ | Opis |
|-------|-----|------|
| `data_dir` | str | Ścieżka do katalogu z danymi PlantNet |
| `checkpoint_dir` | str | Katalog do zapisu checkpointów |
| `subset_classes` | list[str] | Lista ID klas do trenowania |
| `train_samples_per_class` | int | Liczba próbek treningowych na klasę |
| `val_samples_per_class` | int | Liczba próbek walidacyjnych na klasę |
| `model` | str | Typ modelu: `"resnet18"` lub `"efficientnetv2"` |
| `img_size` | int | Rozmiar obrazu wejściowego |
| `batch_size` | int | Rozmiar batcha |
| `epochs` | int | Liczba epok |
| `lr` | float | Learning rate |
| `weight_decay` | float | Weight decay dla regularyzacji |
| `augm_strength` | float | Siła augmentacji (0.0 - 1.0) |

#### Opcjonalne klucze konfiguracji

| Klucz | Typ | Domyślnie | Opis |
|-------|-----|-----------|------|
| `device` | str | auto | Urządzenie: `"cuda"` lub `"cpu"` |
| `variant` | str | - | Wariant EfficientNetV2 (`"b0"`, `"b1"`, `"b2"`, `"s"`, `"m"`, `"l"`) |
| `num_workers` | int | 2 | Liczba workerów dla DataLoader |
| `label_smoothing` | float | 0.1 | Label smoothing dla CrossEntropyLoss |
| `min_lr` | float | 1e-6 | Minimalne LR dla CosineAnnealing |
| `experiment_name` | str | "exp" | Nazwa eksperymentu |

### get_most_popular_classes

Funkcja do identyfikacji najpopularniejszych klas w datasecie.

```python
from plant_care_ai.training import get_most_popular_classes

top_classes, class_counts = get_most_popular_classes(
    data_dir="/path/to/plantnet",
    top_k=100,
    split="train",
)

print(f"Top 100 klas: {len(top_classes)}")
print(f"Całkowita liczba klas: {len(class_counts)}")
```

#### Parametry

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `data_dir` | str | - | Ścieżka do danych PlantNet |
| `top_k` | int | 100 | Liczba najpopularniejszych klas do zwrócenia |
| `split` | str | "train" | Split do analizy: `"train"`, `"val"`, `"test"` |

#### Wartości zwracane

- `tuple[list[str], dict[str, int]]`:
  - Lista `top_k` ID klas posortowanych malejąco wg popularności
  - Słownik `{class_id: count}` dla wszystkich klas w splicie

## Workflow treningu

### 1. Przygotowanie danych

```python
trainer = PlantTrainer(config)
trainer.prepare_data()
```

Metoda `prepare_data()`:
- Tworzy pipeline'y augmentacji dla danych treningowych i walidacyjnych
- Ładuje datasety PlantNet dla splitów train/val
- Tworzy podzbiory na podstawie `subset_classes` i limitów próbek
- Inicjalizuje DataLoadery z odpowiednimi samplerami

### 2. Budowanie modelu

```python
trainer.build_model()
```

Obsługiwane modele:
- **ResNet18** - lekki model, szybki trening
- **EfficientNetV2** - state-of-the-art, różne warianty (b0-l)

### 3. Konfiguracja treningu

```python
trainer.setup_training()
```

Metoda `setup_training()`:
- Tworzy kryterium CrossEntropyLoss z label smoothing
- Inicjalizuje optimizer AdamW
- Konfiguruje scheduler CosineAnnealingLR
- Tworzy katalog checkpointów i zapisuje config.json

### 4. Trening

```python
history = trainer.train()
```

Trening obejmuje:
- Pętlę po epokach z train_epoch() i validate()
- Gradient clipping (max_norm=1.0)
- Śledzenie najlepszego modelu
- Automatyczny zapis checkpointów (last.pth, best.pth)
- Zapis historii do history.json

## Struktura checkpointów

```
checkpoints/
└── exp_v1/
    ├── config.json      # Konfiguracja eksperymentu
    ├── history.json     # Historia metryk
    ├── last.pth         # Ostatni checkpoint
    └── best.pth         # Najlepszy model (wg val_acc)
```

### Zawartość checkpointu

```python
checkpoint = {
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "val_acc": float,
    "val_top5": float,
    "best_acc": float,
    "config": dict,
    "history": dict,
    "class_to_idx": dict,  # plant_id -> model_idx
    "idx_to_class": dict,  # model_idx -> plant_id
    "num_classes": int,
}
```

## Metryki

Trainer śledzi następujące metryki:

| Metryka | Opis |
|---------|------|
| `train_loss` | Średnia strata treningowa na epokę |
| `train_acc` | Dokładność treningowa (%) |
| `val_loss` | Średnia strata walidacyjna |
| `val_acc` | Top-1 accuracy walidacyjna (%) |
| `val_top5` | Top-5 accuracy walidacyjna (%) |
| `lr` | Learning rate na końcu epoki |

## Przykład kompletny

```python
from plant_care_ai.training import PlantTrainer, get_most_popular_classes

# 1. Wybór klas
top_classes, _ = get_most_popular_classes(
    data_dir="/data/plantnet",
    top_k=100,
)

# 2. Konfiguracja
config = {
    "data_dir": "/data/plantnet",
    "checkpoint_dir": "/checkpoints",
    "subset_classes": top_classes,
    "train_samples_per_class": 500,
    "val_samples_per_class": 100,
    "model": "efficientnetv2",
    "variant": "s",
    "img_size": 300,
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.001,
    "weight_decay": 0.01,
    "augm_strength": 0.7,
    "experiment_name": "efficientnet_s_top100",
}

# 3. Trening
trainer = PlantTrainer(config, verbose=True)
trainer.prepare_data()
trainer.build_model()
trainer.setup_training()
history = trainer.train()

print(f"Najlepsza dokładność: {trainer.best_acc:.2f}%")
print(f"Checkpoint: {trainer.checkpoint_dir}/best.pth")
```

## Wskazówki

1. **Wybór modelu**: Dla szybkich eksperymentów użyj ResNet18. Dla produkcji EfficientNetV2-S lub większy.

2. **Augmentacja**: `augm_strength=0.5` to dobry punkt startowy. Zwiększ do 0.7-0.8 przy overfittingu.

3. **Label smoothing**: Domyślne 0.1 działa dobrze. Zwiększ do 0.2 przy wielu klasach.

4. **Learning rate**: Zacznij od 0.001 dla AdamW. Scheduler automatycznie zmniejszy LR.

5. **Batch size**: Większy batch = szybszy trening, ale więcej pamięci GPU. Typowo 32-128.
