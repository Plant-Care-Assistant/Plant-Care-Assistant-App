# Moduł Models - Plant Care AI

Moduł `models` zawiera implementacje architektur sieci neuronowych do klasyfikacji roślin na podstawie datasetu PlantNet-300K.

## Spis treści

- [Przegląd](#przegląd)
- [Wspierane architektury](#wspierane-architektury)
- [Klasy modeli](#klasy-modeli)
  - [Resnet18](#resnet18)
  - [EfficientNetV2](#efficientnetv2)
- [Funkcje pomocnicze](#funkcje-pomocnicze)
- [Przykłady użycia](#przykłady-użycia)
- [Transfer Learning](#transfer-learning)

---

## Przegląd

Moduł oferuje dwie architektury sieci konwolucyjnych zoptymalizowane pod klasyfikację roślin:

| Model | Plik | Parametry | Zalety |
|-------|------|-----------|--------|
| ResNet-18 | `resnet18.py` | ~11M | Szybki, lekki, dobry baseline |
| EfficientNetV2 | `efficientnetv2.py` | ~21M (B3) | Wysokia dokładność, efektywny |

---

## Wspierane architektury

### ResNet-18

Implementacja bazująca na publikacji [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

**Architektura:**
- 4 warstwy rezydualne (layer1-layer4)
- BasicBlock z połączeniami skip
- Global Average Pooling
- Warstwa fully-connected

### EfficientNetV2

Implementacja bazująca na publikacji [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

**Warianty:**

| Wariant | Kanały | Bloki | Przypadek użycia |
|---------|--------|-------|------------------|
| `b0` | 512 | 6 stages | Urządzenia mobilne |
| `b1` | 512 | 6 stages | Balans szybkość/dokładność |
| `b2` | 640 | 6 stages | Wyższa dokładność |
| `b3` | 640 | 6 stages | **Domyślny** - najlepsza dokładność |

---

## Klasy modeli

### Resnet18

Pełna implementacja architektury ResNet-18.

#### Importowanie

```python
from plant_care_ai.models.resnet18 import Resnet18
```

#### Konstruktor

```python
Resnet18(num_classes: int = 1081)
```

**Parametry:**

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `num_classes` | `int` | `1081` | Liczba klas wyjściowych (gatunków roślin) |

#### Metody

| Metoda | Opis |
|--------|------|
| `forward(x: Tensor) -> Tensor` | Przepuszczenie tensora przez sieć |
| `freeze_backbone()` | Zamrożenie wszystkich warstw oprócz FC |
| `unfreeze_backbone()` | Odmrożenie wszystkich warstw |

#### Architektura wewnętrzna

```
Input (3, 224, 224)
    ↓
Conv1 + BN + ReLU (64)
    ↓
MaxPool (3×3, stride=2)
    ↓
Layer1: 2× BasicBlock (64)
    ↓
Layer2: 2× BasicBlock (128, stride=2)
    ↓
Layer3: 2× BasicBlock (256, stride=2)
    ↓
Layer4: 2× BasicBlock (512, stride=2)
    ↓
AdaptiveAvgPool (1×1)
    ↓
FC (512 → num_classes)
    ↓
Output (num_classes)
```

#### BasicBlock

Blok rezydualny z dwoma konwolucjami 3×3:

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    )
```

---

### EfficientNetV2

Zaawansowana architektura z FusedMBConv i MBConv.

#### Importowanie

```python
from plant_care_ai.models.efficientnetv2 import EfficientNetV2
```

#### Konstruktor

```python
EfficientNetV2(
    variant: str = "b3",
    num_classes: int = 1081,
    *,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.2,
    width_mult: float = 1.0,
    depth_mult: float = 1.0
)
```

**Parametry:**

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `variant` | `str` | `"b3"` | Wariant modelu: `b0`, `b1`, `b2`, `b3` |
| `num_classes` | `int` | `1081` | Liczba klas wyjściowych |
| `dropout_rate` | `float` | `0.3` | Dropout przed klasyfikatorem |
| `drop_path_rate` | `float` | `0.2` | Stochastic depth rate |
| `width_mult` | `float` | `1.0` | Mnożnik szerokości kanałów |
| `depth_mult` | `float` | `1.0` | Mnożnik głębokości warstw |

#### Metody

| Metoda | Opis |
|--------|------|
| `forward(x: Tensor) -> Tensor` | Przepuszczenie tensora przez sieć |
| `freeze_stages(num_stages: int)` | Zamrożenie N pierwszych stage'ów |
| `unfreeze_all()` | Odmrożenie wszystkich warstw |

#### Komponenty wewnętrzne

##### StochasticDepth

Regularyzacja poprzez losowe pomijanie ścieżek:

```python
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0)
```

##### SqueezeExcitation

Blok Squeeze-and-Excitation do rekalibracji kanałów:

```python
class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        se_ratio: float = 0.25
    )
```

##### FusedMBConv

Zoptymalizowany blok dla początkowych warstw:

```python
class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        *,
        kernel_size: int = 3,
        drop_path_rate: float = 0.0
    )
```

##### MBConv

Mobile Inverted Bottleneck z depthwise separable convolutions:

```python
class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        *,
        kernel_size: int = 3,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0
    )
```

---

## Funkcje pomocnicze

### `get_model`

Uniwersalna funkcja do tworzenia i ładowania modeli.

```python
from plant_care_ai.models.load_models import get_model

model = get_model(
    model_name: str,
    num_classes: int = 1081,
    weights_path: str | None = None,
    device: str = "cpu",
    **kwargs
) -> torch.nn.Module
```

**Parametry:**

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `model_name` | `str` | - | `"resnet18"` lub `"efficientnetv2"` |
| `num_classes` | `int` | `1081` | Liczba klas wyjściowych |
| `weights_path` | `str \| None` | `None` | Ścieżka do pliku `.pth` z wagami |
| `device` | `str` | `"cpu"` | Urządzenie: `"cpu"` lub `"cuda"` |
| `**kwargs` | - | - | Dodatkowe argumenty (np. `variant="b2"`) |

**Przykład:**

```python
# ResNet-18
model = get_model("resnet18", num_classes=100)

# EfficientNetV2-B2
model = get_model("efficientnetv2", variant="b2", device="cuda")

# Załadowanie wag
model = get_model(
    "resnet18",
    weights_path="/path/to/checkpoint.pth",
    device="cuda"
)
```

### `create_efficientnetv2`

Fabryka dla modeli EfficientNetV2:

```python
from plant_care_ai.models.efficientnetv2 import create_efficientnetv2

model = create_efficientnetv2(
    variant: str = "b3",
    num_classes: int = 1081,
    **kwargs
) -> EfficientNetV2
```

---

## Przykłady użycia

### Tworzenie modelu

```python
from plant_care_ai.models import get_model

# Prosty model ResNet-18
model = get_model("resnet18", num_classes=1081)

# EfficientNetV2-B3 na GPU
model = get_model(
    "efficientnetv2",
    variant="b3",
    num_classes=1081,
    device="cuda"
)
```

### Trening

```python
import torch
import torch.nn as nn
from plant_care_ai.models import get_model
from plant_care_ai.data import PlantNetDataLoader, PlantNetPreprocessor

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model("efficientnetv2", num_classes=1081, device=device)

# Data
preprocessor = PlantNetPreprocessor(augm_strength=0.5)
data = PlantNetDataLoader(
    "/path/to/data",
    batch_size=32,
    train_transform=preprocessor.get_full_transform(),
    val_transform=preprocessor.get_inference_transform()
)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in data.get_train_loader():
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Inferencja

```python
import torch
from plant_care_ai.models import get_model
from plant_care_ai.data.preprocessing import preprocess_single_image

# Załadowanie modelu z wagami
model = get_model(
    "efficientnetv2",
    weights_path="best_model.pth",
    device="cuda"
)
model.eval()

# Przetworzenie obrazu
image_tensor = preprocess_single_image("/path/to/plant.jpg")
image_tensor = image_tensor.to("cuda")

# Predykcja
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = logits.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

print(f"Klasa: {predicted_class}, Pewność: {confidence:.2%}")
```

---

## Transfer Learning

### Fine-tuning ResNet-18

```python
from plant_care_ai.models import get_model

# Załadowanie modelu
model = get_model("resnet18", num_classes=1081)

# Zamrożenie backbone'u
model.freeze_backbone()

# Trenowanie tylko warstwy FC
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# Po kilku epokach - odmrożenie
model.unfreeze_backbone()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

### Stopniowe odmrażanie EfficientNetV2

```python
from plant_care_ai.models import get_model

model = get_model("efficientnetv2", variant="b3")

# Etap 1: Tylko klasyfikator
model.freeze_stages(6)  # Zamrożenie wszystkich stage'ów
# Trenuj...

# Etap 2: Ostatnie 2 stage'y
model.freeze_stages(4)
# Trenuj z niższym LR...

# Etap 3: Pełne fine-tuning
model.unfreeze_all()
# Trenuj z bardzo niskim LR...
```

---

## Inicjalizacja wag

Oba modele używają przemyślanej inicjalizacji:

| Warstwa | Metoda |
|---------|--------|
| `Conv2d` | Kaiming Normal (fan_out, ReLU) |
| `BatchNorm2d` | weight=1, bias=0 |
| `Linear` | Normal(0, 0.01), bias=0 |

---

## Formaty checkpoint'ów

Funkcja `get_model` obsługuje dwa formaty:

1. **Surowy state_dict:**
   ```python
   torch.save(model.state_dict(), "model.pth")
   ```

2. **Checkpoint ze słownikiem:**
   ```python
   torch.save({
       "state_dict": model.state_dict(),
       "epoch": epoch,
       "optimizer": optimizer.state_dict()
   }, "checkpoint.pth")
   ```

---

## Zależności

- `torch`
- `math` (stdlib)
- `pathlib` (stdlib)
