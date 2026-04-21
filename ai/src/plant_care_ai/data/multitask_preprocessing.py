"""Preprocessing pipelines for multi-task plant training.

Copyright 2026 Plant Care Assistant
"""

from torchvision import transforms

from .preprocessing import PlantNetPreprocessor, PlantVillagePreprocessor


class MultitaskPreprocessor:
    """Unified preprocessor for multi-task training"""

    def __init__(
        self,
        img_size: int = 224,
        *,
        plantnet_augm: float = 0.6,
        plantvillage_augm: float = 0.5,
        normalize: bool = True,
        pv_vertical_flip: bool = True,
    ) -> None:
        self._preprocessors: dict[str, PlantNetPreprocessor] = {
            "plantnet": PlantNetPreprocessor(
                img_size=img_size,
                normalize=normalize,
                augm_strength=plantnet_augm,
            ),
            "plantvillage": PlantVillagePreprocessor(
                img_size=img_size,
                normalize=normalize,
                augm_strength=plantvillage_augm,
                add_vertical_flip=pv_vertical_flip,
            ),
        }

    def get_transform(self, source: str, *, train: bool = False) -> transforms.Compose:
        if source not in self._preprocessors.keys():
            raise ValueError(f"Unknown source: {source}")
        return self._preprocessors[source].get_transform(train=train)

    def get_plantnet_train_transform(self) -> transforms.Compose:
        return self.get_transform("plantnet", train=True)

    def get_plantnet_val_transform(self) -> transforms.Compose:
        return self.get_transform("plantnet", train=False)

    def get_plantvillage_train_transform(self) -> transforms.Compose:
        return self.get_transform("plantvillage", train=True)

    def get_plantvillage_val_transform(self) -> transforms.Compose:
        return self.get_transform("plantvillage", train=False)

# temp. for tests!
_DEFAULT_PREPROCESSOR = MultitaskPreprocessor()

def get_multitask_transforms(source: str, split: str) -> transforms.Compose:
    train_mode = split == "train"
    return _DEFAULT_PREPROCESSOR.get_transform(source, train=train_mode)