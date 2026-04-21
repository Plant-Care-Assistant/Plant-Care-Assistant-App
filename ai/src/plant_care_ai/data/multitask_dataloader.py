"""DataLoader wrapper for unified PlantNet + PlantVillage multi-task training.

Copyright 2026 Plant Care Assistant
"""
import random

from torch.utils.data import ConcatDataset, DataLoader

from .multitask_dataset import MultitaskPlantDataset
from .multitask_preprocessing import MultitaskPreprocessor


class MultitaskDataLoader:
    """Creates train / val / test DataLoaders for multi-task plant training"""

    def __init__(
        self,
        plantnet_dir: str | None = None,
        plantvillage_dir: str | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        preprocessor: MultitaskPreprocessor | None = None,
        seed: int = 42,
    ) -> None:
        self.batch_size  = batch_size
        self.num_workers = num_workers

        pre = preprocessor or MultitaskPreprocessor()

        pn_train = pn_val = pn_test = None

        if plantnet_dir is not None:
            pn_train = MultitaskPlantDataset(
                plantnet_dir=plantnet_dir,
                plantvillage_dir=None,
                split="train",
                plantnet_transform=pre.get_plantnet_train_transform(),
            )
            pn_val = MultitaskPlantDataset(
                plantnet_dir=plantnet_dir,
                plantvillage_dir=None,
                split="val",
                plantnet_transform=pre.get_plantnet_val_transform(),
                plantnet_species_to_idx=pn_train.species_to_idx,
            )
            pn_test = MultitaskPlantDataset(
                plantnet_dir=plantnet_dir,
                plantvillage_dir=None,
                split="test",
                plantnet_transform=pre.get_plantnet_val_transform(),
                plantnet_species_to_idx=pn_train.species_to_idx,
            )

            self.species_classes = pn_train.species_classes
            self.species_to_idx  = pn_train.species_to_idx
            self.num_species     = len(self.species_classes)

        pv_train = pv_val = pv_test = None

        if plantvillage_dir is not None:
            pv_probe = MultitaskPlantDataset(
                plantnet_dir=None,
                plantvillage_dir=plantvillage_dir,
                split="train", # unused for PlantVillage
            )
            self.disease_classes = pv_probe.disease_classes
            self.disease_to_idx  = pv_probe.disease_to_idx

            n= len(pv_probe)
            indices = list(range(n))
            random.Random(seed).shuffle(indices)

            n_train = int(train_val_test_ratio[0] * n)
            n_val = int(train_val_test_ratio[1] * n)

            idx_train = indices[:n_train]
            idx_val = indices[n_train : n_train + n_val]
            idx_test = indices[n_train + n_val :]

            shared = dict(
                plantnet_dir=None,
                plantvillage_dir=plantvillage_dir,
                split="train",
                disease_to_idx=pv_probe.disease_to_idx
            )

            pv_train = MultitaskPlantDataset(
                **shared,
                plantvillage_transform=pre.get_plantvillage_train_transform(),
                indices=idx_train,
            )
            pv_val = MultitaskPlantDataset(
                **shared,
                plantvillage_transform=pre.get_plantvillage_val_transform(),
                indices=idx_val,
            )
            pv_test = MultitaskPlantDataset(
                **shared,
                plantvillage_transform=pre.get_plantvillage_val_transform(),
                indices=idx_test,
            )

        if not hasattr(self, "disease_classes"):
            self.disease_classes = []
            self.disease_to_idx  = {}
        self.num_diseases = len(self.disease_classes)

        self.train_dataset = _concat(pn_train, pv_train)
        self.val_dataset   = _concat(pn_val,   pv_val)
        self.test_dataset  = _concat(pn_test,  pv_test)

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def _concat(*datasets):
    valid = [d for d in datasets if d is not None]
    if not valid:
        raise ValueError(
            "At least one of plantnet_dir or plantvillage_dir must be provided."
        )
    return valid[0] if len(valid) == 1 else ConcatDataset(valid)