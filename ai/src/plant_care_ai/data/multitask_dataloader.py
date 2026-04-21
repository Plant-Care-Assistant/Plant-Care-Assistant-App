"""DataLoader wrapper for unified PlantNet + PlantVillage multi-task splits.

Copyright 2026 Plant Care Assistant. All rights reserved.
"""

from torch.utils.data import DataLoader
from torchvision import transforms

from .multitask_dataset import UnifiedPlantDataset


class MultitaskDataloader:
    """Wrapper creating train/val/test DataLoaders for multi-task learning.

    Builds the species and disease mappings once from the training set and
    reuses them for val/test so indices stay consistent across splits.
    """

    def __init__(
        self,
        plantnet_dir: str | None = None,
        plantvillage_dir: str | None = None,
        batch_size: int = 32,
        train_transform: transforms.Compose | None = None,
        val_transform: transforms.Compose | None = None,
        num_workers: int = 4,
    ) -> None:
        """Initialize all splits sharing the same class mappings.

        Args:
            plantnet_dir: root PlantNet directory
            plantvillage_dir: root PlantVillage directory
            batch_size: batch size for all loaders
            train_transform: augmentation pipeline for training data
            val_transform: deterministic transforms for val/test
            num_workers: DataLoader worker processes
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = UnifiedPlantDataset(
            plantnet_dir=plantnet_dir,
            plantvillage_dir=plantvillage_dir,
            split="train",
            transform=train_transform,
        )

        # share mappings built from train to avoid index leakage
        shared_species = self.train_dataset.species_to_idx
        shared_disease = self.train_dataset.disease_to_idx

        self.val_dataset = UnifiedPlantDataset(
            plantnet_dir=plantnet_dir,
            plantvillage_dir=plantvillage_dir,
            split="val",
            transform=val_transform,
            species_to_idx=shared_species,
            disease_to_idx=shared_disease,
        )

        self.test_dataset = UnifiedPlantDataset(
            plantnet_dir=plantnet_dir,
            plantvillage_dir=plantvillage_dir,
            split="test",
            transform=val_transform,
            species_to_idx=shared_species,
            disease_to_idx=shared_disease,
        )

        self.num_species = self.train_dataset.num_species
        self.num_diseases = self.train_dataset.num_diseases

    def get_train_loader(self) -> DataLoader:
        """Get DataLoader for training (shuffled).

        Returns:
            DataLoader configured for training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_val_loader(self) -> DataLoader:
        """Get DataLoader for validation (not shuffled).

        Returns:
            DataLoader configured for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self) -> DataLoader:
        """Get DataLoader for testing (not shuffled).

        Returns:
            DataLoader configured for testing.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )