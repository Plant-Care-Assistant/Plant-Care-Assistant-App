"""DataLoader wrapper for PlantNet dataset splits.

Copyright 2025 Plant Care Assistant
"""

from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import random

from .dataset import PlantVillageDataset, PlantNetDataset
from .preprocessing import PlantVillagePreprocessor


class PlantNetDataLoader:
    """Wrapper for creating train/val/test DataLoaders."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        train_transform: transforms.Compose | None = None,
        val_transform: transforms.Compose | None = None,
    ) -> None:
        """Initialize all dataset splits.

        Args:
            data_dir: root data dir
            batch_size: batch size for all loaders
            train_transform: transformations for training data
            val_transform: transformations for validation/test data

        """
        self.batch_size = batch_size

        self.train_dataset = PlantNetDataset(data_dir, "train", train_transform)
        self.val_dataset = PlantNetDataset(data_dir, "val", val_transform)
        self.test_dataset = PlantNetDataset(data_dir, "test", val_transform)

        self.num_classes = len(self.train_dataset.classes)

    def get_train_loader(self) -> DataLoader:
        """Get DataLoader for training data.

        Returns:
            DataLoader: DataLoader configured for training
            (with shuffling enabled)

        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self) -> DataLoader:
        """Get DataLoader for validation data.

        Returns:
            DataLoader: DataLoader configured for validation
            (without shuffling)

        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def get_test_loader(self) -> DataLoader:
        """Get DataLoader for test data.

        Returns:
            DataLoader: DataLoader configured for testing
            (without shuffling)

        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class PlantVillageDataLoader:
    """Wrapper for creating train/val/test DataLoaders."""
 
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        preprocessor: PlantVillagePreprocessor | None = None,
        train_val_test_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> None:
        self.batch_size  = batch_size
        self.num_workers = num_workers
 
        pre             = preprocessor or PlantVillagePreprocessor(augm_strength=0.5)
        train_transform = pre.get_full_transform()
        val_transform   = pre.get_inference_transform()
 
        #load once (paths only, no images) to discover disease mapping
        probe = PlantVillageDataset(data_dir, transform=None)
 
        self.disease_classes = probe.disease_classes
        self.disease_to_idx  = probe.disease_to_idx
        self.num_classes     = len(self.disease_classes)
 
        # shuffle indices reproducibly, then partition.
        indices = list(range(len(probe)))
        random.Random(seed).shuffle(indices)
 
        n_train = int(train_val_test_ratio[0] * len(probe))
        n_val   = int(train_val_test_ratio[1] * len(probe))
 
        idx_train = indices[:n_train]
        idx_val   = indices[n_train : n_train + n_val]
        idx_test  = indices[n_train + n_val :]
 
        self.train_dataset = PlantVillageDataset(
            data_dir, transform=train_transform, indices=idx_train
        )
        self.val_dataset = PlantVillageDataset(
            data_dir, transform=val_transform, indices=idx_val
        )
        self.test_dataset = PlantVillageDataset(
            data_dir, transform=val_transform, indices=idx_test
        )
 
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