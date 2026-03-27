"""DataLoader wrapper for PlantNet dataset splits.

Copyright 2025 Plant Care Assistant
"""

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import PlantDiseaseDataset, PlantNetDataset


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

class DiseaseDataLoader:
    """Loader specifically for the plant (binary) disease data"""
    
    def __init__(self, data_dir: str, batch_size: int = 32, transform=None):
        full_dataset = PlantDiseaseDataset(data_dir, transform)
        
        train_size = int(0.8 * len(full_dataset)) # 80%, and for 20^% for val and test
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        self.batch_size = batch_size

    def get_loaders(self):
        train = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        val = DataLoader(self.val_ds, batch_size=self.batch_size)
        return train, val