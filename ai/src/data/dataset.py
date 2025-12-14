# Copyright (c) 2025 Plant Care Assistant

"""Dataset class for loading PlantNet plant images."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PlantNetDataset(Dataset):
    """Custom sub-dataset for PlantNet data.

    Loads images from a structured directory:
        data_dir/images/{split}/{species_id}/*.jpg
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: transforms.Compose | None = None
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: root dir with 'images' dir
            split: dataset split ('train', 'val', or 'test') -- 'images' subdirectory
            transform: image transformations to apply

        """
        self.split = split
        self.transform = transform

        # setup paths (main and img- one)
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images" / split

        # connect plants images (paths) with their labels (parents directories)
        self.paths = self._load_paths()

        # class (species) mappings
        self.classes = sorted({label for _, label in self.paths})
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        print(f"Loaded {len(self.paths)} samples, {len(self.classes)} classes")

    def _load_paths(self) -> list[tuple[Path, str]]:
        """Find all .jpg images in the split directory."""
        paths: list[tuple[Path, str]] = []

        for species_dir in self.images_dir.iterdir():
            if species_dir.is_dir():
                species_id = species_dir.name

                paths.extend(
                    (img_path, species_id) for img_path in species_dir.glob("*.jpg")
                )

        return paths

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single dataset item."""
        img_path, species_id = self.paths[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.class_to_idx[species_id]

    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.paths)
