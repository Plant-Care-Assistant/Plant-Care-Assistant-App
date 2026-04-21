"""Dataset classes for loading PlantNet and PlantVillage images.

Copyright 2025 Plant Care Assistant
"""

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
        self, data_dir: str, split: str = "train", transform: transforms.Compose | None = None
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
        """Find all .jpg images in the split directory.

        Returns:
            List of tuples containing (image_path, species_id)

        """
        paths: list[tuple[Path, str]] = []

        for species_dir in self.images_dir.iterdir():
            if species_dir.is_dir():
                species_id = species_dir.name

                paths.extend((img_path, species_id) for img_path in species_dir.glob("*.jpg"))

        return paths

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single dataset item.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple containing (image_tensor, label_index)

        """
        img_path, species_id = self.paths[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.class_to_idx[species_id]

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            Number of samples in the dataset

        """
        return len(self.paths)
    
class PlantVillageDataset(Dataset):
    """Custom sub-dataset for PlantVillage data.

    Loads images from a structured directory:
        data_dir/{color,segmented}/{species__disease}/*.jpg
    """
 
    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose | None = None,
        disease_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.transform = transform

        # we assume that 'color' or 'segmented' dir was already passed as an argument
        # and we will act like that (in a training process)
        path = Path(data_dir)
 
        class_dirs = sorted(d for d in path.iterdir() if d.is_dir())
 
        if disease_to_idx is not None:
            self.disease_to_idx  = disease_to_idx
            self.disease_classes = sorted(disease_to_idx, key=disease_to_idx.get)

        else:
            disease_names = [d.name for d in class_dirs if not d.name.lower().endswith("_healthy")]
            self.disease_classes = disease_names
            self.disease_to_idx  = {n: i for i, n in enumerate(disease_names)}
 
        self._samples: list[dict] = []
        for class_dir in class_dirs:
            is_healthy = class_dir.name.lower().endswith("_healthy")
            health_idx = 0 if is_healthy else 1
            disease_idx = -1 if is_healthy else self.disease_to_idx.get(class_dir.name, -1)
 
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self._samples.append({
                        "path":        img_path,
                        "health_idx":  health_idx,
                        "disease_idx": disease_idx,
                    })
 
        print(f"PlantVillage: {len(self._samples)} samples, {len(self.disease_classes)} disease classes")
 
    def __len__(self) -> int:
        return len(self._samples)
 
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Return (image, health_idx, disease_idx).
 
        disease_idx is -1 for healthy samples.
        """
        s     = self._samples[idx]
        image = Image.open(s["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, s["health_idx"], s["disease_idx"]
 