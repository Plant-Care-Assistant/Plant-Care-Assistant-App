"""Unified dataset combining PlantNet and PlantVillage for multi-task learning.

Copyright 2026 Plant Care Assistant. All rights reserved.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class MultitaskDataset(Dataset):
    """Combines PlantNet (species) and PlantVillage (disease) into one dataset.

    Expected PlantNet layout:
        plantnet_dir/images/{split}/{species_id}/*.jpg

    Expected PlantVillage layout:
        plantvillage_dir/
            healthy/*.jpg
            diseased/{disease_name}/*.jpg

    Each sample returns:
        image             : torch.Tensor
        species_idx       : int (-1 if unknown)
        health_idx        : int (0=healthy, 1=diseased, -1 if unknown)
        disease_idx       : int (-1 if unknown or healthy)
        has_disease_label : bool
    """

    def __init__(
        self,
        plantnet_dir: str | None = None,
        plantvillage_dir: str | None = None,
        split: str = "train",
        transform=None,
        species_to_idx: dict[str, int] | None = None,
        disease_to_idx: dict[str, int] | None = None,
    ) -> None:
        """Initialize unified dataset from one or both data sources.

        Args:
            plantnet_dir: root PlantNet directory containing 'images' subdir
            plantvillage_dir: root PlantVillage directory
            split: dataset split ('train', 'val', 'test') for PlantNet only
            transform: image transformations applied to every sample
            species_to_idx: optional pre-built species mapping (use training set mapping for val/test)
            disease_to_idx: optional pre-built disease mapping
        """
        self.IGNORE = -1 #sentinel indx
        self.transform = transform
        self.samples: list[dict] = []

        self.species_to_idx = species_to_idx or {}
        self.disease_to_idx = disease_to_idx or {}

        if plantnet_dir:
            self._load_plantnet(Path(plantnet_dir), split)

        if plantvillage_dir:
            self._load_plantvillage(Path(plantvillage_dir))

        print(
            f"UnifiedPlantDataset [{split}]: {len(self.samples)} samples "
            f"({sum(1 for s in self.samples if not s['has_disease_label'])} plantnet, "
            f"{sum(1 for s in self.samples if s['has_disease_label'])} plantvillage)"
        )

    def _load_plantnet(self, root: Path, split: str) -> None:
        images_dir = root / "images" / split
        if not images_dir.exists():
            raise FileNotFoundError(f"PlantNet split dir not found: {images_dir}")

        for species_dir in sorted(images_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            species_id = species_dir.name

            if species_id not in self.species_to_idx:
                self.species_to_idx[species_id] = len(self.species_to_idx)

            idx = self.species_to_idx[species_id]
            for img_path in species_dir.glob("*.jpg"):
                self.samples.append({
                    "path": img_path,
                    "species_idx": idx,
                    "health_idx": self.IGNORE,
                    "disease_idx": self.IGNORE,
                    "has_disease_label": False,
                })

    def _load_plantvillage(self, root: Path) -> None:
        # PV uses flat folder names: "{Genus}___{disease}" or "{Genus}___healthy"
        for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            folder_name = class_dir.name

            if "___" not in folder_name:
                continue

            _, condition = folder_name.split("___", maxsplit=1)
            is_healthy = condition.lower() == "healthy"
            health_idx = 0 if is_healthy else 1

            if is_healthy:
                disease_idx = self.IGNORE
            else:
                if folder_name not in self.disease_to_idx:
                    self.disease_to_idx[folder_name] = len(self.disease_to_idx)
                disease_idx = self.disease_to_idx[folder_name]

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append({
                        "path": img_path,
                        "species_idx": self.IGNORE,
                        "health_idx": health_idx,
                        "disease_idx": disease_idx,
                        "has_disease_label": True,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int, int, int, bool]:
        s = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (
            image,
            s["species_idx"],
            s["health_idx"],
            s["disease_idx"],
            s["has_disease_label"],
        )

    @property
    def num_species(self) -> int:
        return len(self.species_to_idx)

    @property
    def num_diseases(self) -> int:
        return len(self.disease_to_idx)