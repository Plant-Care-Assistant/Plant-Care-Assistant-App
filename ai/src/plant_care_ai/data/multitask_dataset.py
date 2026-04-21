"""Unified dataset combining PlantNet and PlantVillage for multi-task training.

Copyright 2026 Plant Care Assistant
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset import PlantNetDataset, PlantVillageDataset


class MultitaskPlantDataset(Dataset):

    IGNORE: int = -1

    def __init__(
        self,
        plantnet_dir: str | None,
        plantvillage_dir: str | None,
        split: str = "train",
        plantnet_transform: transforms.Compose | None = None,
        plantvillage_transform: transforms.Compose | None = None,
        plantnet_species_to_idx: dict[str, int] | None = None,
        disease_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.plantnet_transform    = plantnet_transform
        self.plantvillage_transform = plantvillage_transform
        self._samples: list[dict]  = []

        # plants genera mapping
        self.species_classes: list[str] = []
        self.species_to_idx:  dict[str, int] = {}

        self.disease_classes: list[str] = []
        self.disease_to_idx:  dict[str, int] = {}

        if plantnet_dir is not None:
            self._load_plantnet(plantnet_dir, split, plantnet_species_to_idx)

        if plantvillage_dir is not None:
            self._load_plantvillage(plantvillage_dir, disease_to_idx)

        pn_count = sum(1 for s in self._samples if s["source"] == "plantnet")
        pv_count = len(self._samples) - pn_count
        print(f"[{split}] {len(self._samples)} samples {pn_count} PlantNet, {pv_count} PlantVillage")


    def _load_plantnet(
        self,
        plantnet_dir: str,
        split: str,
        species_to_idx: dict[str, int] | None,
    ) -> None:
        
        pn_ds = PlantNetDataset(
            plantnet_dir,
            split,
            transform=self.plantnet_transform
        )
        s2i = species_to_idx if species_to_idx is not None else pn_ds.class_to_idx

        self.species_classes = pn_ds.classes
        self.species_to_idx  = s2i

        for img_path, species_id in pn_ds.paths:
            self._samples.append({
                "source":"plantnet",
                "path": img_path,
                "species_idx": s2i.get(species_id, self.IGNORE),
                "health_idx": self.IGNORE,
                "disease_idx": self.IGNORE,
                "has_disease_label": False,
            })

    def _load_plantvillage(
        self,
        plantvillage_dir: str,
        disease_to_idx: dict[str, int] | None,
    ) -> None:
        
        pv_ds = PlantVillageDataset(
            plantvillage_dir,
            transform=self.plantvillage_transform,
            disease_to_idx=disease_to_idx,
        )

        self.disease_classes = pv_ds.disease_classes
        self.disease_to_idx = pv_ds.disease_to_idx

        for s in pv_ds._samples:
            self._samples.append({
                "source": "plantvillage",
                "path": s["path"],
                "species_idx": self.IGNORE,
                "health_idx":        s["health_idx"],
                "disease_idx":       s["disease_idx"],
                "has_disease_label": True,
            })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int, int, int, bool]:
        s = self._samples[idx]
        image = Image.open(s["path"]).convert("RGB")
        tf = (self.plantnet_transform
                   if s["source"] == "plantnet"
                   else self.plantvillage_transform)
        if tf is not None:
            image = tf(image)
        return (
            image,
            s["species_idx"],
            s["health_idx"],
            s["disease_idx"],
            s["has_disease_label"],
        )