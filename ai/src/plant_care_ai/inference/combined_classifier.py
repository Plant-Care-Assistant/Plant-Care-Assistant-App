"""Combined plant classifier — merges disease/health predictions with species classification.

Copyright 2026 Plant Care Assistant
"""
from pathlib import Path
from typing import Any

from PIL import Image

from src.plant_care_ai.inference.disease_classifier import DiseasePlantClassifier
from src.plant_care_ai.inference.classifier import PlantClassifier


class CombinedPlantClassifier:
    def __init__(
        self,
        disease_classifier: DiseasePlantClassifier,
        species_classifier: PlantClassifier,
    ) -> None:
        self.disease = disease_classifier
        self.species = species_classifier

    @classmethod
    def from_checkpoints(
        cls,
        disease_checkpoint: str | Path,
        species_checkpoint: str | Path,
        yolo_checkpoint: str | Path,
        device: str | None = None,
        *,
        yolo_conf: float = 0.25,
        verbose: bool = True,
    ) -> "CombinedPlantClassifier":
        disease = DiseasePlantClassifier.from_checkpoints(
            disease_checkpoint=disease_checkpoint,
            yolo_checkpoint=yolo_checkpoint,
            device=device,
            yolo_conf=yolo_conf,
            verbose=verbose,
        )
        species = PlantClassifier.from_checkpoint(
            checkpoint_path=species_checkpoint,
            device=device,
            verbose=verbose,
        )
        return cls(disease, species)

    def predict(
        self,
        image: str | Path | Image.Image,
        top_k_species: int = 5,
        top_k_diseases: int = 3,
    ) -> dict[str, Any]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        disease_result = self.disease.predict(image, top_k_diseases=top_k_diseases)
        species_result = self.species.predict(image, top_k=top_k_species)

        return {
            "species": species_result["predictions"],
            **disease_result,
            "processing_time_ms": round(
                disease_result["processing_time_ms"]
                + species_result["processing_time_ms"],
                1,
            ),
        }