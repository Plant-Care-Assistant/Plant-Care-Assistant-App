"""Combined plant classifier, which merges disease/health predictions with species classification.

Copyright 2026 Plant Care Assistant
"""
from pathlib import Path
from typing import Any

from PIL import Image

from src.plant_care_ai.inference.classifier import PlantClassifier
from src.plant_care_ai.inference.disease_classifier import DiseasePlantClassifier


class CombinedPlantClassifier:
    """Unified classifier combining species identification and disease detection."""

    def __init__(
        self,
        disease_classifier: DiseasePlantClassifier,
        species_classifier: PlantClassifier,
    ) -> None:
        """Initialize with pre-built disease and species classifiers.

        Args:
            disease_classifier: Classifier for plant health and disease detection.
            species_classifier: Classifier for plant species identification.

        """
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
        """Load a CombinedPlantClassifier from checkpoint files.

        Args:
            disease_checkpoint: Path to the disease model checkpoint.
            species_checkpoint: Path to the species model checkpoint.
            yolo_checkpoint: Path to the YOLO leaf-detection model checkpoint.
            device: Target device. Auto-detected if None.
            yolo_conf: Confidence threshold for YOLO detections.
            verbose: Whether to print loading progress.

        Returns:
            CombinedPlantClassifier: Initialised combined classifier.

        """
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
        """Run combined species and disease inference on an image.

        Args:
            image: Input image as a file path or PIL Image.
            top_k_species: Number of top species predictions to return.
            top_k_diseases: Number of top disease predictions to return.

        Returns:
            dict: Contains 'species' predictions, disease/health results,
                and combined 'processing_time_ms'.

        """
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
