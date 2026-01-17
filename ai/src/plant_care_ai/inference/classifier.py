"""Inference module for plant classification models.

Copyright 2025 Plant Care Assistant
"""

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from plant_care_ai.data.preprocessing import get_inference_pipeline
from plant_care_ai.models.efficientnetv2 import create_efficientnetv2
from plant_care_ai.models.resnet18 import Resnet18


class PlantClassifier:
    """Inference class for plant classification models."""

    def __init__(
        self,
        model: nn.Module,
        idx_to_class: dict[int, str],
        img_size: int = 224,
        device: str | None = None,
    ) -> None:
        """Initialize classifier.

        Args:
            model: Trained PyTorch model.
            idx_to_class: Mapping from model output index to plant_id.
            img_size: Input image size for preprocessing.
            device: Device to run inference on. If None, uses CUDA if available.

        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.idx_to_class = idx_to_class
        self.num_classes = len(idx_to_class)
        self.img_size = img_size

        self.transform = get_inference_pipeline(img_size)
        self.id_to_name = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> "PlantClassifier":
        """Load classifier from training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pth).
            device: Device to load model on. If None, uses CUDA if available.

        Returns:
            PlantClassifier instance ready for inference.

        Raises:
            ValueError: If model type is unknown or num_classes cannot be determined.


        Example:
            >>> classifier = PlantClassifier.from_checkpoint("checkpoints/best.pth")
            >>> result = classifier.predict("image.jpg")

        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        config = checkpoint["config"]
        model_type = config["model"]
        num_classes = checkpoint["num_classes"]
        img_size = config.get("img_size", 224)

        if model_type == "resnet18":
            model = Resnet18(num_classes=num_classes)
        elif model_type == "efficientnetv2":
            model = create_efficientnetv2(
                variant=config["variant"],
                num_classes=num_classes,
            )
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

        model.load_state_dict(checkpoint["model_state_dict"])
        idx_to_class = checkpoint["idx_to_class"]

        classifier = cls(
            model=model,
            idx_to_class=idx_to_class,
            img_size=img_size,
            device=device,
        )

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Model: {model_type}")
        print(f"Classes: {num_classes}")
        if "best_acc" in checkpoint:
            print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")

        return classifier

    def set_name_mapping(self, id_to_name: dict[str, str]) -> None:
        """Set plant ID to name mapping for prettier output.

        Args:
            id_to_name: Dictionary mapping plant_id to plant_name.
                       Example: {"2419045": "Rosa canina (Dzika róża)", ...}

        Example:
            >>> name_mapping = load_plant_names()
            >>> classifier.set_name_mapping(name_mapping)

        """
        self.id_to_name = id_to_name

    def predict(
        self,
        image: str | Path | Image.Image,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Perform inference on a single image.

        Args:
            image: Path to image file or PIL Image object.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with predictions and metadata:
            {
                "predictions": [
                    {
                        "class_id": "2419045",
                        "class_name": "Rosa canina",
                        "confidence": 0.87
                    },
                    ...
                ],
                "processing_time_ms": 52.3
            }

        Example:
            >>> result = classifier.predict("rose.jpg", top_k=3)
            >>> print(result["predictions"][0]["class_name"])
            Rosa canina

        """
        start_time = time.time()

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = probabilities.topk(
                min(top_k, self.num_classes),
                dim=1,
            )

        predictions = []
        for prob, idx in zip(
            top_probs[0].cpu().numpy(),
            top_indices[0].cpu().numpy(),
            strict=False,
        ):
            plant_id = self.idx_to_class[int(idx)]

            prediction = {
                "class_id": plant_id,
                "confidence": float(prob),
            }

            # add plant name,if mapping available
            if self.id_to_name and plant_id in self.id_to_name:
                prediction["class_name"] = self.id_to_name[plant_id]

            predictions.append(prediction)

        processing_time = (time.time() - start_time) * 1000  # ms

        return {
            "predictions": predictions,
            "processing_time_ms": round(processing_time, 1),
        }
