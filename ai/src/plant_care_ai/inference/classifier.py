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
from plant_care_ai.models.resnet50 import Resnet50


class PlantClassifier:
    """Inference wrapper for plant species classification models."""

    def __init__(
        self,
        model: nn.Module,
        idx_to_class: dict[int, str],
        img_size: int = 224,
        device: str | None = None,
    ) -> None:
        """Initialize the classifier with a model and class mapping.

        Args:
            model: Trained PyTorch model for species classification.
            idx_to_class: Mapping from class index to class ID string.
            img_size: Input image size expected by the model.
            device: Target device ('cuda' or 'cpu'). Auto-detected if None.

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
        *,
        verbose: bool = True,
    ) -> "PlantClassifier":
        """Load classifier from a training checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.pth).
            device: Device to load model on. If None, uses CUDA if available.
            verbose: Whether to print loading information.

        Returns:
            PlantClassifier instance ready for inference.

        Raises:
            ValueError: If model type is unknown or num_classes cannot be determined.

        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config = checkpoint.get("config", {})
        model_type = config.get("model", checkpoint.get("model_type", "efficientnetv2"))
        num_classes = checkpoint.get("num_classes", config.get("num_classes", 1081))
        img_size = config.get("img_size", 224)

        if model_type == "resnet18":
            model: nn.Module = Resnet18(num_classes=num_classes)
        elif model_type == "resnet50":
            model = Resnet50(num_classes=num_classes, pretrained=False)
        elif model_type == "efficientnetv2":
            model = create_efficientnetv2(
                variant=config.get("variant", "b0"),
                num_classes=num_classes,
            )
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}

        classifier = cls(
            model=model,
            idx_to_class=idx_to_class,
            img_size=img_size,
            device=device,
        )

        if checkpoint.get("id_to_name"):
            classifier.set_name_mapping(checkpoint["id_to_name"])
            if verbose:
                print(f"Loaded {len(checkpoint['id_to_name'])} class name mappings from checkpoint")

        if verbose:
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Model: {model_type}")
            print(f"Classes: {num_classes}")
            if "best_acc" in checkpoint:
                print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")

        return classifier

    def set_name_mapping(self, id_to_name: dict[str, str]) -> None:
        """Set a human-readable name mapping for class IDs.

        Args:
            id_to_name: Mapping from class ID string to display name.

        """
        self.id_to_name = id_to_name

    def predict(
        self,
        image: str | Path | Image.Image,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run inference on an image and return top-k species predictions.

        Args:
            image: Path to image file or PIL Image object.
            top_k: Number of top predictions to return.

        Returns:
            dict: Contains 'predictions' (list of dicts with class_id and
                confidence) and 'processing_time_ms'.

        Raises:
            KeyError: If model output index is not found in idx_to_class mapping.

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
            idx_int = int(idx)
            if idx_int not in self.idx_to_class:
                msg = f"Class index {idx_int} not found in idx_to_class mapping"
                raise KeyError(msg)

            plant_id = self.idx_to_class[idx_int]

            prediction: dict[str, Any] = {
                "class_id": plant_id,
                "confidence": float(prob),
            }

            if self.id_to_name and plant_id in self.id_to_name:
                prediction["class_name"] = self.id_to_name[plant_id]

            predictions.append(prediction)

        return {
            "predictions": predictions,
            "processing_time_ms": round((time.time() - start_time) * 1000, 1),
        }
