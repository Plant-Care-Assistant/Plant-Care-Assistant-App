"""Inference module for plant classification models.

Copyright 2025 Plant Care Assistant
"""

import time
import json
from pathlib import Path
from typing import Any

import torch
import timm
from PIL import Image
from torch import nn

from src.plant_care_ai.data.preprocessing import get_inference_pipeline
from src.plant_care_ai.models.efficientnetv2 import create_efficientnetv2
from src.plant_care_ai.models.resnet50 import Resnet50


class PlantClassifier:
    def __init__(
        self,
        model: nn.Module,
        idx_to_class: dict[int, str],
        img_size: int = 224,
        device: str | None = None,
    ) -> None:
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
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config = checkpoint.get("config", {})
        model_type = config.get("model", checkpoint.get("model_type", "efficientnetv2"))
        variant = config.get("variant", "tf_efficientnetv2_b0")
        num_classes = checkpoint.get("num_classes", 1081)
        img_size = config.get("img_size", 224)

        if model_type == "resnet50":
            model = Resnet50(num_classes=num_classes, pretrained=False)
        elif model_type == "timm" or "efficientnet" in str(model_type) or model_type == "efficientnetv2":
            model = timm.create_model(
                variant,
                num_classes=num_classes,
                pretrained=False
            )
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        
        if "idx_to_class" in checkpoint:
            idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
        else:
            mapping_path = checkpoint_path.parent / "class_id_to_name.json"
            if verbose:
                print(f"Loading mapping from {mapping_path}")
            
            with open(mapping_path, 'r') as f:
                loaded_mapping = json.load(f)
                idx_to_class = {int(k): v['class_id'] for k, v in loaded_mapping.items()}

        classifier = cls(
            model=model,
            idx_to_class=idx_to_class,
            img_size=img_size,
            device=device,
        )

        if checkpoint.get("id_to_name"):
            classifier.set_name_mapping(checkpoint["id_to_name"])

        if verbose:
            print(f"Loaded checkpoint: {checkpoint_path.name}")
            print(f"Model: {model_type} | Classes: {len(idx_to_class)}")

        return classifier

    def set_name_mapping(self, id_to_name: dict[str, str]) -> None:
        self.id_to_name = id_to_name

    def predict(
        self,
        image: str | Path | Image.Image,
        top_k: int = 5,
    ) -> dict[str, Any]:
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
        ):
            idx_int = int(idx)
            plant_id = self.idx_to_class[idx_int]

            prediction = {
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