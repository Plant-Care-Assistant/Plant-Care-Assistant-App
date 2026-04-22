"""Inference module for plant classification multitask models.

Copyright 2026 Plant Care Assistant
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
from src.plant_care_ai.models.multitask_model import MultiTaskPlantModel


class MultiTaskClassifier:
    def __init__(
        self,
        model: nn.Module,
        idx_to_species: dict[int, str],
        idx_to_disease: dict[int, str],
        img_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.idx_to_species = idx_to_species
        self.idx_to_disease = idx_to_disease
        self.num_species = len(idx_to_species)
        self.num_diseases = len(idx_to_disease)
        self.img_size = img_size

        self.transform = get_inference_pipeline(img_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | None = None,
        *,
        verbose: bool = True,
    ) -> "MultiTaskClassifier":
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config = checkpoint.get("config", {})
        variant    = config.get("variant", "b0")
        img_size   = config.get("img_size", 224)
        dropout    = config.get("dropout", 0.4)
        pretrained = config.get("pretrained", False)

        num_species  = checkpoint.get("num_species",  1081)
        num_diseases = checkpoint.get("num_diseases", 38)

        # rebuild
        model = MultiTaskPlantModel(
            num_species=num_species,
            num_diseases=num_diseases,
            dropout=dropout,
            variant=variant,
            pretrained=False
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        if "idx_to_species" in checkpoint:
            idx_to_species = {int(k): v for k, v in checkpoint["idx_to_species"].items()}
        else:
            s2i = checkpoint.get("species_to_idx", {})
            idx_to_species = {int(v): k for k, v in s2i.items()}

        if "idx_to_disease" in checkpoint:
            idx_to_disease = {int(k): v for k, v in checkpoint["idx_to_disease"].items()}
        else:
            d2i = checkpoint.get("disease_to_idx", {})
            idx_to_disease = {int(v): k for k, v in d2i.items()}

        classifier = cls(
            model=model,
            idx_to_species=idx_to_species,
            idx_to_disease=idx_to_disease,
            img_size=img_size,
            device=device,
        )

        if verbose:
            print(f"Loaded checkpoint: {checkpoint_path.name}")
            print(f"  variant:   tf_efficientnetv2_{variant}")
            print(f"  species:   {num_species}")
            print(f"  diseases:  {num_diseases}")
            val_sp = checkpoint.get("val_acc_species")
            val_di = checkpoint.get("val_acc_disease")
            val_he = checkpoint.get("val_acc_health")
            if val_sp is not None:
                print(
                    f"  best val:  sp={val_sp:.2f}%  "
                    f"di={val_di:.2f}%  he={val_he:.2f}%"
                )

        return classifier

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
            out = self.model(image_tensor)  # dict: species, health, disease

            sp_probs = torch.softmax(out["species"], dim=1)
            top_probs, top_indices = sp_probs.topk(
                min(top_k, self.num_species), dim=1
            )
            species_preds = [
                {
                    "species_id":  self.idx_to_species[int(idx)],
                    "confidence":  float(prob),
                }
                for prob, idx in zip(
                    top_probs[0].cpu().tolist(),
                    top_indices[0].cpu().tolist(),
                )
            ]

            health_prob = torch.sigmoid(out["health"][0, 0]).item()
            health = {
                "label":      "healthy" if health_prob < 0.5 else "diseased",
                "confidence": health_prob if health_prob >= 0.5 else 1.0 - health_prob,
            }

            di_probs = torch.softmax(out["disease"], dim=1)
            di_top_probs, di_top_indices = di_probs.topk(
                min(top_k, self.num_diseases), dim=1
            )
            disease_preds = [
                {
                    "disease":    self.idx_to_disease[int(idx)],
                    "confidence": float(prob),
                }
                for prob, idx in zip(
                    di_top_probs[0].cpu().tolist(),
                    di_top_indices[0].cpu().tolist(),
                )
            ]

        return {
            "species":              species_preds,
            "health":               health,
            "disease":              disease_preds,
            "processing_time_ms":   round((time.time() - start_time) * 1000, 1),
        }