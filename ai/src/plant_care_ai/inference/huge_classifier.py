"""Classifier.

Copyright 2026 Plant Care Assistant
"""
import time
from collections import Counter
from pathlib import Path
from typing import Any
 
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from ultralytics import YOLO

from src.plant_care_ai.models.huge_model import HugePlantModel
from src.plant_care_ai.data.preprocessing import get_plantvillage_inference_pipeline
from src.plant_care_ai.models.resnet50 import Resnet50

class HugePlantClassifier:
    def __init__(
        self,
        disease_model: torch.nn.Module,
        yolo_model: YOLO,
        idx_to_species: dict[int, str],
        idx_to_disease: dict[int, str],
        img_size: int = 224,
        yolo_conf: float = 0.25,
        device: str = None
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.disease_model = disease_model.to(self.device).eval()
        self.yolo = yolo_model
        self.idx_to_species = idx_to_species
        self.idx_to_disease = idx_to_disease
        self.num_species = len(idx_to_species)
        self.num_diseases = len(idx_to_disease)
        self.yolo_conf = yolo_conf
        
        self.transform = get_plantvillage_inference_pipeline(img_size)
 
    @classmethod
    def from_checkpoints(
        cls,
        disease_checkpoint: str | Path,
        yolo_checkpoint: str | Path,
        device: str | None = None,
        *,
        yolo_conf: float = 0.25,
        verbose: bool = True,
    ) -> "HugePlantClassifier":
        disease_checkpoint = Path(disease_checkpoint)
        ckpt = torch.load(disease_checkpoint, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        
        variant = config.get("variant", "b0")
        img_size = config.get("img_size", 224)
        num_species = ckpt.get("num_species", 1081)
        num_diseases = ckpt.get("num_diseases", 38)

        model = HugePlantModel(
            model_name=config.get("model_name", "efficientnetv2"),
            num_species=num_species,
            num_diseases=num_diseases,
            variant=variant,
            pretrained=False
        )
        model.load_weights(disease_checkpoint)
 
        idx_sp = ckpt.get("idx_to_species") or {int(v): k for k, v in ckpt.get("species_to_idx", {}).items()}
        idx_di = ckpt.get("idx_to_disease") or {int(v): k for k, v in ckpt.get("disease_to_idx", {}).items()}
 
        yolo = YOLO(str(yolo_checkpoint))

        obj = cls(
            disease_model=model,
            yolo_model=yolo,
            idx_to_species=idx_sp,
            idx_to_disease=idx_di,
            img_size=img_size,
            yolo_conf=yolo_conf,
            device=device
        )

        if verbose:
            print(f"Disease model : {disease_checkpoint.name}  (variant={config.get('model_name', 'efficientnetv2')}_{variant})")
            print(f"YOLO model    : {Path(yolo_checkpoint).name}")
            print(f"Species       : {num_species}   Diseases: {num_diseases}")
 
        return obj


    def _detect_leaves(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        """Run YOLO and return list of (x1, y1, x2, y2) bounding boxes."""
        results = self.yolo.predict(
            image,
            conf=self.yolo_conf,
            verbose=False,
        )
        boxes: list[tuple[int, int, int, int]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().tolist():
                x1, y1, x2, y2 = [int(v) for v in box]

                # skip degenerate boxes
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        return boxes
 
    def _crop_leaves(
        self,
        image: Image.Image,
        boxes: list[tuple[int, int, int, int]],
        padding: int = 8,
    ) -> list[Image.Image]:
        w, h = image.size
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            crops.append(image.crop((x1, y1, x2, y2)))
        return crops
 
    def _classify_batch(
        self, crops: list[Image.Image]
    ) -> dict[str, torch.Tensor]:
        tensors = torch.stack([self.transform(c) for c in crops]).to(self.device)
        with torch.no_grad():
            return self.disease_model(tensors)

    def predict(
        self,
        image: str | Path | Image.Image,
        top_k_species: int = 5,
        top_k_diseases: int = 3,
    ) -> dict[str, Any]:
        t0 = time.time()
 
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
 
        full_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            full_out = self.disease_model(full_tensor)
 
        sp_probs  = torch.softmax(full_out["species"], dim=1)
        top_probs, top_idx = sp_probs.topk(min(top_k_species, self.num_species), dim=1)
        species_preds = [
            {
                "species_id": self.idx_to_species[int(i)],
                "confidence": float(p),
            }
            for p, i in zip(top_probs[0].tolist(), top_idx[0].tolist())
        ]
 
        boxes  = self._detect_leaves(image)
        crops  = self._crop_leaves(image, boxes) if boxes else [image]
        used_full_image_fallback = len(boxes) == 0
 
        out = self._classify_batch(crops)
 
        health_logits  = torch.sigmoid(out["health"]).squeeze(1)
        disease_probs  = torch.softmax(out["disease"], dim=1)
 
        mean_health_logit = float(health_logits.mean().item())
        health_label      = "diseased" if mean_health_logit >= 0.5 else "healthy"
        health_confidence = mean_health_logit if mean_health_logit >= 0.5 else 1.0 - mean_health_logit
 
        aggregated = disease_probs.sum(dim=0)
        aggregated = aggregated / aggregated.sum()
 
        top_di_probs, top_di_idx = aggregated.topk(min(top_k_diseases, self.num_diseases))
        disease_preds = [
            {
                "disease":    self.idx_to_disease[int(i)],
                "confidence": float(p),
            }
            for p, i in zip(top_di_probs.tolist(), top_di_idx.tolist())
        ]
 
        leaf_results = []
        for k in range(len(crops)):
            hlp = float(health_logits[k].item())
            di_p, di_i = disease_probs[k].topk(1)
            leaf_results.append({
                "leaf_index":    k,
                "bbox":          boxes[k] if not used_full_image_fallback else None,
                "health_logit":  hlp,
                "health_label":  "diseased" if hlp >= 0.5 else "healthy",
                "top_disease":   self.idx_to_disease[int(di_i[0])],
                "top_disease_conf": float(di_p[0]),
            })
 
        return {
            "species":              species_preds,
            "health": {
                "label":      health_label,
                "confidence": round(health_confidence, 4),
                "logit":      round(mean_health_logit, 4),
            },
            "diseases":             disease_preds,
            "leaf_count":           len(crops),
            "used_full_image_fallback": used_full_image_fallback,
            "leaf_results":         leaf_results,
            "processing_time_ms":   round((time.time() - t0) * 1000, 1),
        }