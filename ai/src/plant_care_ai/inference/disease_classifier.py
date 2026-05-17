"""Disease inference module: detects plant health and disease via YOLO leaf cropping.

Copyright 2026 Plant Care Assistant
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from plant_care_ai.data.preprocessing import get_plantvillage_inference_pipeline
from plant_care_ai.models.disease_model import DiseasePlantModel

if TYPE_CHECKING:
    from ultralytics import YOLO

DEFAULT_HEALTH_THRESHOLD = 0.5


class DiseasePlantClassifier:
    """Inference wrapper for plant disease detection using YOLO leaf cropping."""

    def __init__(
        self,
        disease_model: torch.nn.Module,
        yolo_model: YOLO,
        idx_to_disease: dict[int, str],
        img_size: int = 224,
        yolo_conf: float = 0.25,
        health_threshold: float = DEFAULT_HEALTH_THRESHOLD,
        device: str | None = None,
    ) -> None:
        """Initialize the disease classifier.

        Args:
            disease_model: Trained PyTorch model for health/disease classification.
            yolo_model: YOLO model for leaf bounding-box detection.
            idx_to_disease: Mapping from class index to disease name.
            img_size: Input image size expected by the disease model.
            yolo_conf: Confidence threshold for YOLO leaf detections.
            health_threshold: Sigmoid threshold above which a plant is considered
                diseased (default 0.5). Lower values increase sensitivity.
            device: Target device ('cuda' or 'cpu'). Auto-detected if None.

        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.disease_model = disease_model.to(self.device).eval()
        self.yolo = yolo_model
        self.idx_to_disease = idx_to_disease
        self.num_diseases = len(idx_to_disease)
        self.yolo_conf = yolo_conf
        self.health_threshold = health_threshold

        self.transform = get_plantvillage_inference_pipeline(img_size)

    @classmethod
    def from_checkpoints(
        cls,
        disease_checkpoint: str | Path,
        yolo_checkpoint: str | Path,
        device: str | None = None,
        *,
        yolo_conf: float = 0.25,
        health_threshold: float = DEFAULT_HEALTH_THRESHOLD,
        verbose: bool = True,
    ) -> DiseasePlantClassifier:
        """Load a DiseasePlantClassifier from checkpoint files.

        Args:
            disease_checkpoint: Path to the disease model checkpoint.
            yolo_checkpoint: Path to the YOLO leaf-detection model checkpoint.
            device: Target device. Auto-detected if None.
            yolo_conf: Confidence threshold for YOLO detections.
            health_threshold: Sigmoid threshold above which a plant is diseased.
            verbose: Whether to print loading progress.

        Returns:
            DiseasePlantClassifier: Initialised classifier with weights loaded.

        """
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "ultralytics is required for DiseasePlantClassifier. "
                "Install it with: pip install ultralytics"
            )
            raise ImportError(msg) from exc

        disease_checkpoint = Path(disease_checkpoint)
        ckpt = torch.load(disease_checkpoint, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})

        variant = config.get("variant", "b0")
        img_size = config.get("img_size", 224)
        num_diseases = ckpt.get("num_diseases") or config.get("num_diseases")
        if num_diseases is None:
            msg = (
                "Cannot determine num_diseases from checkpoint. "
                "Checkpoint must contain 'num_diseases' or 'config.num_diseases'."
            )
            raise ValueError(msg)

        model = DiseasePlantModel(
            model_name=config.get("model_name", "efficientnetv2"),
            num_diseases=num_diseases,
            variant=variant,
            pretrained=False,
        )
        model.load_weights(disease_checkpoint)

        idx_di = ckpt.get("idx_to_disease") or {
            int(v): k for k, v in ckpt.get("disease_to_idx", {}).items()
        }

        yolo = YOLO(str(yolo_checkpoint))

        obj = cls(
            disease_model=model,
            yolo_model=yolo,
            idx_to_disease=idx_di,
            img_size=img_size,
            yolo_conf=yolo_conf,
            health_threshold=health_threshold,
            device=device,
        )

        if verbose:
            print(
                f"Disease model : {disease_checkpoint.name}  "
                f"(variant={config.get('model_name', 'efficientnetv2')}_{variant})"
            )
            print(f"YOLO model    : {Path(yolo_checkpoint).name}")
            print(f"Diseases      : {num_diseases}")

        return obj

    def _detect_leaves(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        """Run YOLO leaf detection on an image.

        Args:
            image: Input PIL image to detect leaves in.

        Returns:
            list[tuple[int, int, int, int]]: Bounding boxes as (x1, y1, x2, y2).

        """
        results = self.yolo.predict(image, conf=self.yolo_conf, verbose=False)
        boxes: list[tuple[int, int, int, int]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().tolist():
                x1, y1, x2, y2 = [int(v) for v in box]
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        return boxes

    @staticmethod
    def _crop_leaves(
        image: Image.Image,
        boxes: list[tuple[int, int, int, int]],
        padding: int = 8,
    ) -> list[Image.Image]:
        """Crop leaf regions from an image with optional padding.

        Args:
            image: Source PIL image to crop from.
            boxes: Bounding boxes as (x1, y1, x2, y2).
            padding: Pixels to expand each bounding box by.

        Returns:
            list[Image.Image]: Cropped leaf images.

        """
        w, h = image.size
        crops = []
        for bx1, by1, bx2, by2 in boxes:
            cx1 = max(0, bx1 - padding)
            cy1 = max(0, by1 - padding)
            cx2 = min(w, bx2 + padding)
            cy2 = min(h, by2 + padding)
            crops.append(image.crop((cx1, cy1, cx2, cy2)))
        return crops

    _MAX_LEAF_BATCH = 16

    def _classify_batch(self, crops: list[Image.Image]) -> dict[str, torch.Tensor]:
        all_health: list[torch.Tensor] = []
        all_disease: list[torch.Tensor] = []
        for i in range(0, len(crops), self._MAX_LEAF_BATCH):
            chunk = crops[i : i + self._MAX_LEAF_BATCH]
            tensors = torch.stack([self.transform(c) for c in chunk]).to(self.device)
            with torch.no_grad():
                out = self.disease_model(tensors)
            all_health.append(out["health"])
            all_disease.append(out["disease"])
        return {
            "health": torch.cat(all_health, dim=0),
            "disease": torch.cat(all_disease, dim=0),
        }

    def predict(
        self,
        image: str | Path | Image.Image,
        top_k_diseases: int = 3,
    ) -> dict[str, Any]:
        """Run disease and health inference on an image.

        Args:
            image: Input image as a file path or PIL Image.
            top_k_diseases: Number of top disease predictions to return.

        Returns:
            dict: Contains 'health' summary, 'diseases' list, 'leaf_count',
                'used_full_image_fallback', 'leaf_results', and
                'processing_time_ms'.

        """
        t0 = time.time()

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        boxes = self._detect_leaves(image)
        crops = self._crop_leaves(image, boxes) if boxes else [image]
        used_full_image_fallback = len(boxes) == 0

        out = self._classify_batch(crops)

        health_logits = torch.sigmoid(out["health"]).squeeze(1)
        disease_probs = torch.softmax(out["disease"], dim=1)

        max_health_logit = float(health_logits.max().item())
        health_label = "diseased" if max_health_logit >= self.health_threshold else "healthy"
        health_confidence = (
            max_health_logit
            if max_health_logit >= self.health_threshold
            else 1.0 - float(health_logits.min().item())
        )

        if used_full_image_fallback:
            health_confidence *= 0.7

        # Weight by per-leaf health probability so healthy leaves contribute less
        eps = 1e-8
        weights = health_logits.unsqueeze(1).clamp(min=eps)
        aggregated = (disease_probs * weights).sum(dim=0)
        denom = aggregated.sum()
        aggregated = aggregated / denom if denom > eps else disease_probs.mean(dim=0)

        top_di_probs, top_di_idx = aggregated.topk(min(top_k_diseases, self.num_diseases))
        disease_preds = [
            {
                "disease": self.idx_to_disease[int(i)],
                "confidence": float(p),
            }
            for p, i in zip(top_di_probs.tolist(), top_di_idx.tolist(), strict=False)
        ]

        leaf_results = self._build_leaf_results(
            crops, boxes, health_logits, disease_probs, used_full_image_fallback
        )

        return {
            "health": {
                "label": health_label,
                "confidence": round(health_confidence, 4),
                "logit": round(max_health_logit, 4),
            },
            "diseases": disease_preds,
            "leaf_count": len(crops),
            "used_full_image_fallback": used_full_image_fallback,
            "leaf_results": leaf_results,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }

    def _build_leaf_results(
        self,
        crops: list[Image.Image],
        boxes: list[tuple[int, int, int, int]],
        health_logits: torch.Tensor,
        disease_probs: torch.Tensor,
        used_full_image_fallback: bool,
    ) -> list[dict[str, Any]]:
        """Build per-leaf result dicts from batch model outputs.

        Args:
            crops: Cropped leaf images.
            boxes: Original bounding boxes for each crop.
            health_logits: Per-crop health sigmoid outputs.
            disease_probs: Per-crop disease softmax outputs.
            used_full_image_fallback: Whether the full image was used as a single crop.

        Returns:
            list[dict]: Per-leaf result dicts.

        """
        leaf_results = []
        for k in range(len(crops)):
            hlp = float(health_logits[k].item())
            di_p, di_i = disease_probs[k].topk(1)
            leaf_results.append(
                {
                    "leaf_index": k,
                    "bbox": boxes[k] if not used_full_image_fallback else None,
                    "health_logit": hlp,
                    "health_label": "diseased" if hlp >= self.health_threshold else "healthy",
                    "top_disease": self.idx_to_disease[int(di_i[0])],
                    "top_disease_conf": float(di_p[0]),
                }
            )
        return leaf_results
