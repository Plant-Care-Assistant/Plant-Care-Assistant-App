"""Multi-task model supporting ResNet50 or EfficientNetV2 backbones.

Copyright 2026 Plant Care Assistant
"""
import pathlib

import timm
import torch
from torch import nn
from torchvision import models


class MultiTaskPlantModel(nn.Module):
    """Multi-task plant model with a shared backbone and three task heads."""

    SUPPORTED = ("resnet50", "efficientnetv2")
    EFFICIENTNETV2_VARIANTS = ("b0", "b1", "b2", "b3", "l", "m")

    def __init__(
        self,
        model_name: str = "resnet50",
        num_species: int = 1087,
        num_diseases: int = 38,
        dropout: float = 0.4,
        *,
        pretrained: bool = True,
        variant: str = "b0",
    ) -> None:
        super().__init__()

        self.model_name = model_name.lower().strip()
        if self.model_name not in self.SUPPORTED:
            raise ValueError(
                f"model_name '{self.model_name}' not supported. "
                f"Choose one of: {self.SUPPORTED}"
            )

        if self.model_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            if variant not in self.EFFICIENTNETV2_VARIANTS:
                raise ValueError(
                    f"variant '{variant}' not supported. "
                    f"Choose one of: {self.EFFICIENTNETV2_VARIANTS}"
                )
            
            self.backbone = timm.create_model(
                f"tf_efficientnetv2_{variant}",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            in_features = self.backbone.num_features

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head_species = nn.Linear(512, num_species)
        self.head_health  = nn.Linear(512, 1)
        self.head_disease = nn.Linear(512, num_diseases)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits for all three heads.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Dictionary with keys ``'species'``, ``'health'``, ``'disease'``.
        """
        shared = self.bottleneck(self.backbone(x))
        return {
            "species": self.head_species(shared),
            "health":  self.head_health(shared),
            "disease": self.head_disease(shared),
        }

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, from_layer: int = 0) -> None:
        if self.model_name == "resnet50":
            for child in list(self.backbone.children())[from_layer:]:
                for param in child.parameters():
                    param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def predict(
        self,
        x: torch.Tensor,
        disease_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Two-stage inference: health check -> disease classification if diseased"""

        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            health_prob = torch.sigmoid(out["health"]).squeeze(1)
            disease_prob, disease_idx = torch.softmax(out["disease"], dim=1).max(dim=1)

        return {
            "species_idx":  out["species"].argmax(dim=1),
            "health_prob":  health_prob,
            "is_diseased":  health_prob >= disease_threshold,
            "disease_idx":  disease_idx,
            "disease_prob": disease_prob,
        }

    def load_weights(self, weights_path: str, device: str = "cpu") -> None:
        if not pathlib.Path(weights_path).exists():
            print(f"Warning: weights file not found at {weights_path}.")
            return

        checkpoint = torch.load(weights_path, map_location=device)
        self.load_state_dict(
            checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint,
            strict=False,
        )