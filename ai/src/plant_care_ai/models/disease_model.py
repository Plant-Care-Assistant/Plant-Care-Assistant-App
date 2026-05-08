"""Model for plant disease and health classification.

Copyright 2026 Plant Care Assistant
"""
from pathlib import Path

import timm
import torch
from torch import nn
from torchvision import models


class DiseasePlantModel(nn.Module):
    """Dual-head model for plant disease classification and health scoring.

    Supports ResNet-50 and EfficientNetV2 backbones. Outputs a disease logit
    vector and a scalar health probability from a shared bottleneck.
    """

    def __init__(
        self,
        model_name: str = "efficientnetv2",
        num_diseases: int = 38,
        dropout: float = 0.4,
        *,
        variant: str = "b0",
        pretrained: bool = True,
    ) -> None:
        """Initialize DiseasePlantModel.

        Args:
            model_name: Backbone architecture; ``"resnet50"`` or ``"efficientnetv2"``.
            num_diseases: Number of disease classes for the classification head.
            dropout: Dropout probability applied in the bottleneck.
            variant: EfficientNetV2 variant suffix (e.g. ``"b0"``, ``"s"``).
                     Ignored when ``model_name`` is ``"resnet50"``.
            pretrained: Whether to load ImageNet-pretrained backbone weights.

        """
        super().__init__()

        self.model_name = model_name

        if self.model_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
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

        self.head_health = nn.Linear(512, 1)  # probability of being sick
        self.head_disease = nn.Linear(512, num_diseases)  # disease name

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a forward pass and return disease logits and health score.

        Args:
            x: Input image tensor of shape ``(N, C, H, W)``.

        Returns:
            Dictionary with keys:

            - ``"disease"``: raw logits of shape ``(N, num_diseases)``.
            - ``"health"``: scalar health logit of shape ``(N, 1)``.

        """
        feat = self.bottleneck(self.backbone(x))

        return {
            "disease": self.head_disease(feat),
            "health": self.head_health(feat),
        }

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters to prevent gradient updates."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, from_layer: int = 0) -> None:
        """Unfreeze backbone parameters, optionally starting from a given layer index.

        For ResNet-50, only children from ``from_layer`` onward are unfrozen.
        For EfficientNetV2, all backbone parameters are unfrozen regardless of
        ``from_layer``.

        Args:
            from_layer: Index of the first child layer to unfreeze (ResNet-50 only).

        """
        if self.model_name == "resnet50":
            for child in list(self.backbone.children())[from_layer:]:
                for p in child.parameters():
                    p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def load_weights(self, path: str | Path, device: str = "cpu") -> None:
        """Load model weights from a checkpoint file.

        Silently returns if the file does not exist. Accepts both raw state-dict
        files and checkpoint dicts that contain a ``"model_state_dict"`` key.
        Missing or unexpected keys are tolerated via ``strict=False``.

        Args:
            path: Path to the checkpoint file.
            device: Device string passed to ``torch.load`` (e.g. ``"cpu"`` or ``"cuda"``).

        """
        path = Path(path)
        if not path.exists():
            return
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.load_state_dict(state, strict=False)
