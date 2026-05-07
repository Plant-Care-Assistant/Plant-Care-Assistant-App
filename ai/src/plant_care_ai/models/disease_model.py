"""Model for plant disease and health classification.

Copyright 2026 Plant Care Assistant
"""
from pathlib import Path
import timm
import torch
from torch import nn
from torchvision import models


class DiseasePlantModel(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnetv2",
        num_diseases: int = 38,
        dropout: float = 0.4,
        *,
        variant: str = "b0",
        pretrained: bool = True,
    ):
        super().__init__()

        self.model_name = model_name

        if self.model_name == "resnet50":
            _backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_features = _backbone.fc.in_features
            self.backbone = nn.Sequential(*list(_backbone.children())[:-1])
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

        self.head_health = nn.Linear(512, 1)          # probability of being sick
        self.head_disease = nn.Linear(512, num_diseases)  # disease name

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.bottleneck(self.backbone(x))

        return {
            "disease": self.head_disease(feat),
            "health": self.head_health(feat),
        }

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, from_layer: int = 0) -> None:
        if self.model_name == "resnet50":
            for child in list(self.backbone.children())[from_layer:]:
                for p in child.parameters():
                    p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def load_weights(self, path: str | Path, device: str = "cpu") -> None:
        path = Path(path)
        if not path.exists():
            return
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.load_state_dict(state, strict=False)