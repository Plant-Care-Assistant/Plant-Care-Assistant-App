"""Multi-task ResNet50: species classification + binary health + specific disease.

Copyright 2026 Plant Care Assistant. All rights reserved.
"""

import torch
from torch import nn
from torchvision import models


class MultitaskResNet(nn.Module):
    """ResNet (temp- resnet50) backbone with three output heads.

    Heads:
        species: num_species-class softmax (PlantNet300k labels)
        disease: num_diseases-class softmax (PlantVillage disease labels)
        health: binary sigmoid (healthy=0 / diseased=1)
    """

    def __init__(
        self,
        num_species: int = 1087,
        num_diseases: int = 38,
        dropout: float = 0.4,
        *,
        pretrained: bool = True,
    ) -> None:
        """Initialize model.

        Args:
            num_species: number of plant species classes
            num_diseases: number of specific disease classes
            dropout: dropout probability in shared bottleneck
            pretrained: load ImageNet pretrained weights for backbone
        """
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features  # 2048

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 2048, 1, 1)

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
        """Forward pass returning all three head logits.

        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            Dict with keys 'species' (B, num_species),
            'health' (B, 1), 'disease' (B, num_diseases).
        """
        feat   = self.backbone(x)
        shared = self.bottleneck(feat)
        return {
            "species": self.head_species(shared),
            "health":  self.head_health(shared),
            "disease": self.head_disease(shared),
        }

    def predict(
        self,
        x: torch.Tensor,
        disease_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Two-stage: binary health check then specific disease.

        Args:
            x: input tensor of shape (B, C, H, W)
            disease_threshold: sigmoid probability cutoff for diseased label
        Returns:
            Dict with species_idx, health_prob, is_diseased,
            disease_idx, disease_prob per sample.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            health_prob  = torch.sigmoid(out["health"]).squeeze(1)
            is_diseased  = health_prob >= disease_threshold
            species_idx  = out["species"].argmax(dim=1)
            disease_prob = torch.softmax(out["disease"], dim=1)
            disease_idx  = disease_prob.argmax(dim=1)
            disease_conf = disease_prob.max(dim=1).values

        return {
            "species_idx":  species_idx,
            "health_prob":  health_prob,
            "is_diseased":  is_diseased,
            "disease_idx":  disease_idx,
            "disease_prob": disease_conf,
        }

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, from_layer: int = 6) -> None:
        """Unfreeze backbone layers starting from `from_layer` (0-indexed children).

        Args:
            from_layer: index into backbone children list from which to unfreeze;
                default 6 unfreezes layer3 + layer4 of ResNet50.
        """
        for child in list(self.backbone.children())[from_layer:]:
            for param in child.parameters():
                param.requires_grad = True

    def state_dict(self, *args, **kwargs) -> dict:
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict, *args, **kwargs) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)