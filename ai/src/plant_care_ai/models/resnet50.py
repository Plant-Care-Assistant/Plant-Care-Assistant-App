"""ResNet-50 with pretrained ImageNet weights from torchvision.

Copyright (c) 2026 Plant Care Assistant. All rights reserved.
"""

import torch
from torch import nn
from torchvision import models


class Resnet50(nn.Module):
    """ResNet-50 wrapper using torchvision pretrained weights."""

    def __init__(self, num_classes: int = 1081, *, pretrained: bool = True) -> None:
        """Initialize ResNet-50 with pretrained ImageNet weights.

        Args:
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained ImageNet weights.

        """
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, num_classes).

        """
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the final fc layer."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def state_dict(self, *args, **kwargs) -> dict:
        """Return state dict of the backbone.

        Returns:
            State dict of the backbone model.

        """
        return self.backbone.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict, *args, **kwargs) -> None:
        """Load state dict into backbone.

        Handles both formats:
        - Standard: fc.weight, fc.bias
        - Sequential: fc.1.weight, fc.1.bias (from some training scripts)
        """
        # Check if state_dict uses Sequential FC format (fc.1.*)
        if "fc.1.weight" in state_dict and "fc.weight" not in state_dict:
            # Remap fc.1.* -> fc.*
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("fc.1."):
                    new_key = key.replace("fc.1.", "fc.")
                    new_state_dict[new_key] = value
                elif key.startswith("fc.0."):
                    # Skip BatchNorm or other layers in Sequential FC
                    continue
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        self.backbone.load_state_dict(state_dict, *args, **kwargs)
