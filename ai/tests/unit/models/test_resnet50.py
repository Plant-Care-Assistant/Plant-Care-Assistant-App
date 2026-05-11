"""ResNet-50 model unit tests.

Copyright 2025 Plant Care Assistant
"""

from unittest.mock import patch

import torch
from torch import nn
from torchvision import models

from plant_care_ai.models.resnet50 import Resnet50

RESNET50_MIN_PARAMS = 22_000_000
RESNET50_MAX_PARAMS = 26_000_000


class TestResnet50:
    """Tests for the Resnet50 model."""

    @staticmethod
    def test_resnet50_output_shape(resnet50_model: Resnet50) -> None:
        """Test that Resnet50 produces correct output shape.

        Args:
            resnet50_model: Resnet50 model fixture

        """
        x = torch.randn(2, 3, 64, 64)
        out = resnet50_model(x)
        assert out.shape == (2, 10)

    @staticmethod
    def test_resnet50_default_num_classes() -> None:
        """Test Resnet50 default num_classes is 1081 (PlantNet)."""
        model = Resnet50(pretrained=False)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 1081)

    @staticmethod
    def test_resnet50_pretrained_uses_imagenet_weights() -> None:
        """Test pretrained path uses ImageNet weights without downloading."""
        captured: dict[str, object] = {}

        class DummyBackbone(nn.Module):
            """Minimal backbone exposing fc.in_features for Resnet50 init."""

            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        def fake_resnet50(*, weights: object | None = None, **kwargs: object) -> nn.Module:
            captured["weights"] = weights
            assert kwargs == {}
            return DummyBackbone()

        with patch(
            "plant_care_ai.models.resnet50.models.resnet50",
            side_effect=fake_resnet50,
        ) as mock_resnet50:
            model = Resnet50(num_classes=10, pretrained=True)

        assert mock_resnet50.call_count == 1
        assert captured["weights"] == models.ResNet50_Weights.IMAGENET1K_V2
        assert isinstance(model.backbone, nn.Module)

    @staticmethod
    def test_resnet50_variable_input_size(resnet50_model: Resnet50) -> None:
        """Test that Resnet50 handles different input sizes via AdaptiveAvgPool.

        Args:
            resnet50_model: Resnet50 model fixture

        """
        x_small = torch.randn(1, 3, 64, 64)
        out_small = resnet50_model(x_small)
        assert out_small.shape == (1, 10)

        x_large = torch.randn(1, 3, 160, 160)
        out_large = resnet50_model(x_large)
        assert out_large.shape == (1, 10)

    @staticmethod
    def test_freeze_backbone_freezes_all_except_fc(resnet50_model: Resnet50) -> None:
        """Test that freeze_backbone freezes all layers except fc.

        Args:
            resnet50_model: Resnet50 model fixture

        """
        resnet50_model.freeze_backbone()

        for name, param in resnet50_model.named_parameters():
            if "fc" in name:
                assert param.requires_grad is True, f"{name} should be trainable"
            else:
                assert param.requires_grad is False, f"{name} should be frozen"

    @staticmethod
    def test_unfreeze_backbone_unfreezes_all(resnet50_model: Resnet50) -> None:
        """Test that unfreeze_backbone makes all params trainable.

        Args:
            resnet50_model: Resnet50 model fixture

        """
        resnet50_model.freeze_backbone()
        resnet50_model.unfreeze_backbone()

        for name, param in resnet50_model.named_parameters():
            assert param.requires_grad is True, f"{name} should be trainable"

    @staticmethod
    def test_resnet50_layer_structure(resnet50_model: Resnet50) -> None:
        """Test that Resnet50 has expected layer structure.

        Args:
            resnet50_model: Resnet50 model fixture

        """
        assert isinstance(resnet50_model.backbone.conv1, nn.Conv2d)
        assert isinstance(resnet50_model.backbone.bn1, nn.BatchNorm2d)
        assert isinstance(resnet50_model.backbone.maxpool, nn.MaxPool2d)
        assert isinstance(resnet50_model.backbone.layer1, nn.Sequential)
        assert isinstance(resnet50_model.backbone.layer2, nn.Sequential)
        assert isinstance(resnet50_model.backbone.layer3, nn.Sequential)
        assert isinstance(resnet50_model.backbone.layer4, nn.Sequential)
        assert isinstance(resnet50_model.backbone.avgpool, nn.AdaptiveAvgPool2d)
        assert isinstance(resnet50_model.backbone.fc, nn.Linear)

    @staticmethod
    def test_resnet50_trainable_params_count() -> None:
        """Test that Resnet50 has expected parameter count."""
        model = Resnet50(num_classes=10, pretrained=False)
        total_params = sum(p.numel() for p in model.parameters())
        assert RESNET50_MIN_PARAMS < total_params < RESNET50_MAX_PARAMS

    @staticmethod
    def test_state_dict_round_trip() -> None:
        """Test that state_dict and load_state_dict round-trip works."""
        model = Resnet50(num_classes=10, pretrained=False)
        with torch.no_grad():
            model.backbone.fc.weight.fill_(0.25)
            model.backbone.fc.bias.fill_(0.1)

        state_dict = model.state_dict()

        new_model = Resnet50(num_classes=10, pretrained=False)
        new_model.load_state_dict(state_dict)

        assert torch.allclose(
            new_model.backbone.fc.weight,
            torch.full_like(new_model.backbone.fc.weight, 0.25),
        )
        assert torch.allclose(
            new_model.backbone.fc.bias,
            torch.full_like(new_model.backbone.fc.bias, 0.1),
        )
