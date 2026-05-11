"""Load models unit tests module.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

import pytest
import torch

from plant_care_ai.models.efficientnetv2 import EfficientNetV2
from plant_care_ai.models.load_models import get_model
from plant_care_ai.models.resnet18 import Resnet18
from plant_care_ai.models.resnet50 import Resnet50


class TestGetModel:
    """Tests for the get_model factory function."""

    @staticmethod
    def test_get_resnet18_default() -> None:
        """Test loading ResNet18 with default parameters."""
        model = get_model("resnet18")
        assert isinstance(model, Resnet18)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1081)

    @staticmethod
    def test_get_resnet18_custom_classes() -> None:
        """Test loading ResNet18 with custom num_classes."""
        model = get_model("resnet18", num_classes=100)
        assert isinstance(model, Resnet18)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 100)

    @staticmethod
    def test_get_resnet50_pretrained_disabled() -> None:
        """Test loading ResNet50 with pretrained disabled."""
        model = get_model("resnet50", num_classes=10, pretrained=False)
        assert isinstance(model, Resnet50)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 10)

    @staticmethod
    def test_get_efficientnetv2_default() -> None:
        """Test loading EfficientNetV2 with default parameters."""
        model = get_model("efficientnetv2")
        assert isinstance(model, EfficientNetV2)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1081)

    @staticmethod
    def test_get_efficientnetv2_with_variant() -> None:
        """Test loading EfficientNetV2 with specific variant."""
        model = get_model("efficientnetv2", num_classes=50, variant="b0")
        assert isinstance(model, EfficientNetV2)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 50)

    @staticmethod
    @pytest.mark.parametrize(
        "model_name",
        [
            "ResNet18",
            "RESNET18",
            "  resnet18  ",
            "EfficientNetV2",
            "EFFICIENTNETV2",
            "efficientnet",
        ],
    )
    def test_case_insensitive_model_names(model_name: str) -> None:
        """Test that model names are case-insensitive and trimmed.

        Args:
            model_name: Various casings of model name

        """
        model = get_model(model_name, num_classes=10)
        assert model is not None

    @staticmethod
    def test_invalid_model_name_raises() -> None:
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            get_model("invalid_model_name")

    @staticmethod
    def test_model_moved_to_cpu() -> None:
        """Test model is on CPU by default."""
        model = get_model("resnet18", num_classes=10)
        for param in model.parameters():
            assert param.device.type == "cpu"

    @staticmethod
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_moved_to_cuda() -> None:
        """Test model can be moved to CUDA."""
        model = get_model("resnet18", num_classes=10, device="cuda")
        for param in model.parameters():
            assert param.device.type == "cuda"


class TestGetModelWithWeights:
    """Tests for loading models with pretrained weights."""

    @staticmethod
    def test_load_weights_from_state_dict(saved_resnet_weights: Path) -> None:
        """Test loading weights from plain state_dict.

        Args:
            saved_resnet_weights: Path to saved weights

        """
        model = get_model("resnet18", num_classes=10, weights_path=str(saved_resnet_weights))
        assert isinstance(model, Resnet18)
        # Verify model can forward pass
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 10)

    @staticmethod
    def test_load_weights_from_checkpoint(saved_resnet_weights_wrapped: Path) -> None:
        """Test loading weights from checkpoint with state_dict key.

        Args:
            saved_resnet_weights_wrapped: Path to saved checkpoint

        """
        model = get_model(
            "resnet18",
            num_classes=10,
            weights_path=str(saved_resnet_weights_wrapped),
        )
        assert isinstance(model, Resnet18)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 10)

    @staticmethod
    def test_nonexistent_weights_path_prints_warning(capsys: pytest.CaptureFixture[str]) -> None:
        """Test that missing weights file prints warning but still creates model.

        Args:
            capsys: Pytest stdout/stderr capture fixture

        """
        model = get_model(
            "resnet18",
            num_classes=10,
            weights_path="/nonexistent/path/weights.pth",
        )
        assert isinstance(model, Resnet18)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "not found" in captured.out


class TestLoadModelsEffectiveWeight:
    """Test that loaded weights are actually used."""

    @staticmethod
    def test_loaded_weights_differ_from_random(tmp_path: Path) -> None:
        """Test loaded weights are different from randomly initialized.

        Args:
            tmp_path: Pytest temporary directory fixture

        """
        # Create and save a model
        original_model = Resnet18(num_classes=10)
        # Modify some weights to make them distinguishable
        with torch.no_grad():
            original_model.fc.weight.fill_(0.5)
            original_model.fc.bias.fill_(0.1)

        weights_path = tmp_path / "modified_weights.pth"
        torch.save(original_model.state_dict(), weights_path)

        # Load the model with weights
        loaded_model = get_model("resnet18", num_classes=10, weights_path=str(weights_path))

        # Create a fresh random model
        random_model = Resnet18(num_classes=10)

        # Check that loaded model has the specific weights
        assert torch.allclose(loaded_model.fc.weight, torch.full_like(loaded_model.fc.weight, 0.5))
        assert torch.allclose(loaded_model.fc.bias, torch.full_like(loaded_model.fc.bias, 0.1))

        # Random model should NOT have these specific weights
        assert not torch.allclose(
            random_model.fc.weight,
            torch.full_like(random_model.fc.weight, 0.5),
        )


class TestLoadModelsConsistency:
    """Test model loading consistency across different scenarios."""

    @staticmethod
    def test_same_weights_produce_same_output(tmp_path: Path) -> None:
        """Test that loading same weights produces identical outputs.

        Args:
            tmp_path: Pytest temporary directory fixture

        """
        torch.manual_seed(42)

        # Create and save model
        original = Resnet18(num_classes=10)
        weights_path = tmp_path / "weights.pth"
        torch.save(original.state_dict(), weights_path)

        # Load same weights twice
        loaded1 = get_model("resnet18", num_classes=10, weights_path=str(weights_path))
        loaded2 = get_model("resnet18", num_classes=10, weights_path=str(weights_path))

        # Test output
        loaded1.eval()
        loaded2.eval()

        x = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            out1 = loaded1(x)
            out2 = loaded2(x)

        assert torch.allclose(out1, out2, atol=1e-6)
