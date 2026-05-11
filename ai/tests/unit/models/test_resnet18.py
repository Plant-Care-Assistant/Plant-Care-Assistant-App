"""ResNet-18 model unit tests.

Copyright 2025 Plant Care Assistant
"""

import torch
from torch import nn

from plant_care_ai.models.resnet18 import BasicBlock, Resnet18

SHORTCUT_LAYER_COUNT = 2
RESNET18_MIN_PARAMS = 10_000_000
RESNET18_MAX_PARAMS = 12_000_000
MIN_OVERFIT_ACCURACY = 0.75


class TestBasicBlock:
    """Tests for the BasicBlock residual block."""

    @staticmethod
    def test_basic_block_output_shape_no_downsample() -> None:
        """Test BasicBlock maintains shape when stride=1 and channels match."""
        block = BasicBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 56, 56)
        out = block(x)
        assert out.shape == x.shape

    @staticmethod
    def test_basic_block_output_shape_with_downsample() -> None:
        """Test BasicBlock downsamples correctly with stride=2."""
        block = BasicBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 56, 56)
        out = block(x)
        assert out.shape == (2, 128, 28, 28)

    @staticmethod
    def test_basic_block_shortcut_created_for_channel_mismatch() -> None:
        """Test that shortcut projection is created when channels differ."""
        block = BasicBlock(in_channels=64, out_channels=128, stride=1)
        assert len(block.shortcut) == SHORTCUT_LAYER_COUNT  # Conv2d + BatchNorm2d

    @staticmethod
    def test_basic_block_shortcut_identity_when_matching() -> None:
        """Test shortcut is identity when channels and stride match."""
        block = BasicBlock(in_channels=64, out_channels=64, stride=1)
        assert len(block.shortcut) == 0  # Empty Sequential


class TestResnet18:
    """Tests for the Resnet18 model."""

    @staticmethod
    def test_resnet18_output_shape(resnet18_model: Resnet18) -> None:
        """Test that Resnet18 produces correct output shape.

        Args:
            resnet18_model: Resnet18 model fixture

        """
        x = torch.randn(4, 3, 224, 224)
        out = resnet18_model(x)
        assert out.shape == (4, 10)

    @staticmethod
    def test_resnet18_default_num_classes() -> None:
        """Test Resnet18 default num_classes is 1081 (PlantNet)."""
        model = Resnet18()
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1081)

    @staticmethod
    def test_resnet18_variable_input_size(resnet18_model: Resnet18) -> None:
        """Test that Resnet18 handles different input sizes due to AdaptiveAvgPool.

        Args:
            resnet18_model: Resnet18 model fixture

        """
        # Smaller input
        x_small = torch.randn(2, 3, 128, 128)
        out_small = resnet18_model(x_small)
        assert out_small.shape == (2, 10)

        # Larger input
        x_large = torch.randn(2, 3, 512, 512)
        out_large = resnet18_model(x_large)
        assert out_large.shape == (2, 10)

    @staticmethod
    def test_freeze_backbone_freezes_all_except_fc(resnet18_model: Resnet18) -> None:
        """Test that freeze_backbone freezes all layers except fc.

        Args:
            resnet18_model: Resnet18 model fixture

        """
        resnet18_model.freeze_backbone()

        for name, param in resnet18_model.named_parameters():
            if "fc" in name:
                assert param.requires_grad is True, f"{name} should be trainable"
            else:
                assert param.requires_grad is False, f"{name} should be frozen"

    @staticmethod
    def test_unfreeze_backbone_unfreezes_all(resnet18_model: Resnet18) -> None:
        """Test that unfreeze_backbone makes all params trainable.

        Args:
            resnet18_model: Resnet18 model fixture

        """
        resnet18_model.freeze_backbone()
        resnet18_model.unfreeze_backbone()

        for name, param in resnet18_model.named_parameters():
            assert param.requires_grad is True, f"{name} should be trainable"

    @staticmethod
    def test_resnet18_layer_structure(resnet18_model: Resnet18) -> None:
        """Test that Resnet18 has correct layer structure.

        Args:
            resnet18_model: Resnet18 model fixture

        """
        assert isinstance(resnet18_model.conv1, nn.Conv2d)
        assert isinstance(resnet18_model.bn1, nn.BatchNorm2d)
        assert isinstance(resnet18_model.maxpool, nn.MaxPool2d)
        assert isinstance(resnet18_model.layer1, nn.Sequential)
        assert isinstance(resnet18_model.layer2, nn.Sequential)
        assert isinstance(resnet18_model.layer3, nn.Sequential)
        assert isinstance(resnet18_model.layer4, nn.Sequential)
        assert isinstance(resnet18_model.avgpool, nn.AdaptiveAvgPool2d)
        assert isinstance(resnet18_model.fc, nn.Linear)

    @staticmethod
    def test_resnet18_trainable_params_count() -> None:
        """Test that Resnet18 has expected parameter count."""
        model = Resnet18(num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        # ResNet-18 has approximately 11M parameters
        assert RESNET18_MIN_PARAMS < total_params < RESNET18_MAX_PARAMS


class TestResnet18OverfitSingleBatch:
    """Test that Resnet18 can overfit on a single batch (learning capability)."""

    @staticmethod
    def test_overfit_single_batch() -> None:
        """Test model can overfit a single batch, proving it can learn."""
        torch.manual_seed(42)

        num_classes = 5
        model = Resnet18(num_classes=num_classes)
        model.train()

        # Create a small batch
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)  # Smaller images for speed
        y = torch.randint(0, num_classes, (batch_size,))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        # Train for multiple iterations on the same batch
        num_iterations = 100
        for i in range(num_iterations):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)

            if i == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Assert that loss decreased significantly
        assert final_loss is not None
        assert initial_loss is not None
        assert final_loss < initial_loss * 0.1, (
            f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )

        # Check model can predict the batch correctly
        model.eval()
        with torch.no_grad():
            predictions = model(x).argmax(dim=1)
            accuracy = (predictions == y).float().mean().item()

        assert accuracy >= MIN_OVERFIT_ACCURACY, (
            f"Model should overfit to at least 75% accuracy, got {accuracy:.2%}"
        )

    @staticmethod
    def test_gradient_flow() -> None:
        """Test that gradients flow through all layers."""
        model = Resnet18(num_classes=10)
        model.train()

        x = torch.randn(2, 3, 64, 64)
        y = torch.randint(0, 10, (2,))

        outputs = model(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
