"""EfficientNetV2 model unit tests.

Copyright 2025 Plant Care Assistant
"""

import pytest
import torch
from torch import nn

from plant_care_ai.models.efficientnetv2 import (
    EfficientNetV2,
    FusedMBConv,
    MBConv,
    SqueezeExcitation,
    StochasticDepth,
    create_efficientnetv2,
)

EXPECTED_SE_REDUCED_CHANNELS = 16
EXPECTED_ROUND_CHANNELS_24 = 24
EXPECTED_ROUND_CHANNELS_40 = 40
EXPECTED_ROUND_CHANNELS_16 = 16
EXPECTED_ROUND_REPEATS_2 = 2
EXPECTED_ROUND_REPEATS_3 = 3
MIN_OVERFIT_ACCURACY = 0.5


class TestStochasticDepth:
    """Tests for the StochasticDepth module."""

    @staticmethod
    def test_stochastic_depth_inference_no_drop() -> None:
        """Test that StochasticDepth passes through in eval mode."""
        sd = StochasticDepth(drop_prob=0.5)
        sd.eval()
        x = torch.randn(2, 64, 32, 32)
        out = sd(x)
        assert torch.allclose(out, x)

    @staticmethod
    def test_stochastic_depth_zero_prob_training() -> None:
        """Test that StochasticDepth with 0 prob passes through in training."""
        sd = StochasticDepth(drop_prob=0.0)
        sd.train()
        x = torch.randn(2, 64, 32, 32)
        out = sd(x)
        assert torch.allclose(out, x)

    @staticmethod
    def test_stochastic_depth_output_shape() -> None:
        """Test StochasticDepth maintains shape."""
        sd = StochasticDepth(drop_prob=0.2)
        sd.train()
        x = torch.randn(4, 128, 16, 16)
        out = sd(x)
        assert out.shape == x.shape


class TestSqueezeExcitation:
    """Tests for the SqueezeExcitation block."""

    @staticmethod
    def test_se_output_shape() -> None:
        """Test SE block maintains input shape."""
        se = SqueezeExcitation(in_channels=64, se_ratio=0.25)
        x = torch.randn(2, 64, 32, 32)
        out = se(x)
        assert out.shape == x.shape

    @staticmethod
    def test_se_reduced_channels() -> None:
        """Test SE block has correct reduced channel count."""
        se = SqueezeExcitation(in_channels=64, se_ratio=0.25)
        # The first conv in excitation should have output channels = 64 * 0.25 = 16
        first_conv = se.excitation[0]
        assert first_conv.out_channels == EXPECTED_SE_REDUCED_CHANNELS

    @staticmethod
    def test_se_minimum_channels() -> None:
        """Test SE block uses at least 1 channel in bottleneck."""
        se = SqueezeExcitation(in_channels=2, se_ratio=0.25)
        first_conv = se.excitation[0]
        assert first_conv.out_channels >= 1


class TestFusedMBConv:
    """Tests for the FusedMBConv block."""

    @staticmethod
    def test_fused_mbconv_output_shape_stride1() -> None:
        """Test FusedMBConv with stride=1 and same channels uses residual."""
        block = FusedMBConv(in_channels=32, out_channels=32, expand_ratio=4, stride=1)
        x = torch.randn(2, 32, 56, 56)
        out = block(x)
        assert out.shape == x.shape
        assert block.use_residual is True

    @staticmethod
    def test_fused_mbconv_output_shape_stride2() -> None:
        """Test FusedMBConv with stride=2 downsamples and no residual."""
        block = FusedMBConv(in_channels=32, out_channels=64, expand_ratio=4, stride=2)
        x = torch.randn(2, 32, 56, 56)
        out = block(x)
        assert out.shape == (2, 64, 28, 28)
        assert block.use_residual is False

    @staticmethod
    def test_fused_mbconv_with_drop_path() -> None:
        """Test FusedMBConv with stochastic depth."""
        block = FusedMBConv(
            in_channels=32, out_channels=32, expand_ratio=4, stride=1, drop_path_rate=0.2
        )
        assert isinstance(block.drop_path, StochasticDepth)


class TestMBConv:
    """Tests for the MBConv block."""

    @staticmethod
    def test_mbconv_output_shape_stride1() -> None:
        """Test MBConv with stride=1 uses residual."""
        block = MBConv(in_channels=64, out_channels=64, expand_ratio=4, stride=1)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        assert out.shape == x.shape
        assert block.use_residual is True

    @staticmethod
    def test_mbconv_output_shape_stride2() -> None:
        """Test MBConv with stride=2 downsamples."""
        block = MBConv(in_channels=64, out_channels=128, expand_ratio=4, stride=2)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        assert out.shape == (2, 128, 14, 14)
        assert block.use_residual is False

    @staticmethod
    def test_mbconv_has_se_block() -> None:
        """Test MBConv includes SE block when se_ratio > 0."""
        block = MBConv(in_channels=64, out_channels=64, expand_ratio=4, stride=1, se_ratio=0.25)
        # Check that SqueezeExcitation is in the block
        has_se = any(isinstance(m, SqueezeExcitation) for m in block.block.modules())
        assert has_se is True

    @staticmethod
    def test_mbconv_no_se_block() -> None:
        """Test MBConv excludes SE block when se_ratio = 0."""
        block = MBConv(in_channels=64, out_channels=64, expand_ratio=4, stride=1, se_ratio=0.0)
        has_se = any(isinstance(m, SqueezeExcitation) for m in block.block.modules())
        assert has_se is False


class TestEfficientNetV2:
    """Tests for the EfficientNetV2 model."""

    @staticmethod
    def test_efficientnetv2_output_shape(model_b0: EfficientNetV2) -> None:
        """Test EfficientNetV2 produces correct output shape.

        Args:
            model_b0: EfficientNetV2 model fixture

        """
        x = torch.randn(4, 3, 224, 224)
        out = model_b0(x)
        assert out.shape == (4, 10)

    @staticmethod
    def test_efficientnetv2_default_num_classes() -> None:
        """Test EfficientNetV2 default num_classes is 1081."""
        model = EfficientNetV2(variant="b0")
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1081)

    @staticmethod
    @pytest.mark.parametrize("variant", ["b0", "b1", "b2", "b3"])
    def test_efficientnetv2_variants_forward(variant: str) -> None:
        """Test all EfficientNetV2 variants can forward pass.

        Args:
            variant: Model variant string

        """
        model = EfficientNetV2(variant=variant, num_classes=10)
        x = torch.randn(1, 3, 128, 128)  # Smaller for speed
        out = model(x)
        assert out.shape == (1, 10)

    @staticmethod
    def test_efficientnetv2_invalid_variant_raises() -> None:
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            EfficientNetV2(variant="b999", num_classes=10)

    @staticmethod
    def test_efficientnetv2_variable_input_size(model_b0: EfficientNetV2) -> None:
        """Test EfficientNetV2 handles different input sizes.

        Args:
            model_b0: EfficientNetV2 model fixture

        """
        x_small = torch.randn(2, 3, 128, 128)
        out_small = model_b0(x_small)
        assert out_small.shape == (2, 10)

        x_large = torch.randn(2, 3, 384, 384)
        out_large = model_b0(x_large)
        assert out_large.shape == (2, 10)

    @staticmethod
    def test_freeze_stages(model_b0: EfficientNetV2) -> None:
        """Test freeze_stages freezes correct number of stages.

        Args:
            model_b0: EfficientNetV2 model fixture

        """
        model_b0.freeze_stages(num_stages=2)

        # Stem should be frozen
        for param in model_b0.stem.parameters():
            assert param.requires_grad is False

        # First 2 stages should be frozen
        for param in model_b0.stages[0].parameters():
            assert param.requires_grad is False
        for param in model_b0.stages[1].parameters():
            assert param.requires_grad is False

        # Later stages should still be trainable
        for param in model_b0.stages[2].parameters():
            assert param.requires_grad is True

    @staticmethod
    def test_unfreeze_all(model_b0: EfficientNetV2) -> None:
        """Test unfreeze_all makes all parameters trainable.

        Args:
            model_b0: EfficientNetV2 model fixture

        """
        model_b0.freeze_stages(num_stages=3)
        model_b0.unfreeze_all()

        for name, param in model_b0.named_parameters():
            assert param.requires_grad is True, f"{name} should be trainable"

    @staticmethod
    def test_round_channels() -> None:
        """Test _round_channels rounds to nearest divisor."""
        # 24 * 1.0 = 24, already divisible by 8
        assert EfficientNetV2._round_channels(24, 1.0) == EXPECTED_ROUND_CHANNELS_24  # noqa: SLF001
        # 24 * 1.1 = 26.4, rounds to 24
        assert EfficientNetV2._round_channels(24, 1.1) == EXPECTED_ROUND_CHANNELS_24  # noqa: SLF001
        # 24 * 1.5 = 36, rounds to 40
        assert EfficientNetV2._round_channels(24, 1.5) == EXPECTED_ROUND_CHANNELS_40  # noqa: SLF001

    @staticmethod
    def test_round_channels_adds_divisor_when_too_small() -> None:
        """Test _round_channels adds divisor when rounded value is < 90% of scaled."""
        # 9 * 1.0 = 9, rounds to 8, but 8 < 0.9 * 9 = 8.1, so add divisor -> 16
        assert EfficientNetV2._round_channels(9, 1.0) == EXPECTED_ROUND_CHANNELS_16  # noqa: SLF001
        # 17 * 1.0 = 17, rounds to 16, but 16 < 0.9 * 17 = 15.3? No, 16 > 15.3
        # Let's find a case: channels=11, width_mult=1.0 -> 11, rounds to 8
        # 8 < 0.9 * 11 = 9.9, so add divisor -> 16
        assert EfficientNetV2._round_channels(11, 1.0) == EXPECTED_ROUND_CHANNELS_16  # noqa: SLF001

    @staticmethod
    def test_round_repeats() -> None:
        """Test _round_repeats rounds up correctly."""
        assert EfficientNetV2._round_repeats(2, 1.0) == EXPECTED_ROUND_REPEATS_2  # noqa: SLF001
        assert EfficientNetV2._round_repeats(2, 1.1) == EXPECTED_ROUND_REPEATS_3  # noqa: SLF001
        assert EfficientNetV2._round_repeats(4, 0.5) == EXPECTED_ROUND_REPEATS_2  # noqa: SLF001


class TestCreateEfficientNetV2:
    """Tests for the factory function."""

    @staticmethod
    def test_create_efficientnetv2_default() -> None:
        """Test create_efficientnetv2 with defaults."""
        model = create_efficientnetv2()
        assert isinstance(model, EfficientNetV2)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1081)

    @staticmethod
    def test_create_efficientnetv2_custom() -> None:
        """Test create_efficientnetv2 with custom params."""
        model = create_efficientnetv2(variant="b0", num_classes=100, dropout_rate=0.5)
        assert isinstance(model, EfficientNetV2)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 100)


class TestEfficientNetV2OverfitSingleBatch:
    """Test that EfficientNetV2 can overfit on a single batch."""

    @staticmethod
    def test_overfit_single_batch() -> None:
        """Test model can overfit a single batch, proving it can learn."""
        torch.manual_seed(42)

        num_classes = 5
        model = EfficientNetV2(variant="b0", num_classes=num_classes, dropout_rate=0.0)
        model.train()

        # Create a small batch with smaller images for speed
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)
        y = torch.randint(0, num_classes, (batch_size,))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        initial_loss = None
        final_loss = None

        # Train for multiple iterations
        num_iterations = 150
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
        assert final_loss < initial_loss * 0.2, (
            f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )

        # Check model can predict the batch reasonably well
        model.eval()
        with torch.no_grad():
            predictions = model(x).argmax(dim=1)
            accuracy = (predictions == y).float().mean().item()

        assert accuracy >= MIN_OVERFIT_ACCURACY, (
            f"Model should overfit to at least 50% accuracy, got {accuracy:.2%}"
        )

    @staticmethod
    def test_gradient_flow() -> None:
        """Test that gradients flow through all layers."""
        model = EfficientNetV2(variant="b0", num_classes=10)
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
