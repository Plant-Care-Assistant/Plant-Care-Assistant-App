"""EfficientNetV2 implementation for PlantNet-300K.

Based on https://arxiv.org/abs/2104.00298.

Copyright 2026 Plant Care Assistant
"""

import math
from typing import ClassVar

import torch
from torch import nn


class StochasticDepth(nn.Module):
    """Stochastic Depth (Drop Path) for regularization."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize Stochastic Depth.

        Args:
            drop_prob: Probability of dropping the entire path.

        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass.

        Args:
            x: Input tensor.

        Returns:
            The scaled input tensor or zeroed tensor during training.

        """
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation Block."""

    def __init__(self, in_channels: int, se_ratio: float = 0.25) -> None:
        """Initialize SE Block.

        Args:
            in_channels: Number of input channels.
            se_ratio: Ratio to reduce channels in the bottleneck.

        """
        super().__init__()
        reduced_ch = max(1, int(in_channels * se_ratio))

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_ch, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_ch, in_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass.

        Args:
            x: Input feature map.

        Returns:
            Channel-reweighted feature map.

        """
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class FusedMBConv(nn.Module):
    """Fused Mobile Inverted BottleNeck Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        *,
        kernel_size: int = 3,
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize Fused MBConv.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            expand_ratio: Expansion factor for hidden dimension.
            stride: Stride of the convolution.
            kernel_size: Size of the convolutional kernel.
            drop_path_rate: Rate for stochastic depth.

        """
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio
        padding = (kernel_size - 1) // 2

        layers = []

        layers.extend([
            nn.Conv2d(in_channels, hidden_dim, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
            nn.SiLU(inplace=True),
        ])

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        ])

        self.block = nn.Sequential(*layers)
        self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after fused convolution and residual addition.

        """
        if self.use_residual:
            return x + self.drop_path(self.block(x))
        return self.block(x)


class MBConv(nn.Module):
    """Mobile Inverted BottleNeck Convolution with se."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        *,
        kernel_size: int = 3,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize MBConv.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            expand_ratio: Expansion factor.
            stride: Convolutional stride.
            kernel_size: Kernel size.
            se_ratio: Squeeze-and-Excitation ratio.
            drop_path_rate: Stochastic depth rate.

        """
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio
        padding = (kernel_size - 1) // 2

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
                nn.SiLU(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
            nn.SiLU(inplace=True),
        ])

        if se_ratio > 0:
            layers.append(SqueezeExcitation(hidden_dim, se_ratio))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        ])

        self.block = nn.Sequential(*layers)
        self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after MBConv block.

        """
        if self.use_residual:
            return x + self.drop_path(self.block(x))
        return self.block(x)


class EfficientNetV2(nn.Module):
    """EfficientNetV2 architecture implementation."""

    ARCH_CONFIGS: ClassVar[dict[str, list[list]]] = {
        "b0": [
            ["fused", 1, 32, 1, 1, 3, 0],
            ["fused", 4, 64, 2, 2, 3, 0],
            ["fused", 4, 96, 2, 2, 3, 0],
            ["mbconv", 4, 192, 3, 2, 3, 0.25],
            ["mbconv", 6, 256, 5, 1, 3, 0.25],
            ["mbconv", 6, 512, 8, 2, 3, 0.25],
        ],
        "b1": [
            ["fused", 1, 32, 2, 1, 3, 0],
            ["fused", 4, 64, 3, 2, 3, 0],
            ["fused", 4, 96, 3, 2, 3, 0],
            ["mbconv", 4, 192, 4, 2, 3, 0.25],
            ["mbconv", 6, 256, 6, 1, 3, 0.25],
            ["mbconv", 6, 512, 9, 2, 3, 0.25],
        ],
        "b2": [
            ["fused", 1, 32, 2, 1, 3, 0],
            ["fused", 4, 64, 3, 2, 3, 0],
            ["fused", 4, 96, 3, 2, 3, 0],
            ["mbconv", 4, 208, 4, 2, 3, 0.25],
            ["mbconv", 6, 352, 6, 1, 3, 0.25],
            ["mbconv", 6, 640, 10, 2, 3, 0.25],
        ],
        "b3": [
            ["fused", 1, 32, 2, 1, 3, 0],
            ["fused", 4, 64, 4, 2, 3, 0],
            ["fused", 4, 96, 4, 2, 3, 0],
            ["mbconv", 4, 232, 6, 2, 3, 0.25],
            ["mbconv", 6, 384, 9, 1, 3, 0.25],
            ["mbconv", 6, 640, 15, 2, 3, 0.25],
        ],
    }

    def __init__(
        self,
        variant: str = "b3",
        num_classes: int = 1081,
        *,
        dropout_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
    ) -> None:
        """Initialize EfficientNetV2.

        Args:
            variant: Model variant (b0, b1, b2, b3).
            num_classes: Number of output classes.
            dropout_rate: Dropout rate before classifier.
            drop_path_rate: Stochastic depth rate.
            width_mult: Width multiplier.
            depth_mult: Depth multiplier.

        Raises:
            ValueError: If variant is not supported.

        """
        super().__init__()

        if variant not in self.ARCH_CONFIGS:
            msg = f"Variant {variant} not supported. Choose from {list(self.ARCH_CONFIGS.keys())}"
            raise ValueError(msg)

        config = self.ARCH_CONFIGS[variant]

        stem_channels = self._round_channels(24, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels, eps=1e-3, momentum=0.01),
            nn.SiLU(inplace=True),
        )

        total_blocks = sum(self._round_repeats(cfg[3], depth_mult) for cfg in config)

        self.stages = nn.ModuleList()
        in_channels = stem_channels
        block_idx = 0

        for block_type, expand, channels, layers_count, stride, kernel, se_ratio in config:
            out_channels = self._round_channels(channels, width_mult)
            actual_layers = self._round_repeats(layers_count, depth_mult)

            stage_blocks = []
            for i in range(actual_layers):
                s = stride if i == 0 else 1
                drop_rate = drop_path_rate * block_idx / total_blocks

                if block_type == "fused":
                    stage_blocks.append(
                        FusedMBConv(
                            in_channels, out_channels, expand, s,
                            kernel_size=kernel, drop_path_rate=drop_rate
                        )
                    )
                else:
                    stage_blocks.append(
                        MBConv(
                            in_channels, out_channels, expand, s,
                            kernel_size=kernel, se_ratio=se_ratio, drop_path_rate=drop_rate
                        )
                    )

                in_channels = out_channels
                block_idx += 1

            self.stages.append(nn.Sequential(*stage_blocks))

        head_channels = self._round_channels(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels, eps=1e-3, momentum=0.01),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(head_channels, num_classes),
        )

        self._initialize_weights()

    @staticmethod
    def _round_channels(channels: int, width_mult: float, divisor: int = 8) -> int:
        """Round channels to nearest divisor.

        Args:
            channels: Original channel count.
            width_mult: Width multiplier.
            divisor: Divisor for hardware alignment.

        Returns:
            The rounded integer channel count.

        """
        channels_scaled = channels * width_mult
        new_channels = max(divisor, int(channels_scaled + divisor / 2) // divisor * divisor)
        if new_channels < 0.9 * channels_scaled:
            new_channels += divisor
        return int(new_channels)

    @staticmethod
    def _round_repeats(repeats: int, depth_mult: float) -> int:
        """Round number of repeats based on depth multiplier.

        Args:
            repeats: Original repeat count.
            depth_mult: Depth multiplier.

        Returns:
            The rounded integer repeat count.

        """
        return math.ceil(depth_mult * repeats)

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass.

        Args:
            x: Input image tensor.

        Returns:
            Prediction logits.

        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def freeze_stages(self, num_stages: int) -> None:
        """Freeze specific stages of the model.

        Args:
            num_stages: Number of stages to freeze.

        """
        if num_stages > 0:
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(min(num_stages, len(self.stages))):
            for param in self.stages[i].parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = True


def create_efficientnetv2(
    variant: str = "b3",
    num_classes: int = 1081,
    **kwargs: object
) -> EfficientNetV2:
    """Create an EfficientNetV2 model instance.

    Args:
        variant: Model variant string (e.g., 'b0').
        num_classes: Output class count.
        **kwargs: Additional hyperparameters.

    Returns:
        The instantiated EfficientNetV2 model.

    """
    return EfficientNetV2(variant=variant, num_classes=num_classes, **kwargs)
