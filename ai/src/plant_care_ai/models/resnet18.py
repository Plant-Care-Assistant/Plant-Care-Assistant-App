"""ResNet-18 implementation for PlantNet-300K.

Based on https://arxiv.org/abs/1512.03385

Copyright 2025 Plant Care Assistant
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class Resnet18(nn.Module):
    """ResNet-18 architecture implementation."""

    def __init__(self, num_classes: int = 1081) -> None:
        """Initialize the ResNet-18 model.

        Args:
            num_classes: The number of output classes for the final fully connected layer

        """
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a sequential layer of BasicBlocks.

        Args:
            out_channels: number of output filters for the blocks in this layer
            num_blocks: number of BasicBlocks to stack
            stride: stride for the first block

        Returns:
            A torch.nn.Sequential container of residual blocks

        """
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        layers.extend(
            BasicBlock(out_channels, out_channels, stride=1)
            for _ in range(1, num_blocks)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize weights for convolutions, batch norm, and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the model.

        Args:
            x: input image tensor of shape (Batch, Channels, Height, Width)

        Returns:
            Logits tensor of shape (Batch, num_classes)

        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except for the final fully connected layer."""
        for name, param in self.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers in the model for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True


class BasicBlock(nn.Module):
    """Residual block implementation."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the BasicBlock.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            stride: stride for the first convolution in the block.

        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the resisudual block.

        Args:
            x: Input tensor from the previous layer

        Returns:
            Tensor with residual connection and ReLU applied

        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
