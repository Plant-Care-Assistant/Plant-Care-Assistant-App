"""Module for methods to initialize and load model architectures.

Copyright (c) 2026 Plant Care Assistant. All rights reserved.
"""

import pathlib

import torch

from .efficientnetv2 import EfficientNetV2
from .resnet18 import Resnet18
from .resnet50 import Resnet50


def get_model(
    model_name: str,
    num_classes: int = 1081,
    weights_path: str | None = None,
    device: str = "cpu",
    **kwargs: object,
) -> torch.nn.Module:
    """Initialize models and (optionally) load weights.

    Args:
        model_name: 'resnet18', 'resnet50', or 'efficientnetv2' (case insensitive)
        num_classes: number of output classes (default 1081)
        weights_path: path to .pth file to load state_dict
        device: 'cpu' or 'cuda' to move the model to
        **kwargs: additional arguments for specific models (e.g., variant='b3', pretrained=True)

    Returns:
        The initialized and configured torch.nn.Module

    Raises:
        ValueError: If model_name is not supported

    """
    model_name = model_name.lower().strip()

    pretrained = kwargs.pop("pretrained", True)

    if model_name == "resnet18":
        model = Resnet18(num_classes=num_classes)

    elif model_name == "resnet50":
        model = Resnet50(num_classes=num_classes, pretrained=pretrained)

    elif "efficientnet" in model_name:
        variant = kwargs.pop("variant", "b3")
        model = EfficientNetV2(variant=variant, num_classes=num_classes, **kwargs)

    else:
        msg = (
            f"Model '{model_name}' not supported. "
            "Choose 'resnet18', 'resnet50', or 'efficientnetv2'."
        )
        raise ValueError(msg)

    if weights_path:
        path = pathlib.Path(weights_path)
        if path.exists():
            checkpoint = torch.load(weights_path, map_location=device)

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Weights file not found at {weights_path}.")

    return model.to(device)
