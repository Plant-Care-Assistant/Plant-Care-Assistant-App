"""Module for methods to initialize and load model architectures.

Copyright (c) 2026 Plant Care Assistant. All rights reserved.
"""

import pathlib

import torch

from .disease_model import DiseasePlantModel
from .efficientnetv2 import EfficientNetV2
from .resnet18 import Resnet18
from .resnet50 import Resnet50


def get_model(
    model_name: str,
    num_classes: int = 1081,
    weights_path: str | None = None,
    device: str = "cpu",
    combine: bool = False,
    num_diseases: int = 38,
    **kwargs: object,
) -> torch.nn.Module:
    """Initialize and optionally load a plant classification model.

    Args:
        model_name: 'resnet18', 'resnet50', or 'efficientnetv2' (case-insensitive).
                    When combine=True, also accepts 'resnet50' for the disease backbone.
        num_classes: Output classes for single-task models (default 1081).
        weights_path: Path to .pth file to load state_dict.
        device: 'cpu' or 'cuda' target device.
        combine: If True, return a DiseasePlantModel with disease + health heads.
        num_diseases: Disease output classes used when combine=True (default 38).
        **kwargs: Extra arguments forwarded to the model constructor
                  (e.g. variant='b3', pretrained=True).

    Returns:
        The initialized and configured torch.nn.Module.

    Raises:
        ValueError: If model_name is not supported.

    """
    model_name = model_name.lower().strip()
    pretrained = kwargs.pop("pretrained", True)

    if combine:
        model: torch.nn.Module = DiseasePlantModel(
            model_name=model_name,
            num_diseases=num_diseases,
            pretrained=pretrained,
            **kwargs,
        )
    elif model_name == "resnet18":
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
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Weights file not found at {weights_path}.")

    return model.to(device)
