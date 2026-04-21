"""Module for methods to initialize and load model architectures.

Copyright (c) 2026 Plant Care Assistant. All rights reserved.
"""

import pathlib
import torch
from torchinfo import summary

from .efficientnetv2 import EfficientNetV2
from .resnet50 import Resnet50
from src.plant_care_ai.models.multitask_model import MultiTaskPlantModel


def get_model(
    model_name: str,
    num_species: int = 1087,
    num_diseases: int = 38,
    num_classes: int | None = None,
    weights_path: str | None = None,
    device: str = "cpu",
    multitask: bool = True,
    **kwargs,
) -> torch.nn.Module:
    """
    Initialize models and (optionally) load weights. Supports both multi-task 
    and single-task architectures.
    
    Args:
        model_name: 'resnet50', or 'efficientnetv2' (case insensitive)
        num_species: number of species output classes (used if multitask=True).
        num_diseases: number of disease output classes (used if multitask=True).
        num_classes: standard output classes (used if multitask=False)
        weights_path: path to .pth file to load state_dict
        device: 'cpu' or 'cuda' to move the model to
        multitask: if True, uses MultiTaskPlantModel, if False, uses base architectures
        **kwargs: additional arguments for specific models (e.g., variant='b3', pretrained=True)

    Returns:
        Configured torch.nn.Module on the specified device.

    Raises:
        ValueError: If model_name is not supported.
    """
    model_name = model_name.lower().strip()
    pretrained = kwargs.pop("pretrained", True)

    if multitask:
        model = MultiTaskPlantModel(
            model_name=model_name,
            num_species=num_species,
            num_diseases=num_diseases,
            **kwargs,
        )
    else:
        target_classes = num_classes if num_classes is not None else num_species

        if model_name == "resnet50":
            model = Resnet50(num_classes=target_classes, pretrained=pretrained)
        elif "efficientnet" in model_name:
            variant = kwargs.pop("variant", "b3")
            model = EfficientNetV2(variant=variant, num_classes=target_classes, **kwargs)
        else:
            msg = f"Model '{model_name}' not supported. Choose 'resnet50' or 'efficientnetv2'."
            raise ValueError(msg)

    if weights_path:
        path = pathlib.Path(weights_path)
        if path.exists():
            checkpoint = torch.load(weights_path, map_location=device)
            
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Weights file not found at {weights_path}.")

    return model.to(device)