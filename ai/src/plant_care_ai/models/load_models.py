"""Module for methods to initialize and load model architectures.

Copyright (c) 2026 Plant Care Assistant. All rights reserved.
"""

import pathlib
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
import timm

from .disease_model import DiseasePlantModel


def get_model(
    model_name: str,
    num_diseases: int = 38,
    num_classes: int | None = None,
    weights_path: str | None = None,
    device: str = "cpu",
    combine: bool = True,
    **kwargs,
) -> torch.nn.Module:
    model_name = model_name.lower().strip()
    target_classes = num_classes if num_classes is not None else num_diseases
    
    pretrained = kwargs.pop("pretrained", True)

    if combine:
        model = DiseasePlantModel(
            model_name=model_name,
            num_diseases=num_diseases,
            pretrained=pretrained,
            **kwargs,
        )
    else:
        if model_name == "resnet50":
            w = ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=w)
            if target_classes != 1000:
                model.fc = torch.nn.Linear(model.fc.in_features, target_classes)
                
        elif "efficientnet" in model_name:
            model = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=target_classes, 
                **kwargs
            )
        else:
            raise ValueError(f"Model '{model_name}' not supported.")

    if weights_path:
        path = pathlib.Path(weights_path)
        if path.exists():
            checkpoint = torch.load(weights_path, map_location=device)
            state_dict = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)

    return model.to(device)