import torch
import os

from .effecientnetv2 import EfficientNetV2
from .resnet18 import Resnet18

def get_model(
    model_name: str, 
    num_classes: int = 1081, 
    weights_path: str = None, 
    device: str = "cpu",
    **kwargs
) -> torch.nn.Module:
    """
    Initialize models and (optionally) load weights.

    Args:
        model_name (str): 'resnet18' or 'efficientnetv2' (case insensitive)
        num_classes (int): num of output classes (default 1081)
        weights_path (str, optional): path to .pth file to load state_dict
        device (str): 'cpu' or 'cuda' to move the model to
        **kwargs: Additional arguments for specific models (e.g., variant='b0')

    Returns:
        torch.nn.Module: initialized model
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        model = Resnet18(num_classes=num_classes)
    
    elif "efficientnet" in model_name:
        variant = kwargs.get("variant", "b3")
        model = EfficientNetV2(
            variant=variant, 
            num_classes=num_classes,
            **kwargs
        )
    
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Choose 'resnet18' or 'efficientnetv2'.")

    if weights_path:
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device)
            
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Weights file not found at {weights_path}. Initializing with random weights.")

    # 3. Move to device and return
    model = model.to(device)
    return model