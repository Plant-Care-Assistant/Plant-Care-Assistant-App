"""Image preprocessing pipeline for plant identification.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


class PlantNetPreprocessor:
    """Preprocessor - with optional augmentation - pipeline."""

    # constants for augmentation strength thresholds
    ROTATION_THRESHOLD = 0.3
    COLOR_JITTER_THRESHOLD = 0.5
    AFFINE_THRESHOLD = 0.7

    def __init__(
        self, img_size: int = 224, *, normalize: bool = True, augm_strength: float = 0.0
    ) -> None:
        """Initialize preprocessor.

        Args:
            img_size: target image size (sqr)
            normalize: whether to apply ImageNet norm
            augm_strength: augmentation intensity [0.0 to 1.0]

        """
        self.img_size = img_size
        self.normalize = normalize
        self.augm_strength = max(0.0, min(1.0, augm_strength))  # [0;1]

        # normalization stats of ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def get_full_transform(self) -> transforms.Compose:
        """Get training pipeline with augmentation.

        Returns:
            Composed transform pipeline for training

        """
        # resize, do a random crop
        transforms_list = [
            transforms.Resize(256),
            transforms.RandomCrop(self.img_size),
        ]

        # apply augmentation
        if self.augm_strength > 0:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

            # random rotations (up to 30 deg)
            if self.augm_strength >= self.ROTATION_THRESHOLD:
                rotation_deg = int(30 * self.augm_strength)
                transforms_list.append(transforms.RandomRotation(rotation_deg))

            # random brightness, contrast etc
            if self.augm_strength >= self.COLOR_JITTER_THRESHOLD:
                transforms_list.append(
                    transforms.ColorJitter(
                        brightness=0.3 * self.augm_strength,
                        contrast=0.3 * self.augm_strength,
                        saturation=0.3 * self.augm_strength,
                        hue=0.1 * self.augm_strength,
                    )
                )

            # random translation, scaling
            if self.augm_strength >= self.AFFINE_THRESHOLD:
                transforms_list.append(
                    transforms.RandomAffine(
                        degrees=0,  # done before
                        translate=(0.1 * self.augm_strength, 0.1 * self.augm_strength),
                        scale=(1 - 0.1 * self.augm_strength, 1 + 0.1 * self.augm_strength),
                    )
                )

        transforms_list.append(transforms.ToTensor())

        if self.normalize:
            transforms_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        return transforms.Compose(transforms_list)

    def get_interference_transform(self) -> transforms.Compose:
        """Get validation/inference pipeline without augmentation.

        Returns:
            Composed transform pipeline for validation/inference

        """
        transforms_list = [
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ]

        if self.normalize:
            transforms_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        return transforms.Compose(transforms_list)

    def get_transform(self, *, train: bool = False) -> transforms.Compose:
        """Get transform based on mode.

        Args:
            train: If True, return training transform (with augmentation),
                   otherwise return validation transform (no augmentation)

        Returns:
            Composed transform pipeline

        """
        return self.get_full_transform() if train else self.get_interference_transform()


def get_training_pipeline(img_size: int = 224, augm_strength: float = 0.5) -> transforms.Compose:
    """Get training pipeline with augmentation.

    Args:
        img_size: Target image size
        augm_strength: Augmentation strength (0.0-1.0)

    Returns:
        transforms.Compose: Training transform pipeline with augmentation

    """
    preprocessor = PlantNetPreprocessor(
        img_size=img_size, normalize=True, augm_strength=augm_strength
    )
    return preprocessor.get_full_transform()


def get_inference_pipeline(img_size: int = 224) -> transforms.Compose:
    """Get inference pipeline without augmentation.

    Args:
        img_size: Target image size

    Returns:
        transforms.Compose: Inference transform pipeline

    """
    preprocessor = PlantNetPreprocessor(img_size=img_size, normalize=True, augm_strength=0.0)
    return preprocessor.get_interference_transform()


def preprocess_single_image(image: str | Path, img_size: int = 224) -> torch.Tensor:
    """Preprocess single image for inference.

    Args:
        image: PIL Image or path to image file
        img_size: Target size

    Returns:
        torch.Tensor: Tensor with shape (1, C, H, W) ready for model input

    """
    image = Image.open(image).convert("RGB")

    transform = get_inference_pipeline(img_size)
    tensor = transform(image)

    return tensor.unsqueeze(0)
