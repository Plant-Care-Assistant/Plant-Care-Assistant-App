"""Image preprocessing pipelines for plant identification.

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
        self,
        img_size: int = 224,
        *,
        normalize: bool = True,
        augm_strength: float = 0.0,
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


    def _augmentation_transforms(self) -> list:
        transforms_list = []

        if self.augm_strength <= 0:
            return transforms_list

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

        return transforms_list

    def _normalisation_transforms(self) -> list:
        t = [transforms.ToTensor()]
        if self.normalize:
            t.append(transforms.Normalize(mean=self.mean, std=self.std))
        return t

    def get_full_transform(self) -> transforms.Compose:
        """Training pipeline: resize -> random crop -> augmentation -> tensor -> norm.

        Returns:
            Composed transform pipeline for training.

        """
        pipeline = [
            transforms.Resize(256),
            transforms.RandomCrop(self.img_size),
            *self._augmentation_transforms(),
            *self._normalisation_transforms(),
        ]
        return transforms.Compose(pipeline)

    def get_inference_transform(self) -> transforms.Compose:
        """Validation / inference pipeline: resize -> centre crop -> tensor -> norm.

        Returns:
            Composed transform pipeline for validation / inference.

        """
        pipeline = [
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            *self._normalisation_transforms(),
        ]
        return transforms.Compose(pipeline)

    def get_transform(self, *, train: bool = False) -> transforms.Compose:
        return self.get_full_transform() if train else self.get_inference_transform()


class PlantVillagePreprocessor(PlantNetPreprocessor):
    """PlantVillage preprocessor - with optional augmentation - pipeline."""

    def __init__(
        self,
        img_size: int = 224,
        *,
        normalize: bool = True,
        augm_strength: float = 0.0,
        add_vertical_flip: bool = True,
    ) -> None:
        super().__init__(img_size=img_size, normalize=normalize, augm_strength=augm_strength)
        self.add_vertical_flip = add_vertical_flip

    def _augmentation_transforms(self) -> list:
        """Extend base augmentations with an optional vertical flip."""
        augs = super()._augmentation_transforms()

        if self.augm_strength > 0 and self.add_vertical_flip:
            # add vertical flip after the horizontal one (index 1 if present, else 0)
            insert_at = 1 if augs and isinstance(augs[0], transforms.RandomHorizontalFlip) else 0
            augs.insert(insert_at, transforms.RandomVerticalFlip(p=0.3))

        return augs

    def get_full_transform(self) -> transforms.Compose:
        pipeline = [
            # RandomResizedCrop avoids the hard resize->RandomCrop boundary
            # artefacts common in tightly-cropped leaf images.
            transforms.RandomResizedCrop(
                self.img_size,
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
            ),
            *self._augmentation_transforms(),
            *self._normalisation_transforms(),
        ]
        return transforms.Compose(pipeline)

    def get_inference_transform(self) -> transforms.Compose:
        pipeline = [
            transforms.Resize(int(self.img_size * 256 / 224)),   # same ratio as in PN
            transforms.CenterCrop(self.img_size),
            *self._normalisation_transforms(),
        ]
        return transforms.Compose(pipeline)


def get_training_pipeline(img_size: int = 224, augm_strength: float = 0.5) -> transforms.Compose:
    """PlantNet training pipeline with augmentation.

    Args:
        img_size: Target image size.
        augm_strength: Augmentation strength [0.0 – 1.0].

    Returns:
        Training transform pipeline.

    """
    return PlantNetPreprocessor(
        img_size=img_size, normalize=True, augm_strength=augm_strength
    ).get_full_transform()


def get_inference_pipeline(img_size: int = 224) -> transforms.Compose:
    """PlantNet inference pipeline without augmentation.

    Args:
        img_size: Target image size.

    Returns:
        Inference transform pipeline.

    """
    return PlantNetPreprocessor(img_size=img_size, normalize=True).get_inference_transform()


def get_plantvillage_training_pipeline(
    img_size: int = 224,
    augm_strength: float = 0.5,
) -> transforms.Compose:
    """PlantVillage training pipeline with augmentation.

    Args:
        img_size: Target image size.
        augm_strength: Augmentation strength [0.0 – 1.0].

    Returns:
        Training transform pipeline.

    """
    return PlantVillagePreprocessor(
        img_size=img_size, normalize=True, augm_strength=augm_strength
    ).get_full_transform()


def get_plantvillage_inference_pipeline(img_size: int = 224) -> transforms.Compose:
    """PlantVillage inference pipeline without augmentation.

    Args:
        img_size: Target image size.

    Returns:
        Inference transform pipeline.

    """
    return PlantVillagePreprocessor(img_size=img_size, normalize=True).get_inference_transform()


def preprocess_single_image(image: str | Path, img_size: int = 224) -> torch.Tensor:
    """Preprocess a single image for inference.

    Args:
        image: PIL Image or path to image file.
        img_size: Target size.

    Returns:
        Tensor with shape (1, C, H, W) ready for model input.

    """
    image = Image.open(image).convert("RGB")
    tensor = get_inference_pipeline(img_size)(image)
    return tensor.unsqueeze(0)