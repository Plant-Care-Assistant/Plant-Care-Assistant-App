"""Preprocessing unit tests module.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

import torch
from torchvision import transforms

from plant_care_ai.data.preprocessing import (
    PlantNetPreprocessor,
    get_inference_pipeline,
    get_training_pipeline,
    preprocess_single_image,
)


def test_get_full_transform_contains_expected_steps() -> None:
    """Test that full transform includes expected augmentation steps."""
    preprocessor = PlantNetPreprocessor(img_size=224, augm_strength=0.8)
    pipeline = preprocessor.get_full_transform()

    expected_types = [
        transforms.Resize,
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.RandomRotation,
        transforms.ColorJitter,
        transforms.RandomAffine,
        transforms.ToTensor,
        transforms.Normalize,
    ]

    assert [type(t) for t in pipeline.transforms] == expected_types


def test_get_inference_transform() -> None:
    """Test that inference transform includes expected steps."""
    img_size = 128
    pipeline = PlantNetPreprocessor(img_size=img_size).get_inference_transform()

    assert [type(t) for t in pipeline.transforms] == [
        transforms.Resize,
        transforms.CenterCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    crop_size = pipeline.transforms[1].size
    assert crop_size in {img_size, (img_size, img_size)}


def test_get_transform_switches_by_mode() -> None:
    """Test that get_transform returns correct pipeline based on mode."""
    preprocessor = PlantNetPreprocessor(img_size=224, augm_strength=0.5)

    train_pipeline = preprocessor.get_transform(train=True)
    val_pipeline = preprocessor.get_transform(train=False)

    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_pipeline.transforms)
    assert any(isinstance(t, transforms.RandomCrop) for t in train_pipeline.transforms)
    assert any(isinstance(t, transforms.CenterCrop) for t in val_pipeline.transforms)
    assert not any(isinstance(t, transforms.RandomCrop) for t in val_pipeline.transforms)


def test_get_training_pipeline_helper_matches_preprocessor() -> None:
    """Test that helper function for training pipeline matches manual preprocessor."""
    helper_pipeline = get_training_pipeline(img_size=224, augm_strength=0.3)
    manual_pipeline = PlantNetPreprocessor(img_size=224, augm_strength=0.3).get_full_transform()

    assert [type(t) for t in helper_pipeline.transforms] == [
        type(t) for t in manual_pipeline.transforms
    ]


def test_get_inference_pipeline_helper_matches_preprocessor() -> None:
    """Test that helper function for inference pipeline matches manual preprocessor."""
    helper_pipeline = get_inference_pipeline(img_size=256)
    manual_pipeline = PlantNetPreprocessor(img_size=256).get_inference_transform()

    assert [type(t) for t in helper_pipeline.transforms] == [
        type(t) for t in manual_pipeline.transforms
    ]


def test_preprocess_single_image_returns_batched_tensor(sample_image_path: Path) -> None:
    """Test that single image preprocessing returns correct tensor."""
    tensor = preprocess_single_image(sample_image_path, img_size=128)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 128, 128)
    assert torch.isfinite(tensor).all()
    assert tensor.abs().max() > 0
