"""Pytest fixtures for unit tests.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

import pytest
import torch

from plant_care_ai.models.efficientnetv2 import EfficientNetV2
from plant_care_ai.models.resnet18 import Resnet18

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class DummyTransform:
    """No-op transform tracking whether it was called."""

    def __init__(self) -> None:
        """Initialize called flag."""
        self.called = False

    def __call__(self, image: object) -> object:
        """Apply the transform.

        Args:
            image: input image

        Returns:
            object: unchanged image

        """
        self.called = True
        return image


@pytest.fixture
def sample_data_dir() -> Path:
    """Provide the artifacts tree with images/<split>/<class>/<file>.jpg.

    Returns:
        Path: Path to the sample data directory

    """
    return ARTIFACTS_DIR


@pytest.fixture
def dummy_transform() -> DummyTransform:
    """Return a fresh DummyTransform instance.

    Returns:
        DummyTransform: no-op transform

    """
    return DummyTransform()


@pytest.fixture
def sample_image_path() -> Path:
    """Return a deterministic sample image path from artifacts/train.

    Returns:
        Path: Path to a sample image in the training set

    Raises:
        FileNotFoundError: if no sample images are found

    """
    image_paths = sorted((ARTIFACTS_DIR / "images" / "train").rglob("*.jpg"))
    if not image_paths:
        msg = "No sample images found in artifacts/images/train"
        raise FileNotFoundError(msg)
    return image_paths[0]


@pytest.fixture
def model_b0() -> EfficientNetV2:
    """Create a small EfficientNetV2-B0 for testing.

    Returns:
        EfficientNetV2 model instance with 10 classes

    """
    return EfficientNetV2(variant="b0", num_classes=10)


@pytest.fixture
def resnet18_model() -> Resnet18:
    """Create a Resnet18 instance with 10 classes.

    Returns:
        Resnet18 model instance

    """
    return Resnet18(num_classes=10)


@pytest.fixture
def saved_resnet_weights(tmp_path: Path) -> Path:
    """Create a temporary file with ResNet18 weights.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to saved weights file

    """
    model = Resnet18(num_classes=10)
    weights_path = tmp_path / "resnet18.pth"
    torch.save(model.state_dict(), weights_path)
    return weights_path


@pytest.fixture
def saved_resnet_weights_wrapped(tmp_path: Path) -> Path:
    """Create weights with state_dict key (checkpoint format).

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to saved checkpoint file

    """
    model = Resnet18(num_classes=10)
    weights_path = tmp_path / "resnet18_checkpoint.pth"
    checkpoint = {
        "state_dict": model.state_dict(),
        "epoch": 100,
        "optimizer": {},
    }
    torch.save(checkpoint, weights_path)
    return weights_path
