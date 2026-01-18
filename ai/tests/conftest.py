"""Pytest fixtures for unit tests.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch

from plant_care_ai.models.efficientnetv2 import EfficientNetV2
from plant_care_ai.models.resnet18 import Resnet18

if TYPE_CHECKING:
    from collections.abc import Generator

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


# ===== Integration Test Fixtures =====


@pytest.fixture
def test_image_path() -> Path:
    """Get path to test image for API tests.

    Returns:
        Path to a test image file.

    """
    image_path = ARTIFACTS_DIR / "images" / "test" / "1355868"
    image_files = list(image_path.glob("*.jpg"))
    if image_files:
        return image_files[0]
    # Fallback to any test image
    return next((ARTIFACTS_DIR / "images" / "test").rglob("*.jpg"))


@pytest.fixture
def mock_classifier() -> "Generator[MagicMock, None, None]":
    """Mock the classifier for predict endpoint tests.

    Yields:
        MagicMock classifier that returns fake predictions.

    """
    import plant_care_ai.api.main as main_module  # noqa: PLC0415

    all_predictions = [
        {"class_id": "1363227", "class_name": "Rosa canina", "confidence": 0.87},
        {"class_id": "1392475", "class_name": "Bellis perennis", "confidence": 0.08},
        {"class_id": "1356022", "class_name": "Taraxacum officinale", "confidence": 0.03},
        {"class_id": "1364099", "class_name": "Trifolium repens", "confidence": 0.01},
        {"class_id": "1355937", "class_name": "Plantago major", "confidence": 0.005},
    ]

    def mock_predict(_image: object, top_k: int = 5) -> dict:
        """Return fake predictions respecting top_k parameter.

        Args:
            _image: Input image (unused in mock).
            top_k: Number of predictions to return.

        Returns:
            Dictionary with predictions and processing time.

        """
        return {
            "predictions": all_predictions[:top_k],
            "processing_time_ms": 52.3,
        }

    mock = MagicMock()
    mock.num_classes = 100
    mock.predict.side_effect = mock_predict

    # Store original and set mock
    original_classifier = main_module.classifier
    main_module.classifier = mock

    yield mock

    # Restore original
    main_module.classifier = original_classifier
