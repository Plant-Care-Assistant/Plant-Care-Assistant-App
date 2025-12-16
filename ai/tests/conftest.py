"""Pytest fixtures for unit tests.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

import pytest

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
