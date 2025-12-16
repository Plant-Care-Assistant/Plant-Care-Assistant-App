"""Dataset unit tests module.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

import pytest
import torch
import torchvision.transforms as T  # noqa: N812

from plant_care_ai.data.dataset import PlantNetDataset


@pytest.mark.parametrize(
    ("split", "expected_count"),
    [
        ("train", 2),
        ("val", 2),
        ("test", 2),
    ],
)
def test_len_counts_images(sample_data_dir: Path, split: str, expected_count: int) -> None:
    """Test that dataset length matches number of images in each split.

    Args:
        sample_data_dir: Path to the sample data directory
        split: Dataset split to test
        expected_count: Expected number of images in the split

    """
    dataset = PlantNetDataset(data_dir=sample_data_dir, split=split)
    imgs_in_train_dir = expected_count
    assert len(dataset) == imgs_in_train_dir


@pytest.mark.parametrize(
    ("split", "expected_classes"),
    [
        ("train", ["1355936"]),
        ("val", ["1355932", "1355868"]),
        ("test", ["1355932", "1355868"]),
    ],
)
def test_classes_and_mapping(
    sample_data_dir: Path, split: str, expected_classes: list[str]
) -> None:
    """Test that dataset classes and class_to_idx mapping are correct.

    Args:
        sample_data_dir: Path to the sample data directory
        split: Dataset split to test
        expected_classes: Expected list of class names (species IDs)

    """
    dataset = PlantNetDataset(data_dir=sample_data_dir, split=split)

    dataset_classes = sorted(dataset.classes)
    expected_classes = sorted(expected_classes)

    assert dataset_classes == expected_classes
    assert dataset.class_to_idx == {cls: idx for idx, cls in enumerate(expected_classes)}


def test_getitem_returns_tensor_and_label(sample_data_dir: Path) -> None:
    """Test that __getitem__ returns an image tensor and correct label index.

    Args:
        sample_data_dir: Path to the sample data directory

    """
    dataset = PlantNetDataset(
        data_dir=sample_data_dir,
        split="train",
        transform=T.ToTensor(),
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)

    rgb_channels = 3

    assert image.shape[0] == rgb_channels
    assert label in dataset.class_to_idx.values()


def test_transform_called(sample_data_dir: Path, dummy_transform: object) -> None:
    """Test that the transform is called during __getitem__.

    Args:
        sample_data_dir: Path to the sample data directory
        dummy_transform: A dummy transform function that records if it was called

    """
    dataset = PlantNetDataset(
        data_dir=sample_data_dir,
        split="train",
        transform=dummy_transform,
    )

    _ = dataset[0]

    assert dummy_transform.called is True


def test_missing_split_dir_raises(tmp_path: Path) -> None:
    """Test that initializing with a missing split directory raises FileNotFoundError.

    Args:
        tmp_path: Temporary directory provided by pytest

    """
    with pytest.raises(FileNotFoundError):
        PlantNetDataset(data_dir=tmp_path, split="train")
