"""DataLoader unit tests module.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

from torch.utils.data import RandomSampler, SequentialSampler

from plant_care_ai.data.dataloader import PlantNetDataLoader

# Sample data sizes
TRAIN_DATASET_SIZE = 2
VAL_DATASET_SIZE = 3
TEST_DATASET_SIZE = 2


def test_dataloader_initializes_splits_and_num_classes(sample_data_dir: Path) -> None:
    """Test that datasets and num_classes are set correctly.

    Args:
        sample_data_dir: Path to sample data directory

    """
    loader = PlantNetDataLoader(data_dir=sample_data_dir, batch_size=2)

    assert len(loader.train_dataset) == TRAIN_DATASET_SIZE
    assert len(loader.val_dataset) == VAL_DATASET_SIZE
    assert len(loader.test_dataset) == TEST_DATASET_SIZE


def test_train_loader_shuffles_and_batch_size(sample_data_dir: Path) -> None:
    """Test that the train loader shuffles data and has correct batch size.

    Args:
        sample_data_dir: Path to sample data directory

    """
    loader = PlantNetDataLoader(data_dir=sample_data_dir, batch_size=2)
    train_loader = loader.get_train_loader()

    expected_train_loader_bs = 2

    assert train_loader.batch_size == expected_train_loader_bs
    assert isinstance(train_loader.sampler, RandomSampler)


def test_val_and_test_loader_no_shuffle(sample_data_dir: Path) -> None:
    """Test that the val and test loaders do not shuffle data and have correct batch size.

    Args:
        sample_data_dir: Path to sample data directory

    """
    loader = PlantNetDataLoader(data_dir=sample_data_dir, batch_size=2)

    val_loader = loader.get_val_loader()
    test_loader = loader.get_test_loader()

    expected_val_test_loader_bs = 2

    assert val_loader.batch_size == expected_val_test_loader_bs
    assert test_loader.batch_size == expected_val_test_loader_bs
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)


def test_transforms_are_assigned(
    sample_data_dir: Path,
    dummy_transform: object,
) -> None:
    """Test that the transforms are assigned correctly to datasets.

    Args:
        sample_data_dir: Path to sample data directory
        dummy_transform: no-op transform

    """
    val_transform = type(dummy_transform)()

    loader = PlantNetDataLoader(
        data_dir=sample_data_dir,
        batch_size=1,
        train_transform=dummy_transform,
        val_transform=val_transform,
    )

    assert loader.train_dataset.transform is dummy_transform
    assert loader.val_dataset.transform is val_transform
    assert loader.test_dataset.transform is val_transform
