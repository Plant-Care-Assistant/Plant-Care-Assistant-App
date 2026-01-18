"""Training module __init__ tests.

Copyright 2025 Plant Care Assistant
"""

from plant_care_ai.training import PlantTrainer, get_most_popular_classes


def test_plant_trainer_is_exported() -> None:
    """Test that PlantTrainer is exported from training module."""
    assert PlantTrainer is not None


def test_get_most_popular_classes_is_exported() -> None:
    """Test that get_most_popular_classes is exported from training module."""
    assert get_most_popular_classes is not None


def test_plant_trainer_has_required_attributes() -> None:
    """Test that PlantTrainer class has expected attributes."""
    assert hasattr(PlantTrainer, "REQUIRED_CONFIG_KEYS")
    assert hasattr(PlantTrainer, "prepare_data")
    assert hasattr(PlantTrainer, "build_model")
    assert hasattr(PlantTrainer, "setup_training")
    assert hasattr(PlantTrainer, "train")
    assert hasattr(PlantTrainer, "validate")
    assert hasattr(PlantTrainer, "save_checkpoint")


def test_get_most_popular_classes_is_callable() -> None:
    """Test that get_most_popular_classes is callable."""
    assert callable(get_most_popular_classes)
