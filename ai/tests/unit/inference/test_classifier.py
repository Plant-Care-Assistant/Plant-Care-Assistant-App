"""Classifier unit tests module.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from plant_care_ai.inference.classifier import PlantClassifier
from plant_care_ai.models.efficientnetv2 import create_efficientnetv2
from plant_care_ai.models.resnet18 import Resnet18
from plant_care_ai.models.resnet50 import Resnet50

# Test constants
DEFAULT_NUM_CLASSES = 10
DEFAULT_IMG_SIZE = 224
TOP_K_3 = 3
TOP_K_2 = 2


class TestPlantClassifierInit:
    """Tests for PlantClassifier initialization."""

    @staticmethod
    def test_classifier_init_with_model() -> None:
        """Test that classifier initializes correctly with model."""
        model = Resnet18(num_classes=DEFAULT_NUM_CLASSES)
        idx_to_class = {i: str(1000 + i) for i in range(DEFAULT_NUM_CLASSES)}

        classifier = PlantClassifier(
            model=model,
            idx_to_class=idx_to_class,
            img_size=DEFAULT_IMG_SIZE,
            device="cpu",
        )

        assert classifier.num_classes == DEFAULT_NUM_CLASSES
        assert classifier.img_size == DEFAULT_IMG_SIZE
        assert classifier.device == "cpu"
        assert classifier.id_to_name is None

    @staticmethod
    def test_classifier_init_default_device() -> None:
        """Test that classifier uses CUDA if available, otherwise CPU."""
        model = Resnet18(num_classes=5)
        idx_to_class = {i: str(i) for i in range(5)}

        classifier = PlantClassifier(
            model=model,
            idx_to_class=idx_to_class,
        )

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert classifier.device == expected_device

    @staticmethod
    def test_classifier_model_in_eval_mode() -> None:
        """Test that classifier puts model in eval mode."""
        model = Resnet18(num_classes=5)
        model.train()  # Ensure model is in train mode initially
        idx_to_class = {i: str(i) for i in range(5)}

        classifier = PlantClassifier(
            model=model,
            idx_to_class=idx_to_class,
            device="cpu",
        )

        assert not classifier.model.training


class TestPlantClassifierFromCheckpoint:
    """Tests for PlantClassifier.from_checkpoint class method."""

    @staticmethod
    @pytest.fixture
    def resnet_checkpoint(tmp_path: Path) -> Path:
        """Create a resnet18 checkpoint file.

        Args:
            tmp_path: Pytest temporary directory fixture

        Returns:
            Path to checkpoint file

        """
        model = Resnet18(num_classes=DEFAULT_NUM_CLASSES)
        checkpoint = {
            "config": {"model": "resnet18", "img_size": DEFAULT_IMG_SIZE},
            "num_classes": DEFAULT_NUM_CLASSES,
            "model_state_dict": model.state_dict(),
            "idx_to_class": {i: str(1000 + i) for i in range(DEFAULT_NUM_CLASSES)},
            "best_acc": 95.5,
        }
        checkpoint_path = tmp_path / "resnet_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @staticmethod
    @pytest.fixture
    def efficientnet_checkpoint(tmp_path: Path) -> Path:
        """Create an efficientnetv2 checkpoint file.

        Args:
            tmp_path: Pytest temporary directory fixture

        Returns:
            Path to checkpoint file

        """
        model = create_efficientnetv2(variant="b0", num_classes=DEFAULT_NUM_CLASSES)
        checkpoint = {
            "config": {"model": "efficientnetv2", "variant": "b0", "img_size": DEFAULT_IMG_SIZE},
            "num_classes": DEFAULT_NUM_CLASSES,
            "model_state_dict": model.state_dict(),
            "idx_to_class": {i: str(1000 + i) for i in range(DEFAULT_NUM_CLASSES)},
        }
        checkpoint_path = tmp_path / "efficientnet_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @staticmethod
    @pytest.fixture
    def resnet50_checkpoint(tmp_path: Path) -> Path:
        """Create a resnet50 checkpoint file.

        Args:
            tmp_path: Pytest temporary directory fixture

        Returns:
            Path to checkpoint file

        """
        model = Resnet50(num_classes=DEFAULT_NUM_CLASSES, pretrained=False)
        checkpoint = {
            "config": {"model": "resnet50", "img_size": DEFAULT_IMG_SIZE},
            "num_classes": DEFAULT_NUM_CLASSES,
            "model_state_dict": model.state_dict(),
            "idx_to_class": {i: str(1000 + i) for i in range(DEFAULT_NUM_CLASSES)},
        }
        checkpoint_path = tmp_path / "resnet50_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @staticmethod
    def test_from_checkpoint_resnet(resnet_checkpoint: Path) -> None:
        """Test loading classifier from resnet18 checkpoint.

        Args:
            resnet_checkpoint: Path to resnet checkpoint

        """
        classifier = PlantClassifier.from_checkpoint(
            checkpoint_path=resnet_checkpoint,
            device="cpu",
            verbose=False,
        )

        assert classifier.num_classes == DEFAULT_NUM_CLASSES
        assert classifier.img_size == DEFAULT_IMG_SIZE
        assert isinstance(classifier.model, Resnet18)

    @staticmethod
    def test_from_checkpoint_resnet50(resnet50_checkpoint: Path) -> None:
        """Test loading classifier from resnet50 checkpoint.

        Args:
            resnet50_checkpoint: Path to resnet50 checkpoint

        """
        classifier = PlantClassifier.from_checkpoint(
            checkpoint_path=resnet50_checkpoint,
            device="cpu",
            verbose=False,
        )

        assert classifier.num_classes == DEFAULT_NUM_CLASSES
        assert classifier.img_size == DEFAULT_IMG_SIZE
        assert isinstance(classifier.model, Resnet50)

    @staticmethod
    def test_from_checkpoint_efficientnet(efficientnet_checkpoint: Path) -> None:
        """Test loading classifier from efficientnetv2 checkpoint.

        Args:
            efficientnet_checkpoint: Path to efficientnet checkpoint

        """
        classifier = PlantClassifier.from_checkpoint(
            checkpoint_path=efficientnet_checkpoint,
            device="cpu",
            verbose=False,
        )

        assert classifier.num_classes == DEFAULT_NUM_CLASSES
        assert classifier.img_size == DEFAULT_IMG_SIZE

    @staticmethod
    def test_from_checkpoint_unknown_model_raises(tmp_path: Path) -> None:
        """Test that from_checkpoint raises ValueError for unknown model type.

        Args:
            tmp_path: Pytest temporary directory fixture

        """
        checkpoint = {
            "config": {"model": "unknown_model", "img_size": 224},
            "num_classes": 10,
            "model_state_dict": {},
            "idx_to_class": {0: "class_0"},
        }
        checkpoint_path = tmp_path / "unknown_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(ValueError, match="Unknown model type"):
            PlantClassifier.from_checkpoint(checkpoint_path, device="cpu")

    @staticmethod
    def test_from_checkpoint_verbose_output(
        resnet_checkpoint: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that from_checkpoint prints info when verbose=True.

        Args:
            resnet_checkpoint: Path to resnet checkpoint
            capsys: Pytest capture fixture

        """
        PlantClassifier.from_checkpoint(
            checkpoint_path=resnet_checkpoint,
            device="cpu",
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Loaded checkpoint" in captured.out
        assert "Model: resnet18" in captured.out
        assert "Classes: 10" in captured.out
        assert "Best validation accuracy: 95.50%" in captured.out


class TestSetNameMapping:
    """Tests for PlantClassifier.set_name_mapping method."""

    @staticmethod
    def test_set_name_mapping() -> None:
        """Test that set_name_mapping sets the id_to_name attribute."""
        model = Resnet18(num_classes=3)
        idx_to_class = {0: "1000", 1: "1001", 2: "1002"}
        classifier = PlantClassifier(model=model, idx_to_class=idx_to_class, device="cpu")

        name_mapping = {
            "1000": "Rosa canina",
            "1001": "Bellis perennis",
            "1002": "Taraxacum officinale",
        }
        classifier.set_name_mapping(name_mapping)

        assert classifier.id_to_name == name_mapping


class TestPredict:
    """Tests for PlantClassifier.predict method."""

    @staticmethod
    @pytest.fixture
    def classifier_with_model() -> PlantClassifier:
        """Create a classifier with a small model for testing.

        Returns:
            PlantClassifier instance

        """
        model = Resnet18(num_classes=5)
        idx_to_class = {
            0: "1355936",
            1: "1355932",
            2: "1355868",
            3: "1363227",
            4: "1392475",
        }

        return PlantClassifier(
            model=model,
            idx_to_class=idx_to_class,
            img_size=224,
            device="cpu",
        )

    @staticmethod
    def test_predict_with_pil_image(classifier_with_model: PlantClassifier) -> None:
        """Test prediction with PIL Image input.

        Args:
            classifier_with_model: PlantClassifier fixture

        """
        image = Image.new("RGB", (256, 256), color="green")
        result = classifier_with_model.predict(image, top_k=TOP_K_3)

        assert "predictions" in result
        assert "processing_time_ms" in result
        assert len(result["predictions"]) == TOP_K_3

        for pred in result["predictions"]:
            assert "class_id" in pred
            assert "confidence" in pred
            assert 0.0 <= pred["confidence"] <= 1.0

    @staticmethod
    def test_predict_with_image_path(
        classifier_with_model: PlantClassifier,
        sample_image_path: Path,
    ) -> None:
        """Test prediction with image path input.

        Args:
            classifier_with_model: PlantClassifier fixture
            sample_image_path: Path to sample image

        """
        result = classifier_with_model.predict(sample_image_path, top_k=TOP_K_2)

        assert "predictions" in result
        assert len(result["predictions"]) == TOP_K_2

    @staticmethod
    def test_predict_with_string_path(
        classifier_with_model: PlantClassifier,
        sample_image_path: Path,
    ) -> None:
        """Test prediction with string path input.

        Args:
            classifier_with_model: PlantClassifier fixture
            sample_image_path: Path to sample image

        """
        result = classifier_with_model.predict(str(sample_image_path), top_k=1)

        assert "predictions" in result
        assert len(result["predictions"]) == 1

    @staticmethod
    def test_predict_with_name_mapping(classifier_with_model: PlantClassifier) -> None:
        """Test that predict includes class names when mapping is set.

        Args:
            classifier_with_model: PlantClassifier fixture

        """
        name_mapping = {
            "1355936": "Rosa canina",
            "1355932": "Bellis perennis",
            "1355868": "Taraxacum officinale",
            "1363227": "Trifolium repens",
            "1392475": "Plantago major",
        }
        classifier_with_model.set_name_mapping(name_mapping)

        image = Image.new("RGB", (256, 256), color="red")
        result = classifier_with_model.predict(image, top_k=5)

        # At least one prediction should have a class_name
        has_class_name = any("class_name" in pred for pred in result["predictions"])
        assert has_class_name

    @staticmethod
    def test_predict_top_k_exceeds_num_classes(classifier_with_model: PlantClassifier) -> None:
        """Test that predict handles top_k greater than num_classes.

        Args:
            classifier_with_model: PlantClassifier fixture

        """
        image = Image.new("RGB", (256, 256), color="blue")
        result = classifier_with_model.predict(image, top_k=100)

        # Should return at most num_classes predictions
        assert len(result["predictions"]) == classifier_with_model.num_classes

    @staticmethod
    def test_predict_processing_time_positive(classifier_with_model: PlantClassifier) -> None:
        """Test that processing time is positive.

        Args:
            classifier_with_model: PlantClassifier fixture

        """
        image = Image.new("RGB", (256, 256), color="yellow")
        result = classifier_with_model.predict(image, top_k=1)

        assert result["processing_time_ms"] > 0

    @staticmethod
    def test_predict_confidences_sum_reasonable(classifier_with_model: PlantClassifier) -> None:
        """Test that prediction confidences are reasonable.

        Args:
            classifier_with_model: PlantClassifier fixture

        """
        image = Image.new("RGB", (256, 256), color="purple")
        result = classifier_with_model.predict(image, top_k=5)

        confidences = [pred["confidence"] for pred in result["predictions"]]

        # All confidences should be between 0 and 1
        assert all(0.0 <= c <= 1.0 for c in confidences)

        # Sum of all confidences should not exceed 1 (they're probabilities)
        assert sum(confidences) <= 1.0 + 1e-6  # Small epsilon for floating point

    @staticmethod
    def test_predict_unknown_class_index_raises() -> None:
        """Test that predict raises when output index is missing from mapping."""

        class DummyModel(torch.nn.Module):
            """Return logits where the top class index is not in idx_to_class."""

            @staticmethod
            def forward(x: torch.Tensor) -> torch.Tensor:
                return x.new_tensor([[0.0, 1.0]])

        classifier = PlantClassifier(
            model=DummyModel(),
            idx_to_class={0: "class_0"},
            img_size=DEFAULT_IMG_SIZE,
            device="cpu",
        )

        image = Image.new("RGB", (256, 256), color="white")

        with pytest.raises(KeyError, match="not found in idx_to_class mapping"):
            classifier.predict(image, top_k=1)
