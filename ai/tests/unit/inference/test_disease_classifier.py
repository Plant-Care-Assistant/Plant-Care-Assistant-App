"""DiseasePlantClassifier unit tests.

Copyright 2026 Plant Care Assistant
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from plant_care_ai.inference.disease_classifier import (
    DEFAULT_HEALTH_THRESHOLD,
    DiseasePlantClassifier,
)
from plant_care_ai.models.disease_model import DiseasePlantModel

NUM_DISEASES = 5
IMG_SIZE = 64


# ===== Fixtures =====


@pytest.fixture
def idx_to_disease() -> dict[int, str]:
    """Return a small disease index mapping.

    Returns:
        dict[int, str]: Mapping from class index to disease name.

    """
    return {
        0: "Apple___Apple_scab",
        1: "Apple___Black_rot",
        2: "Apple___Cedar_apple_rust",
        3: "Apple___healthy",
        4: "Tomato___Early_blight",
    }


@pytest.fixture
def disease_model() -> DiseasePlantModel:
    """Return a tiny DiseasePlantModel for CPU testing.

    Returns:
        DiseasePlantModel with 5 disease classes.

    """
    return DiseasePlantModel(
        model_name="efficientnetv2",
        num_diseases=NUM_DISEASES,
        variant="b0",
        pretrained=False,
    )


@pytest.fixture
def mock_yolo() -> MagicMock:
    """Return a YOLO mock that reports no leaf detections.

    Returns:
        MagicMock behaving like a YOLO model with empty predictions.

    """
    result = MagicMock()
    result.boxes = None
    mock = MagicMock()
    mock.predict.return_value = [result]
    return mock


@pytest.fixture
def classifier(
    disease_model: DiseasePlantModel,
    mock_yolo: MagicMock,
    idx_to_disease: dict[int, str],
) -> DiseasePlantClassifier:
    """Return an initialized DiseasePlantClassifier with mocked YOLO.

    Args:
        disease_model: Tiny DiseasePlantModel fixture.
        mock_yolo: YOLO mock fixture.
        idx_to_disease: Disease index mapping fixture.

    Returns:
        Ready-to-use DiseasePlantClassifier.

    """
    return DiseasePlantClassifier(
        disease_model=disease_model,
        yolo_model=mock_yolo,
        idx_to_disease=idx_to_disease,
        img_size=IMG_SIZE,
        device="cpu",
    )


# ===== Tests =====


class TestInit:
    """Tests for DiseasePlantClassifier.__init__."""

    @staticmethod
    def test_device_assigned(
        disease_model: DiseasePlantModel,
        mock_yolo: MagicMock,
        idx_to_disease: dict[int, str],
    ) -> None:
        """Test that device is correctly assigned.

        Args:
            disease_model: DiseasePlantModel fixture.
            mock_yolo: YOLO mock fixture.
            idx_to_disease: Disease index mapping fixture.

        """
        clf = DiseasePlantClassifier(
            disease_model=disease_model,
            yolo_model=mock_yolo,
            idx_to_disease=idx_to_disease,
            device="cpu",
        )
        assert clf.device == "cpu"

    @staticmethod
    def test_num_diseases(classifier: DiseasePlantClassifier) -> None:
        """Test that num_diseases is derived from idx_to_disease.

        Args:
            classifier: DiseasePlantClassifier fixture.

        """
        assert classifier.num_diseases == NUM_DISEASES

    @staticmethod
    def test_default_health_threshold(classifier: DiseasePlantClassifier) -> None:
        """Test that health_threshold defaults to _DEFAULT_HEALTH_THRESHOLD.

        Args:
            classifier: DiseasePlantClassifier fixture.

        """
        assert classifier.health_threshold == DEFAULT_HEALTH_THRESHOLD

    @staticmethod
    def test_custom_health_threshold(
        disease_model: DiseasePlantModel,
        mock_yolo: MagicMock,
        idx_to_disease: dict[int, str],
    ) -> None:
        """Test that a custom health_threshold is stored correctly.

        Args:
            disease_model: DiseasePlantModel fixture.
            mock_yolo: YOLO mock fixture.
            idx_to_disease: Disease index mapping fixture.

        """
        clf = DiseasePlantClassifier(
            disease_model=disease_model,
            yolo_model=mock_yolo,
            idx_to_disease=idx_to_disease,
            health_threshold=0.3,
            device="cpu",
        )
        assert clf.health_threshold == pytest.approx(0.3)

    @staticmethod
    def test_model_in_eval_mode(classifier: DiseasePlantClassifier) -> None:
        """Test that the disease model is put in eval mode on init.

        Args:
            classifier: DiseasePlantClassifier fixture.

        """
        assert not classifier.disease_model.training


class TestFromCheckpointsImportError:
    """Tests that from_checkpoints raises ImportError without ultralytics."""

    @staticmethod
    def test_raises_without_ultralytics(tmp_path: Path) -> None:
        """Test ImportError is raised when ultralytics is not installed.

        Args:
            tmp_path: Pytest temporary directory fixture.

        """
        fake_ckpt = tmp_path / "disease.pth"
        torch.save({"config": {}, "num_diseases": 5}, fake_ckpt)

        with (
            patch.dict("sys.modules", {"ultralytics": None}),
            pytest.raises(ImportError, match="ultralytics"),
        ):
            DiseasePlantClassifier.from_checkpoints(
                disease_checkpoint=fake_ckpt,
                yolo_checkpoint=tmp_path / "yolo.pt",
            )


class TestCropLeaves:
    """Tests for DiseasePlantClassifier._crop_leaves."""

    @staticmethod
    def test_crop_returns_correct_count() -> None:
        """Test that the number of crops equals the number of bounding boxes."""
        image = Image.new("RGB", (400, 400), color="green")
        boxes = [(10, 10, 100, 100), (200, 200, 300, 300)]
        crops = DiseasePlantClassifier._crop_leaves(image, boxes, padding=0)
        assert len(crops) == 2

    @staticmethod
    def test_crop_respects_image_bounds() -> None:
        """Test that padding does not extend crop outside the image boundary."""
        image = Image.new("RGB", (100, 100), color="green")
        boxes = [(0, 0, 50, 50)]
        crops = DiseasePlantClassifier._crop_leaves(image, boxes, padding=20)
        assert len(crops) == 1
        w, h = crops[0].size
        assert w <= 100
        assert h <= 100

    @staticmethod
    def test_empty_boxes_returns_empty() -> None:
        """Test that an empty boxes list returns no crops."""
        image = Image.new("RGB", (200, 200), color="green")
        crops = DiseasePlantClassifier._crop_leaves(image, [], padding=0)
        assert crops == []


class TestDetectLeaves:
    """Tests for DiseasePlantClassifier._detect_leaves."""

    @staticmethod
    def test_no_detections_returns_empty(classifier: DiseasePlantClassifier) -> None:
        """Test that no YOLO detections returns an empty list.

        Args:
            classifier: DiseasePlantClassifier fixture (YOLO mock returns no boxes).

        """
        image = Image.new("RGB", (224, 224), color="green")
        boxes = classifier._detect_leaves(image)
        assert boxes == []

    @staticmethod
    def test_with_detections(
        disease_model: DiseasePlantModel,
        idx_to_disease: dict[int, str],
    ) -> None:
        """Test that valid YOLO detections are parsed into bounding boxes.

        Args:
            disease_model: DiseasePlantModel fixture.
            idx_to_disease: Disease index mapping fixture.

        """
        box_tensor = torch.tensor([[10.0, 20.0, 100.0, 150.0]])
        mock_result = MagicMock()
        mock_result.boxes.xyxy = box_tensor

        mock_yolo = MagicMock()
        mock_yolo.predict.return_value = [mock_result]

        clf = DiseasePlantClassifier(
            disease_model=disease_model,
            yolo_model=mock_yolo,
            idx_to_disease=idx_to_disease,
            img_size=IMG_SIZE,
            device="cpu",
        )

        image = Image.new("RGB", (224, 224), color="green")
        boxes = clf._detect_leaves(image)
        assert boxes == [(10, 20, 100, 150)]


class TestPredict:
    """Tests for DiseasePlantClassifier.predict."""

    @staticmethod
    def test_predict_returns_required_keys(classifier: DiseasePlantClassifier) -> None:
        """Test that predict output contains all expected keys.

        Args:
            classifier: DiseasePlantClassifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = classifier.predict(image, top_k_diseases=2)

        assert "health" in result
        assert "diseases" in result
        assert "leaf_count" in result
        assert "used_full_image_fallback" in result
        assert "leaf_results" in result
        assert "processing_time_ms" in result

    @staticmethod
    def test_predict_uses_fallback_when_no_boxes(
        classifier: DiseasePlantClassifier,
    ) -> None:
        """Test that the full image is used as fallback when YOLO detects nothing.

        Args:
            classifier: DiseasePlantClassifier fixture (YOLO mock returns no boxes).

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = classifier.predict(image)

        assert result["used_full_image_fallback"] is True
        assert result["leaf_count"] == 1

    @staticmethod
    def test_predict_health_label_values(classifier: DiseasePlantClassifier) -> None:
        """Test that health label is one of the two valid values.

        Args:
            classifier: DiseasePlantClassifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = classifier.predict(image)

        assert result["health"]["label"] in {"healthy", "diseased"}

    @staticmethod
    def test_predict_disease_count(classifier: DiseasePlantClassifier) -> None:
        """Test that predict returns at most top_k_diseases disease predictions.

        Args:
            classifier: DiseasePlantClassifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = classifier.predict(image, top_k_diseases=2)

        assert len(result["diseases"]) <= 2

    @staticmethod
    def test_predict_disease_confidences_valid(classifier: DiseasePlantClassifier) -> None:
        """Test that all disease confidence scores are in [0, 1].

        Args:
            classifier: DiseasePlantClassifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = classifier.predict(image, top_k_diseases=3)

        for d in result["diseases"]:
            assert 0.0 <= d["confidence"] <= 1.0

    @staticmethod
    def test_predict_processing_time_positive(classifier: DiseasePlantClassifier) -> None:
        """Test that reported processing time is positive.

        Args:
            classifier: DiseasePlantClassifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = classifier.predict(image)

        assert result["processing_time_ms"] > 0

    @staticmethod
    def test_predict_from_path(classifier: DiseasePlantClassifier, tmp_path: Path) -> None:
        """Test that predict accepts a file path.

        Args:
            classifier: DiseasePlantClassifier fixture.
            tmp_path: Pytest temporary directory.

        """
        img_path = tmp_path / "leaf.jpg"
        Image.new("RGB", (224, 224), color="green").save(img_path)

        result = classifier.predict(img_path)
        assert "health" in result

    @staticmethod
    def test_health_threshold_affects_label(
        disease_model: DiseasePlantModel,
        mock_yolo: MagicMock,
        idx_to_disease: dict[int, str],
    ) -> None:
        """Test that a very low threshold always reports 'diseased'.

        Args:
            disease_model: DiseasePlantModel fixture.
            mock_yolo: YOLO mock fixture.
            idx_to_disease: Disease index mapping fixture.

        """
        clf_low = DiseasePlantClassifier(
            disease_model=disease_model,
            yolo_model=mock_yolo,
            idx_to_disease=idx_to_disease,
            img_size=IMG_SIZE,
            health_threshold=0.0,
            device="cpu",
        )
        image = Image.new("RGB", (224, 224), color="green")
        result = clf_low.predict(image)
        assert result["health"]["label"] == "diseased"
