"""CombinedPlantClassifier unit tests.

Copyright 2026 Plant Care Assistant
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from plant_care_ai.inference.combined_classifier import CombinedPlantClassifier
from plant_care_ai.inference.disease_classifier import DiseasePlantClassifier
from plant_care_ai.models.disease_model import DiseasePlantModel
from plant_care_ai.models.resnet18 import Resnet18

NUM_DISEASES = 4
NUM_SPECIES = 5
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
    }


@pytest.fixture
def idx_to_species() -> dict[int, str]:
    """Return a small species index mapping.

    Returns:
        dict[int, str]: Mapping from class index to species ID.

    """
    return {i: str(1000000 + i) for i in range(NUM_SPECIES)}


@pytest.fixture
def mock_yolo() -> MagicMock:
    """Return a YOLO mock that reports no detections.

    Returns:
        MagicMock behaving like a YOLO model.

    """
    result = MagicMock()
    result.boxes = None
    mock = MagicMock()
    mock.predict.return_value = [result]
    return mock


@pytest.fixture
def disease_clf(
    idx_to_disease: dict[int, str],
    mock_yolo: MagicMock,
) -> DiseasePlantClassifier:
    """Return a DiseasePlantClassifier backed by a tiny model.

    Args:
        idx_to_disease: Disease index mapping fixture.
        mock_yolo: YOLO mock fixture.

    Returns:
        DiseasePlantClassifier ready for CPU testing.

    """
    model = DiseasePlantModel(
        model_name="efficientnetv2",
        num_diseases=NUM_DISEASES,
        variant="b0",
        pretrained=False,
    )
    return DiseasePlantClassifier(
        disease_model=model,
        yolo_model=mock_yolo,
        idx_to_disease=idx_to_disease,
        img_size=IMG_SIZE,
        device="cpu",
    )


@pytest.fixture
def species_clf() -> MagicMock:
    """Return a mock PlantClassifier that returns predictable species output.

    Returns:
        MagicMock behaving like a PlantClassifier.

    """
    predictions = [
        {"class_id": str(1000000 + i), "class_name": f"Plant {i}", "confidence": 0.1}
        for i in range(NUM_SPECIES)
    ]
    mock = MagicMock()
    mock.predict.return_value = {
        "predictions": predictions[:3],
        "processing_time_ms": 10.0,
    }
    return mock


@pytest.fixture
def combined_clf(
    disease_clf: DiseasePlantClassifier,
    species_clf: MagicMock,
) -> CombinedPlantClassifier:
    """Return a CombinedPlantClassifier composed of disease + species fixtures.

    Args:
        disease_clf: DiseasePlantClassifier fixture.
        species_clf: Mock species classifier fixture.

    Returns:
        Ready-to-use CombinedPlantClassifier.

    """
    return CombinedPlantClassifier(
        disease_classifier=disease_clf,
        species_classifier=species_clf,
    )


# ===== Tests =====


class TestInit:
    """Tests for CombinedPlantClassifier.__init__."""

    @staticmethod
    def test_attributes_assigned(
        combined_clf: CombinedPlantClassifier,
        disease_clf: DiseasePlantClassifier,
        species_clf: MagicMock,
    ) -> None:
        """Test that disease and species classifiers are stored.

        Args:
            combined_clf: CombinedPlantClassifier fixture.
            disease_clf: DiseasePlantClassifier fixture.
            species_clf: Mock species classifier fixture.

        """
        assert combined_clf.disease is disease_clf
        assert combined_clf.species is species_clf


class TestPredict:
    """Tests for CombinedPlantClassifier.predict."""

    @staticmethod
    def test_returns_required_keys(combined_clf: CombinedPlantClassifier) -> None:
        """Test that predict output contains all expected keys.

        Args:
            combined_clf: CombinedPlantClassifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = combined_clf.predict(image)

        assert "species" in result
        assert "health" in result
        assert "diseases" in result
        assert "leaf_count" in result
        assert "used_full_image_fallback" in result
        assert "leaf_results" in result
        assert "processing_time_ms" in result

    @staticmethod
    def test_species_predictions_present(combined_clf: CombinedPlantClassifier) -> None:
        """Test that species predictions are included in the response.

        Args:
            combined_clf: CombinedPlantClassifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = combined_clf.predict(image, top_k_species=3)

        assert len(result["species"]) == 3

    @staticmethod
    def test_processing_time_is_sum(
        combined_clf: CombinedPlantClassifier,
        species_clf: MagicMock,
    ) -> None:
        """Test that processing_time_ms is the sum of both classifiers' times.

        Args:
            combined_clf: CombinedPlantClassifier fixture.
            species_clf: Mock species classifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        result = combined_clf.predict(image)

        species_time = species_clf.predict.return_value["processing_time_ms"]
        disease_time = combined_clf.disease.predict(image)["processing_time_ms"]

        assert result["processing_time_ms"] == pytest.approx(species_time + disease_time, rel=0.1)

    @staticmethod
    def test_accepts_file_path(combined_clf: CombinedPlantClassifier, tmp_path: Path) -> None:
        """Test that predict accepts a file path as input.

        Args:
            combined_clf: CombinedPlantClassifier fixture.
            tmp_path: Pytest temporary directory.

        """
        img_path = tmp_path / "plant.jpg"
        Image.new("RGB", (224, 224), color="green").save(img_path)

        result = combined_clf.predict(img_path)
        assert "species" in result

    @staticmethod
    def test_species_classifier_called_once(
        combined_clf: CombinedPlantClassifier,
        species_clf: MagicMock,
    ) -> None:
        """Test that the species classifier is called exactly once per predict.

        Args:
            combined_clf: CombinedPlantClassifier fixture.
            species_clf: Mock species classifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        combined_clf.predict(image)

        species_clf.predict.assert_called_once()

    @staticmethod
    def test_top_k_species_forwarded(
        combined_clf: CombinedPlantClassifier,
        species_clf: MagicMock,
    ) -> None:
        """Test that top_k_species is forwarded to the species classifier.

        Args:
            combined_clf: CombinedPlantClassifier fixture.
            species_clf: Mock species classifier fixture.

        """
        image = Image.new("RGB", (224, 224), color="green")
        combined_clf.predict(image, top_k_species=2)

        _, kwargs = species_clf.predict.call_args
        assert kwargs.get("top_k") == 2


class TestInitFromCheckpoints:
    """Tests for CombinedPlantClassifier.from_checkpoints path (mocked)."""

    @staticmethod
    def test_from_checkpoints_builds_combined(tmp_path: Path) -> None:
        """Test that from_checkpoints returns a CombinedPlantClassifier.

        Args:
            tmp_path: Pytest temporary directory.

        """
        # Build minimal disease checkpoint
        disease_model = DiseasePlantModel(
            model_name="efficientnetv2",
            num_diseases=NUM_DISEASES,
            variant="b0",
            pretrained=False,
        )
        disease_ckpt = tmp_path / "disease.pth"
        torch.save(
            {
                "config": {"model_name": "efficientnetv2", "variant": "b0", "img_size": IMG_SIZE},
                "num_diseases": NUM_DISEASES,
                "model_state_dict": disease_model.state_dict(),
                "idx_to_disease": {i: f"disease_{i}" for i in range(NUM_DISEASES)},
            },
            disease_ckpt,
        )

        # Build minimal species checkpoint
        species_model = Resnet18(num_classes=NUM_SPECIES)
        species_ckpt = tmp_path / "species.pth"
        torch.save(
            {
                "config": {"model": "resnet18", "img_size": IMG_SIZE},
                "num_classes": NUM_SPECIES,
                "model_state_dict": species_model.state_dict(),
                "idx_to_class": {i: str(1000000 + i) for i in range(NUM_SPECIES)},
            },
            species_ckpt,
        )

        # Mock YOLO to avoid ultralytics dependency in test environment
        mock_yolo_cls = MagicMock()
        mock_yolo_instance = MagicMock()
        result = MagicMock()
        result.boxes = None
        mock_yolo_instance.predict.return_value = [result]
        mock_yolo_cls.return_value = mock_yolo_instance

        import sys  # noqa: PLC0415

        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO = mock_yolo_cls
        sys.modules.setdefault("ultralytics", mock_ultralytics)

        clf = CombinedPlantClassifier.from_checkpoints(
            disease_checkpoint=disease_ckpt,
            species_checkpoint=species_ckpt,
            yolo_checkpoint=tmp_path / "yolo.pt",
            verbose=False,
        )

        assert isinstance(clf, CombinedPlantClassifier)
        assert isinstance(clf.disease, DiseasePlantClassifier)
