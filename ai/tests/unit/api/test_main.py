"""API main module unit tests.

Copyright 2025 Plant Care Assistant
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from plant_care_ai.api.main import load_classifier, load_name_mapping

if TYPE_CHECKING:
    from collections.abc import Generator


class TestLoadNameMapping:
    """Tests for the load_name_mapping function."""

    @staticmethod
    def test_load_name_mapping_returns_empty_when_file_not_exists() -> None:
        """Test that load_name_mapping returns empty dict when file doesn't exist."""
        with patch("plant_care_ai.api.main.CLASS_MAPPING_PATH", "/nonexistent/path.json"):
            result = load_name_mapping()
            assert result == {}

    @staticmethod
    def test_load_name_mapping_returns_dict_when_file_exists(tmp_path: Path) -> None:
        """Test that load_name_mapping returns mapping when file exists.

        Args:
            tmp_path: Pytest temporary directory fixture

        """
        mapping_file = tmp_path / "class_id_to_name.json"
        test_mapping = {"1355936": "Rosa canina", "1355932": "Bellis perennis"}
        mapping_file.write_text(json.dumps(test_mapping))

        with patch("plant_care_ai.api.main.CLASS_MAPPING_PATH", str(mapping_file)):
            result = load_name_mapping()
            assert result == {"1355936": "Rosa canina", "1355932": "Bellis perennis"}

    @staticmethod
    def test_load_name_mapping_converts_int_keys_to_str(tmp_path: Path) -> None:
        """Test that load_name_mapping converts integer keys to strings.

        Args:
            tmp_path: Pytest temporary directory fixture

        """
        mapping_file = tmp_path / "class_id_to_name.json"
        # JSON doesn't support int keys, but we test the conversion logic
        test_mapping = {1355936: "Rosa canina", 1355932: "Bellis perennis"}
        mapping_file.write_text(json.dumps(test_mapping))

        with patch("plant_care_ai.api.main.CLASS_MAPPING_PATH", str(mapping_file)):
            result = load_name_mapping()
            assert all(isinstance(k, str) for k in result)


class TestLoadClassifier:
    """Tests for the load_classifier function."""

    @staticmethod
    def test_load_classifier_raises_when_checkpoint_not_exists() -> None:
        """Test that load_classifier raises FileNotFoundError when checkpoint missing."""
        with (
            patch("plant_care_ai.api.main.CHECKPOINT_PATH", "/nonexistent/model.pth"),
            pytest.raises(FileNotFoundError, match="Checkpoint not found"),
        ):
            load_classifier()

    @staticmethod
    def test_load_classifier_loads_checkpoint_and_mapping(tmp_path: Path) -> None:
        """Test that load_classifier loads checkpoint and sets name mapping.

        Args:
            tmp_path: Pytest temporary directory fixture

        """
        # Create mock checkpoint
        checkpoint_path = tmp_path / "best.pth"
        mapping_path = tmp_path / "class_id_to_name.json"

        test_mapping = {"1355936": "Rosa canina"}
        mapping_path.write_text(json.dumps(test_mapping))

        mock_classifier = MagicMock()
        mock_classifier.num_classes = 100

        with (
            patch("plant_care_ai.api.main.CHECKPOINT_PATH", str(checkpoint_path)),
            patch("plant_care_ai.api.main.CLASS_MAPPING_PATH", str(mapping_path)),
            patch(
                "plant_care_ai.api.main.PlantClassifier.from_checkpoint",
                return_value=mock_classifier,
            ),
        ):
            checkpoint_path.write_text("")  # Create empty file

            result = load_classifier()

            assert result == mock_classifier
            mock_classifier.set_name_mapping.assert_called_once_with(test_mapping)

    @staticmethod
    def test_load_classifier_without_name_mapping(tmp_path: Path) -> None:
        """Test that load_classifier works without name mapping file.

        Args:
            tmp_path: Pytest temporary directory fixture

        """
        checkpoint_path = tmp_path / "best.pth"
        checkpoint_path.write_text("")

        mock_classifier = MagicMock()
        mock_classifier.num_classes = 100

        with (
            patch("plant_care_ai.api.main.CHECKPOINT_PATH", str(checkpoint_path)),
            patch("plant_care_ai.api.main.CLASS_MAPPING_PATH", "/nonexistent/path.json"),
            patch(
                "plant_care_ai.api.main.PlantClassifier.from_checkpoint",
                return_value=mock_classifier,
            ),
        ):
            result = load_classifier()

            assert result == mock_classifier
            mock_classifier.set_name_mapping.assert_not_called()


class TestLifespan:
    """Tests for the application lifespan handler."""

    @staticmethod
    @pytest.fixture
    def mock_load_classifier() -> "Generator[MagicMock, None, None]":
        """Mock the load_classifier function.

        Yields:
            MagicMock for load_classifier

        """
        mock_classifier = MagicMock()
        mock_classifier.num_classes = 100

        with patch(
            "plant_care_ai.api.main.load_classifier",
            return_value=mock_classifier,
        ) as mock:
            yield mock

    @staticmethod
    @pytest.mark.asyncio
    async def test_lifespan_loads_classifier(mock_load_classifier: MagicMock) -> None:
        """Test that lifespan handler loads classifier on startup.

        Args:
            mock_load_classifier: Mocked load_classifier function

        """
        import plant_care_ai.api.main as main_module  # noqa: PLC0415
        from plant_care_ai.api.main import lifespan  # noqa: PLC0415

        original_classifier = main_module.classifier
        main_module.classifier = None

        try:
            mock_app = MagicMock()
            async with lifespan(mock_app):
                mock_load_classifier.assert_called_once()
                assert main_module.classifier is not None
        finally:
            main_module.classifier = original_classifier

    @staticmethod
    @pytest.mark.asyncio
    async def test_lifespan_raises_on_missing_checkpoint() -> None:
        """Test that lifespan raises FileNotFoundError when checkpoint missing."""
        import plant_care_ai.api.main as main_module  # noqa: PLC0415
        from plant_care_ai.api.main import lifespan  # noqa: PLC0415

        original_classifier = main_module.classifier
        main_module.classifier = None

        with patch(
            "plant_care_ai.api.main.load_classifier",
            side_effect=FileNotFoundError("Checkpoint not found"),
        ):
            try:
                mock_app = MagicMock()
                with pytest.raises(FileNotFoundError):
                    async with lifespan(mock_app):
                        pass
            finally:
                main_module.classifier = original_classifier
