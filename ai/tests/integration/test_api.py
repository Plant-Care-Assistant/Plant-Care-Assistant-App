"""Integration tests for AI API service.

Copyright (c) 2026 Plant Care Assistant. All rights reserved.
"""

from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import plant_care_ai.api.main as main_module
from plant_care_ai.api.main import app

client = TestClient(app)

# Constants for expected values
DEFAULT_TOP_K = 5
TOP_K_TEST_VALUE = 3


# ===== Health Endpoint Tests =====


def test_health_check_returns_200() -> None:
    """Health endpoint should return 200 OK."""
    response = client.get("/health")
    assert response.status_code == HTTPStatus.OK


def test_health_check_contains_model_info() -> None:
    """Health endpoint should return model information."""
    response = client.get("/health")
    data = response.json()

    assert data["status"] in {"healthy", "not_ready"}
    assert "device" in data
    assert "num_classes" in data
    assert isinstance(data["checkpoint_loaded"], bool)


# ===== Root Endpoint Tests =====


def test_root_returns_api_info() -> None:
    """Root endpoint should return API information."""
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    data = response.json()

    assert "service" in data
    assert "version" in data
    assert "endpoints" in data


# ===== Predict Endpoint Tests =====


@pytest.mark.usefixtures("mock_classifier")
def test_predict_with_valid_jpeg_image(test_image_path: Path) -> None:
    """Predict endpoint should accept valid JPEG images."""
    if not test_image_path.exists():
        pytest.skip("Test image not found")

    with test_image_path.open("rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")},
            params={"top_k": TOP_K_TEST_VALUE},
        )

    assert response.status_code == HTTPStatus.OK
    data = response.json()

    # Check response structure
    assert "predictions" in data
    assert "processing_time_ms" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == TOP_K_TEST_VALUE

    # Check prediction structure
    for pred in data["predictions"]:
        assert "class_id" in pred
        assert "confidence" in pred
        assert isinstance(pred["class_id"], str)
        # class_name is optional (only if name mapping is loaded)
        if "class_name" in pred and pred["class_name"] is not None:
            assert isinstance(pred["class_name"], str)
        assert 0.0 <= pred["confidence"] <= 1.0

    # Predictions should be sorted by confidence (descending)
    confidences = [pred["confidence"] for pred in data["predictions"]]
    assert confidences == sorted(confidences, reverse=True)


@pytest.mark.usefixtures("mock_classifier")
def test_predict_with_default_top_k(test_image_path: Path) -> None:
    """Predict should use default top_k=5 if not specified."""
    if not test_image_path.exists():
        pytest.skip("Test image not found")

    with test_image_path.open("rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert len(data["predictions"]) == DEFAULT_TOP_K


@pytest.mark.usefixtures("mock_classifier")
def test_predict_with_top_k_1(test_image_path: Path) -> None:
    """Predict should return only 1 prediction when top_k=1."""
    if not test_image_path.exists():
        pytest.skip("Test image not found")

    with test_image_path.open("rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")},
            params={"top_k": 1},
        )

    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert len(data["predictions"]) == 1


@pytest.mark.usefixtures("mock_classifier")
def test_predict_with_invalid_file_type() -> None:
    """Predict should reject non-image file types."""
    response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "Invalid file type" in response.json()["detail"]


@pytest.mark.usefixtures("mock_classifier")
def test_predict_with_invalid_top_k_too_low(test_image_path: Path) -> None:
    """Predict should reject top_k < 1."""
    if not test_image_path.exists():
        pytest.skip("Test image not found")

    with test_image_path.open("rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")},
            params={"top_k": 0},
        )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "top_k must be between 1 and 20" in response.json()["detail"]


@pytest.mark.usefixtures("mock_classifier")
def test_predict_with_invalid_top_k_too_high(test_image_path: Path) -> None:
    """Predict should reject top_k > 20."""
    if not test_image_path.exists():
        pytest.skip("Test image not found")

    with test_image_path.open("rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")},
            params={"top_k": 21},
        )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "top_k must be between 1 and 20" in response.json()["detail"]


@pytest.mark.usefixtures("mock_classifier")
def test_predict_processing_time_is_positive(test_image_path: Path) -> None:
    """Processing time should be a positive number."""
    if not test_image_path.exists():
        pytest.skip("Test image not found")

    with test_image_path.open("rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert data["processing_time_ms"] > 0


def test_predict_without_classifier_returns_503() -> None:
    """Predict should return 503 when classifier is not loaded."""
    # Ensure classifier is None
    original = main_module.classifier
    main_module.classifier = None

    try:
        response = client.post(
            "/predict", files={"file": ("test.jpg", b"fake image data", "image/jpeg")}
        )
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "Model not loaded" in response.json()["detail"]
    finally:
        main_module.classifier = original


@pytest.mark.usefixtures("mock_classifier")
def test_predict_handles_classifier_exception(test_image_path: Path) -> None:
    """Predict should return 500 when classifier raises an exception."""
    if not test_image_path.exists():
        pytest.skip("Test image not found")

    # Create a mock that raises an exception
    mock = MagicMock()
    mock.num_classes = 100
    mock.predict.side_effect = RuntimeError("Model inference failed")

    original = main_module.classifier
    main_module.classifier = mock

    try:
        with test_image_path.open("rb") as f:
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert "Prediction failed" in response.json()["detail"]
    finally:
        main_module.classifier = original


# ===== API Documentation Tests =====


def test_openapi_docs_available() -> None:
    """OpenAPI docs should be available at /docs."""
    response = client.get("/docs")
    assert response.status_code == HTTPStatus.OK


def test_openapi_json_available() -> None:
    """OpenAPI JSON schema should be available."""
    response = client.get("/openapi.json")
    assert response.status_code == HTTPStatus.OK
    data = response.json()

    assert "openapi" in data
    assert "info" in data
    assert "paths" in data
    assert "/health" in data["paths"]
    assert "/predict" in data["paths"]
