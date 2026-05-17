"""AI inference API for plant identification and disease detection.

Copyright (c) 2026 Plant Care Assistant. All rights reserved.
"""

import io
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from plant_care_ai.inference import CombinedPlantClassifier, DiseasePlantClassifier, PlantClassifier

CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT_PATH",
    str(Path(__file__).parent.parent.parent.parent / "models/best.pth"),
)
DISEASE_CHECKPOINT_PATH = os.getenv("DISEASE_CHECKPOINT_PATH", "")
YOLO_CHECKPOINT_PATH = os.getenv("YOLO_CHECKPOINT_PATH", "")
DEVICE = os.getenv("DEVICE", "cpu")
CLASS_MAPPING_PATH = os.getenv(
    "CLASS_MAPPING_PATH",
    str(Path(__file__).parent.parent.parent.parent / "models/class_id_to_name.json"),
)

MAX_TOP_K = 20
MIN_TOP_K = 1
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

classifier: PlantClassifier | None = None
disease_classifier: DiseasePlantClassifier | None = None
combined_classifier: CombinedPlantClassifier | None = None


def load_name_mapping() -> dict[str, str]:
    """Load plant ID to name mapping from JSON file.

    Supports two formats:
    1. Simple: {"class_id": "name", ...}
    2. Extended (from kaggle training): {"idx": {"class_id": "...", "name": "..."}, ...}

    Returns:
        Dictionary mapping plant_id (str) to plant_name (str)

    """
    if not Path(CLASS_MAPPING_PATH).exists():
        print(f"Warning: Class mapping file not found at {CLASS_MAPPING_PATH}")
        return {}

    with Path(CLASS_MAPPING_PATH).open(encoding="utf-8") as f:
        raw_mapping = json.load(f)

    first_value = next(iter(raw_mapping.values()), None)
    if isinstance(first_value, dict) and "class_id" in first_value:
        return {v["class_id"]: v["name"] for v in raw_mapping.values()}

    return {str(k): v for k, v in raw_mapping.items()}


def load_classifier() -> PlantClassifier:
    """Load species classifier from checkpoint.

    Returns:
        Initialized PlantClassifier

    Raises:
        FileNotFoundError: If checkpoint file does not exist

    """
    checkpoint_path = Path(CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        msg = f"Checkpoint not found at {CHECKPOINT_PATH}"
        raise FileNotFoundError(msg)

    clf = PlantClassifier.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=DEVICE,
        verbose=True,
    )

    name_mapping = load_name_mapping()
    if name_mapping:
        clf.set_name_mapping(name_mapping)
        print(f"Loaded {len(name_mapping)} class name mappings")

    return clf


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events.

    Raises:
        FileNotFoundError: If model checkpoint is not found during startup

    """
    global classifier, disease_classifier, combined_classifier  # noqa: PLW0603

    print("\n" + "=" * 60)
    print("Loading Plant Care AI Service...")
    print("=" * 60)

    try:
        classifier = load_classifier()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please provide a valid checkpoint file.")
        raise

    if DISEASE_CHECKPOINT_PATH and YOLO_CHECKPOINT_PATH:
        try:
            disease_classifier = DiseasePlantClassifier.from_checkpoints(
                disease_checkpoint=DISEASE_CHECKPOINT_PATH,
                yolo_checkpoint=YOLO_CHECKPOINT_PATH,
                device=DEVICE,
                verbose=True,
            )
            combined_classifier = CombinedPlantClassifier(disease_classifier, classifier)
            print("Disease + combined classifiers loaded.")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Disease classifier not loaded: {e}")

    print("=" * 60)
    print("Plant Care AI Service Started!")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Classes: {classifier.num_classes}")
    print(f"Disease detection: {'enabled' if disease_classifier else 'disabled'}")
    print("=" * 60)
    print("API available at: http://0.0.0.0:8001")
    print("Documentation: http://0.0.0.0:8001/docs")
    print("=" * 60 + "\n")

    yield


app = FastAPI(
    title="Plant Care AI Service",
    description="Plant identification and disease detection inference API",
    version="2.0.0",
    lifespan=lifespan,
)


class PredictionResult(BaseModel):
    """Single species prediction result."""

    class_id: str = Field(..., description="Plant species ID")
    class_name: str | None = Field(None, description="Plant species name (if available)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0-1]")


class PredictionResponse(BaseModel):
    """Response with top-K species predictions."""

    predictions: list[PredictionResult]
    processing_time_ms: float


class HealthSummary(BaseModel):
    """Plant health assessment."""

    label: str = Field(..., description="'healthy' or 'diseased'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    logit: float = Field(..., description="Raw sigmoid output (higher = more diseased)")


class DiseaseResult(BaseModel):
    """Single disease prediction."""

    disease: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class LeafResult(BaseModel):
    """Per-leaf detection and classification result."""

    leaf_index: int
    bbox: tuple[int, int, int, int] | None
    health_logit: float
    health_label: str
    top_disease: str
    top_disease_conf: float


class DiseaseResponse(BaseModel):
    """Response from disease detection endpoint."""

    health: HealthSummary
    diseases: list[DiseaseResult]
    leaf_count: int
    used_full_image_fallback: bool
    leaf_results: list[LeafResult]
    processing_time_ms: float


class CombinedResponse(BaseModel):
    """Response from combined species + disease endpoint."""

    species: list[PredictionResult]
    health: HealthSummary
    diseases: list[DiseaseResult]
    leaf_count: int
    used_full_image_fallback: bool
    leaf_results: list[LeafResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    device: str
    num_classes: int
    checkpoint_loaded: bool
    disease_detection_available: bool


async def _read_image(file: UploadFile) -> Image.Image:
    """Validate and read an uploaded image file.

    Args:
        file: The uploaded file from the request.

    Returns:
        PIL Image in RGB mode.

    Raises:
        HTTPException: On invalid file type or file too large.

    """
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG allowed.",
        )
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _validate_top_k(top_k: int) -> None:
    """Raise HTTPException if top_k is out of range.

    Args:
        top_k: Requested number of top predictions.

    Raises:
        HTTPException: If top_k is outside [MIN_TOP_K, MAX_TOP_K].

    """
    if not MIN_TOP_K <= top_k <= MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}",
        )


@app.get("/health")
def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status with model information

    """
    return HealthResponse(
        status="healthy" if classifier else "not_ready",
        device=DEVICE,
        num_classes=classifier.num_classes if classifier else 0,
        checkpoint_loaded=classifier is not None,
        disease_detection_available=disease_classifier is not None,
    )


@app.post("/predict")
async def predict_plant(
    file: Annotated[UploadFile, File(description="Plant image (JPEG/PNG)")],
    top_k: int = 5,
) -> PredictionResponse:
    """Identify plant species from uploaded image.

    Args:
        file: Uploaded image file
        top_k: Number of top predictions to return (default: 5, max: 20)

    Returns:
        PredictionResponse with top-K predictions sorted by confidence

    Raises:
        HTTPException: If file type is invalid, model not loaded, or prediction fails

    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _validate_top_k(top_k)

    try:
        image = await _read_image(file)
        result = classifier.predict(image, top_k=top_k)

        predictions = [
            PredictionResult(
                class_id=pred["class_id"],
                class_name=pred.get("class_name"),
                confidence=pred["confidence"],
            )
            for pred in result["predictions"]
        ]

        return PredictionResponse(
            predictions=predictions,
            processing_time_ms=result["processing_time_ms"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e


@app.post("/predict/disease")
async def predict_disease(
    file: Annotated[UploadFile, File(description="Plant image (JPEG/PNG)")],
    top_k_diseases: int = 3,
) -> DiseaseResponse:
    """Detect plant health status and diseases from uploaded image.

    Uses YOLO to detect individual leaves, then classifies each leaf
    for health status and disease type.

    Args:
        file: Uploaded image file
        top_k_diseases: Number of top disease predictions to return (default: 3)

    Returns:
        DiseaseResponse with health assessment and disease predictions

    Raises:
        HTTPException: If disease detection is not configured or prediction fails

    """
    if disease_classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Disease detection not available. "
                "Set DISEASE_CHECKPOINT_PATH and YOLO_CHECKPOINT_PATH env vars."
            ),
        )

    _validate_top_k(top_k_diseases)

    try:
        image = await _read_image(file)
        result = disease_classifier.predict(image, top_k_diseases=top_k_diseases)

        return DiseaseResponse(
            health=HealthSummary(**result["health"]),
            diseases=[DiseaseResult(**d) for d in result["diseases"]],
            leaf_count=result["leaf_count"],
            used_full_image_fallback=result["used_full_image_fallback"],
            leaf_results=[LeafResult(**lr) for lr in result["leaf_results"]],
            processing_time_ms=result["processing_time_ms"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {e!s}") from e


@app.post("/predict/combined")
async def predict_combined(
    file: Annotated[UploadFile, File(description="Plant image (JPEG/PNG)")],
    top_k_species: int = 5,
    top_k_diseases: int = 3,
) -> CombinedResponse:
    """Identify plant species and detect diseases in a single request.

    Args:
        file: Uploaded image file
        top_k_species: Number of top species predictions to return (default: 5)
        top_k_diseases: Number of top disease predictions to return (default: 3)

    Returns:
        CombinedResponse with species identification and disease detection results

    Raises:
        HTTPException: If combined classifier is not configured or prediction fails

    """
    if combined_classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Combined classifier not available. "
                "Set DISEASE_CHECKPOINT_PATH and YOLO_CHECKPOINT_PATH env vars."
            ),
        )

    _validate_top_k(top_k_species)
    _validate_top_k(top_k_diseases)

    try:
        image = await _read_image(file)
        result = combined_classifier.predict(
            image,
            top_k_species=top_k_species,
            top_k_diseases=top_k_diseases,
        )

        species = [
            PredictionResult(
                class_id=pred["class_id"],
                class_name=pred.get("class_name"),
                confidence=pred["confidence"],
            )
            for pred in result["species"]
        ]

        return CombinedResponse(
            species=species,
            health=HealthSummary(**result["health"]),
            diseases=[DiseaseResult(**d) for d in result["diseases"]],
            leaf_count=result["leaf_count"],
            used_full_image_fallback=result["used_full_image_fallback"],
            leaf_results=[LeafResult(**lr) for lr in result["leaf_results"]],
            processing_time_ms=result["processing_time_ms"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined prediction failed: {e!s}") from e


@app.get("/")
def root() -> dict[str, str | dict[str, str]]:
    """Root endpoint with API information.

    Returns:
        Dictionary with service info and available endpoints

    """
    return {
        "service": "Plant Care AI",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict_species": "/predict",
            "predict_disease": "/predict/disease",
            "predict_combined": "/predict/combined",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104
