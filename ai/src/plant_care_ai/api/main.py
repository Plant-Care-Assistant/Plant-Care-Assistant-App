"""AI inference API for plant identification.

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

from plant_care_ai.inference import PlantClassifier

# ===== CONFIGURATION =====
CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "/app/models/best.pth")
DEVICE = os.getenv("DEVICE", "cpu")
CLASS_MAPPING_PATH = os.getenv("CLASS_MAPPING_PATH", "/app/models/class_id_to_name.json")

# Constants
MAX_TOP_K = 20
MIN_TOP_K = 1
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ===== GLOBAL CLASSIFIER =====
classifier: PlantClassifier | None = None


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

    # Detect format
    first_value = next(iter(raw_mapping.values()), None)
    if isinstance(first_value, dict) and "class_id" in first_value:
        # Extended format from training: {"0": {"class_id": "123", "name": "..."}}
        return {v["class_id"]: v["name"] for v in raw_mapping.values()}

    # Simple format: {"class_id": "name"}
    return {str(k): v for k, v in raw_mapping.items()}


def load_classifier() -> PlantClassifier:
    """Load classifier from checkpoint.

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

    # Load name mapping if available
    name_mapping = load_name_mapping()
    if name_mapping:
        clf.set_name_mapping(name_mapping)
        print(f"Loaded {len(name_mapping)} class name mappings")

    return clf


# ===== LIFESPAN =====
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: RUF029
    """Application lifespan handler for startup/shutdown events.

    Raises:
        FileNotFoundError: If model checkpoint is not found during startup

    """
    global classifier  # noqa: PLW0603

    # Startup
    print("\n" + "=" * 60)
    print("Loading Plant Care AI Service...")
    print("=" * 60)

    try:
        classifier = load_classifier()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please provide a valid checkpoint file.")
        raise

    print("=" * 60)
    print("Plant Care AI Service Started!")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Classes: {classifier.num_classes}")
    print("=" * 60)
    print("API available at: http://0.0.0.0:8001")
    print("Documentation: http://0.0.0.0:8001/docs")
    print("=" * 60 + "\n")

    yield

    # Shutdown (nothing to do)


# ===== FASTAPI APP =====
app = FastAPI(
    title="Plant Care AI Service",
    description="Plant identification inference API using PyTorch models",
    version="2.0.0",
    lifespan=lifespan,
)


# ===== RESPONSE MODELS =====
class PredictionResult(BaseModel):
    """Single prediction result."""

    class_id: str = Field(..., description="Plant species ID")
    class_name: str | None = Field(None, description="Plant species name (if available)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0-1]")


class PredictionResponse(BaseModel):
    """Response with top-K predictions."""

    predictions: list[PredictionResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    device: str
    num_classes: int
    checkpoint_loaded: bool


# ===== ENDPOINTS =====
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
        HTTPException: If file type is invalid or prediction fails

    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG allowed.",
        )

    # Validate top_k
    if not MIN_TOP_K <= top_k <= MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}",
        )

    # Read image and validate size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run inference using PlantClassifier
        result = classifier.predict(image, top_k=top_k)

        # Format response
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e


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
            "predict": "/predict",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104
