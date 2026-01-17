"""AI inference API for plant identification.

Copyright (c) 2026 Plant Care Assistant. All rights reserved.
"""

import io
import json
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from plant_care_ai.data.preprocessing import get_inference_pipeline
from plant_care_ai.models.load_models import get_model

# ===== CONFIGURATION =====
MODEL_NAME = os.getenv("MODEL_NAME", "resnet18")
WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "/app/models/best_model.pth")
DEVICE = os.getenv("DEVICE", "cpu")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "100"))
CLASS_MAPPING_PATH = os.getenv("CLASS_MAPPING_PATH", "/app/models/class_id_to_name.json")

# Constants
MAX_TOP_K = 20
MIN_TOP_K = 1

# ===== LOAD CLASS MAPPING =====
CLASS_ID_TO_NAME: dict[int, str] = {}


def load_class_mapping() -> dict[int, str]:
    """Load class ID to name mapping from JSON file.

    Returns:
        Dictionary mapping class_id (int) to class_name (str)

    """
    if Path(CLASS_MAPPING_PATH).exists():
        with Path(CLASS_MAPPING_PATH).open(encoding="utf-8") as f:
            raw_mapping = json.load(f)
            # Convert string keys to int keys
            return {int(k): v for k, v in raw_mapping.items()}
    print(f"Warning: Class mapping file not found at {CLASS_MAPPING_PATH}")
    print("Using placeholder names.")
    return {}


CLASS_ID_TO_NAME = load_class_mapping()

# ===== LOAD MODEL AT STARTUP =====
print("=" * 60)
print(f"Loading model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Weights path: {WEIGHTS_PATH}")
print(f"Number of classes: {NUM_CLASSES}")
print("=" * 60)

model = get_model(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    weights_path=WEIGHTS_PATH if Path(WEIGHTS_PATH).exists() else None,
    device=DEVICE,
)
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")
print(f"Loaded {len(CLASS_ID_TO_NAME)} class name mappings")
print("=" * 60)

# ===== PREPROCESSING PIPELINE =====
transform = get_inference_pipeline(img_size=224)


# ===== LIFESPAN =====
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: RUF029
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print("\n" + "=" * 60)
    print("Plant Care AI Service Started!")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Class mappings loaded: {len(CLASS_ID_TO_NAME)}")
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
    version="1.0.0",
    lifespan=lifespan,
)


# ===== RESPONSE MODELS =====
class PredictionResult(BaseModel):
    """Single prediction result."""

    class_id: int = Field(..., description="Numeric class ID")
    class_name: str = Field(..., description="Plant species name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0-1]")


class PredictionResponse(BaseModel):
    """Response with top-K predictions."""

    predictions: list[PredictionResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    device: str
    num_classes: int
    has_weights: bool
    num_class_mappings: int


# ===== ENDPOINTS =====
@app.get("/health")
def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status with model information

    """
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        has_weights=Path(WEIGHTS_PATH).exists(),
        num_class_mappings=len(CLASS_ID_TO_NAME),
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

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

        # Inference
        start_time = time.time()

        with torch.no_grad():
            logits = model(input_tensor)  # (1, num_classes)
            probabilities = torch.softmax(logits, dim=1)  # Convert to probabilities

        processing_time_ms = (time.time() - start_time) * 1000

        # Get top-K predictions
        top_k_clamped = min(top_k, NUM_CLASSES)  # Ensure top_k <= num_classes
        top_probs, top_indices = torch.topk(probabilities[0], k=top_k_clamped)

        # Format results
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy(), strict=False):
            class_id = int(idx)
            class_name = CLASS_ID_TO_NAME.get(class_id, f"Unknown Species (ID: {class_id})")

            predictions.append(
                PredictionResult(class_id=class_id, class_name=class_name, confidence=float(prob))
            )

        return PredictionResponse(predictions=predictions, processing_time_ms=processing_time_ms)

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
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104
