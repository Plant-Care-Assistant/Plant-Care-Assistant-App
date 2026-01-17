#!/usr/bin/env python3
"""Simple script to test AI API endpoints.

Usage:
    python scripts/test_api.py
    python scripts/test_api.py --image path/to/image.jpg

Copyright (c) 2026 Plant Care Assistant
"""

import argparse
import sys
import time
from pathlib import Path

import requests


def test_health(base_url: str) -> bool:
    """Test /health endpoint.

    Args:
        base_url: The base URL of the API.

    Returns:
        True if health check passed, False otherwise.

    """
    print("\n" + "=" * 60)
    print("Testing /health endpoint...")
    print("=" * 60)

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()

        print("✓ Health check passed!")
        print(f"  Status: {data['status']}")
        print(f"  Device: {data['device']}")
        print(f"  Classes: {data['num_classes']}")
        print(f"  Checkpoint loaded: {data['checkpoint_loaded']}")

    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        return False
    else:
        return True


def test_predict(image_path: Path, base_url: str, top_k: int) -> bool:
    """Test /predict endpoint with an image.

    Args:
        image_path: Path to the image file.
        base_url: The base URL of the API.
        top_k: Number of top predictions to return.

    Returns:
        True if prediction was successful, False otherwise.

    """
    print("\n" + "=" * 60)
    print("Testing /predict endpoint...")
    print("=" * 60)

    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        return False

    print(f"Image: {image_path}")
    print(f"Top K: {top_k}")

    try:
        with image_path.open("rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            params = {"top_k": top_k}

            start = time.time()
            response = requests.post(f"{base_url}/predict", files=files, params=params, timeout=30)
            elapsed = (time.time() - start) * 1000

            response.raise_for_status()
            data = response.json()

        print("✓ Prediction successful!")
        print(f"  Total request time: {elapsed:.1f}ms")
        print(f"  Processing time: {data['processing_time_ms']:.1f}ms")
        print(f"  Network overhead: {elapsed - data['processing_time_ms']:.1f}ms")
        print("\nTop predictions:")

        for i, pred in enumerate(data["predictions"], 1):
            confidence_pct = pred["confidence"] * 100
            class_name = pred.get("class_name") or "Unknown"
            print(
                f"  {i}. {class_name} (ID: {pred['class_id']}, confidence: {confidence_pct:.2f}%)"
            )

    except requests.exceptions.RequestException as e:
        print(f"✗ Prediction failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.text}")
        return False
    else:
        return True


def find_test_image() -> Path | None:
    """Find a test image in the artifacts directory.

    Returns:
        Path to a test image if found, None otherwise.

    """
    test_dirs = [
        Path("tests/artifacts/images/test"),
        Path("tests/artifacts/images/val"),
        Path("tests/artifacts/images/train"),
    ]

    for test_dir in test_dirs:
        if test_dir.exists():
            images = list(test_dir.rglob("*.jpg"))
            if images:
                return images[0]

    return None


def main() -> None:
    """Run API tests."""
    parser = argparse.ArgumentParser(description="Test Plant Care AI API")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--image", type=Path, help="Path to test image")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Plant Care AI - API Tester")
    print("=" * 60)
    print(f"API URL: {args.url}")

    # Test health endpoint
    if not test_health(args.url):
        print("\n✗ API is not available. Make sure it's running:")
        print("  uvicorn plant_care_ai.api.main:app --reload")
        sys.exit(1)

    # Test predict endpoint
    image_path = args.image
    if image_path is None:
        print("\nNo image specified. Looking for test images...")
        image_path = find_test_image()

    if image_path is None:
        print("✗ No test images found. Please specify --image path/to/image.jpg")
        sys.exit(1)

    success = test_predict(image_path, args.url, args.top_k)

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
