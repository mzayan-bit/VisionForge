"""Integration test suite for Embeddings API endpoints."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from visionforge.main import app

client = TestClient(app)


def _create_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_get_embedding_model_info():
    """Verify GET /api/v1/embeddings/model-info endpoint."""
    response = client.get("/api/v1/embeddings/model-info")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["name"] == "siglip-base-patch16-224"
    assert data["dimension"] == 768


def test_generate_embedding_endpoint():
    """Verify POST /api/v1/embeddings/generate with valid image file upload."""
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("test_image.jpg", img_bytes, "image/jpeg")}

    response = client.post("/api/v1/embeddings/generate", files=files)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True

    data = json_data["data"]
    assert data["dimension"] == 768
    assert len(data["embedding"]) == 768
    assert data["model"] == "siglip-base-patch16-224"
    assert "vector_stats" in data


def test_generate_embedding_invalid_file():
    """Verify error handling for invalid file uploads."""
    files = {"file": ("test.txt", b"text data", "text/plain")}
    response = client.post("/api/v1/embeddings/generate", files=files)
    assert response.status_code == 400


def test_embedding_model_lifecycle_endpoints():
    """Verify model load and unload API triggers."""
    # Load
    load_res = client.post("/api/v1/embeddings/model/load?device=cpu")
    assert load_res.status_code == 200
    assert load_res.json()["data"]["status"] == "ready"

    # Unload
    unload_res = client.post("/api/v1/embeddings/model/unload")
    assert unload_res.status_code == 200
    assert unload_res.json()["data"]["status"] == "unloaded"
