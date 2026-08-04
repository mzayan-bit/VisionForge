"""Test model manager API endpoints."""


from fastapi.testclient import TestClient

from visionforge.main import app

client = TestClient(app)


def test_list_models_empty():
    """Verify listing models returns empty list when no models are installed."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["total"] == 0
    assert data["models"] == []


def test_get_manager_status():
    """Verify manager status endpoint."""
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["status"] == "ready"
    assert "installed_models" in data
    assert "storage" in data
    assert isinstance(data["storage"]["exists"], bool)


def test_get_storage_usage():
    """Verify storage stats endpoint."""
    response = client.get("/api/v1/models/storage")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert "root_directory" in data
    assert "total_size_bytes" in data
    assert "total_size_mb" in data
    assert "models_count" in data


def test_get_model_detail_not_found():
    """Verify looking up non-existent model returns 404."""
    response = client.get("/api/v1/models/fake-model")
    assert response.status_code == 404
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "MODEL_NOT_INSTALLED"


def test_validate_model_not_found():
    """Verify validating non-existent model returns gracefully with valid=False."""
    response = client.post("/api/v1/models/fake-model/validate")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["name"] == "fake-model"
    assert data["valid"] is False
    assert len(data["errors"]) > 0
    assert "not installed" in data["errors"][0]
