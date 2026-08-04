"""Test system info and root endpoints."""

from fastapi.testclient import TestClient

from visionforge.main import app

client = TestClient(app)


def test_system_info_endpoint():
    """Verify /api/v1/system/info returns system runtime metadata envelope."""
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    json_data = response.json()

    assert json_data["success"] is True
    assert json_data["error"] is None

    data = json_data["data"]
    assert data["project"] == "VisionForge"
    assert data["status"] == "ready"
    assert "python_version" in data
    assert "platform" in data
    assert "total_routes" in data
    assert "ai_core" in data
    assert "vision_engine" in data
    assert isinstance(data["registered_endpoints"], list)
    assert "/api/v1/health" in data["registered_endpoints"]
    assert "/api/v1/system/info" in data["registered_endpoints"]


def test_root_endpoint():
    """Verify root / endpoint returns API info in unified response format."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()

    assert json_data["success"] is True
    data = json_data["data"]
    assert data["name"] == "VisionForge"
    assert data["health"] == "/api/v1/health"
    assert data["system"] == "/api/v1/system/info"
