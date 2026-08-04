"""Test health check endpoint."""

from fastapi.testclient import TestClient

from visionforge.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /api/v1/health returns HTTP 200 with standardized envelope structure."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    json_data = response.json()

    assert json_data["success"] is True
    assert json_data["error"] is None
    assert "timestamp" in json_data["meta"]

    data = json_data["data"]
    assert data["status"] == "ok"
    assert data["service"] == "visionforge-backend"
    assert "version" in data
    assert "environment" in data
    assert "uptime_seconds" in data
    assert "ai_core" in data
    assert "optimal_device" in data["ai_core"]
    assert "installed_models" in data["ai_core"]
    assert "available_storage" in data["ai_core"]
    assert "model_manager_status" in data["ai_core"]
    assert data["ai_core"]["status"] == "ready"
