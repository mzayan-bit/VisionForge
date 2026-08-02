"""Test system info endpoint."""

from fastapi.testclient import TestClient

from visionforge.main import app

client = TestClient(app)


def test_system_info_endpoint():
    """Verify /api/v1/system/info returns system metadata."""
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "VisionForge Workbench"
    assert data["status"] == "ready"
    assert "python_version" in data
    assert "platform" in data


def test_root_endpoint():
    """Verify root / endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "VisionForge"
    assert data["health"] == "/api/v1/health"
