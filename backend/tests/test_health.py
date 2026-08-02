"""Test health check endpoint."""

from fastapi.testclient import TestClient

from visionforge.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /api/v1/health returns HTTP 200 with expected structure."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "visionforge-backend"
    assert "version" in data
