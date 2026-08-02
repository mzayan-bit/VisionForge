"""Test centralized exception handlers and error response envelope."""

from fastapi.testclient import TestClient

from visionforge.main import app

client = TestClient(app)


def test_404_not_found_error_format():
    """Verify non-existent routes return standardized 404 error envelope."""
    response = client.get("/api/v1/nonexistent-route")
    assert response.status_code == 404
    json_data = response.json()

    assert json_data["success"] is False
    assert json_data["data"] is None
    assert json_data["error"]["code"] == "NOT_FOUND"
    assert "Not Found" in json_data["error"]["message"]
    assert "timestamp" in json_data["meta"]
