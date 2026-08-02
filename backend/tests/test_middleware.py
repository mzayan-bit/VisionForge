"""Test middleware components."""

from fastapi.testclient import TestClient

from visionforge.main import app

client = TestClient(app)


def test_request_tracing_middleware_headers():
    """Verify request tracing attaches X-Request-ID and X-Process-Time response headers."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers
    assert response.headers["X-Process-Time"].endswith("ms")


def test_custom_request_id_preservation():
    """Verify incoming X-Request-ID is preserved and echoed in response headers."""
    custom_id = "test-custom-id-12345"
    response = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id
