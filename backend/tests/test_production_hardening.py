"""Deterministic Unit & Integration Tests for Production Hardening & System Diagnostics."""

from fastapi.testclient import TestClient

from visionforge.core.pagination import paginate_sequence
from visionforge.core.telemetry import MetricsCollector
from visionforge.main import app

client = TestClient(app)


def test_request_id_middleware_and_headers():
    """Verify X-Request-ID is generated and returned on all API requests."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert "x-request-id" in res.headers
    assert "x-process-time" in res.headers
    assert len(res.headers["x-request-id"]) > 0

    # Custom request ID forwarding
    custom_id = "req_custom_test_12345"
    res_custom = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert res_custom.headers["x-request-id"] == custom_id


def test_standardized_error_format_and_request_id():
    """Verify error responses follow the standardized error envelope with request_id."""
    res = client.get("/api/v1/datasets/non_existent_dataset_id_999999")
    assert res.status_code == 404
    data = res.json()
    assert data["success"] is False
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert "meta" in data
    assert "request_id" in data["meta"]
    assert len(data["meta"]["request_id"]) > 0


def test_generic_pagination_utility():
    """Verify pagination utility handles empty, standard, and out-of-bounds page requests."""
    items = [f"item_{i}" for i in range(55)]

    # Page 1 (20 items)
    p1 = paginate_sequence(items, page=1, page_size=20)
    assert p1.total == 55
    assert len(p1.items) == 20
    assert p1.page == 1
    assert p1.total_pages == 3
    assert p1.has_next is True
    assert p1.has_prev is False

    # Page 3 (15 items)
    p3 = paginate_sequence(items, page=3, page_size=20)
    assert len(p3.items) == 15
    assert p3.has_next is False
    assert p3.has_prev is True

    # Empty list
    p_empty = paginate_sequence([], page=1, page_size=20)
    assert p_empty.total == 0
    assert p_empty.total_pages == 1
    assert len(p_empty.items) == 0


def test_metrics_collector_telemetry():
    """Verify thread-safe operational metrics collection and failure logging."""
    collector = MetricsCollector()
    collector.record_request(duration_ms=15.5, is_error=False)
    collector.record_request(duration_ms=25.0, is_error=False)
    collector.record_request(duration_ms=50.0, is_error=True)

    collector.register_job("job_train_01", is_running=False)
    collector.register_job("job_video_01", is_running=True)

    collector.record_failure(
        service="training_engine",
        error_code="CUDA_OOM",
        message="Out of memory on GPU device 0",
        job_id="job_train_01",
    )

    snapshot = collector.get_snapshot()
    assert snapshot.total_requests == 3
    assert snapshot.total_errors >= 1
    assert snapshot.active_jobs_count == 1
    assert snapshot.queued_jobs_count == 1
    assert len(snapshot.recent_failures) >= 1
    assert snapshot.recent_failures[0].error_code == "CUDA_OOM"

    collector.complete_job("job_video_01")
    snapshot2 = collector.get_snapshot()
    assert snapshot2.active_jobs_count == 0


def test_health_check_subsystem_distinction():
    """Verify /health endpoint returns granular subsystem states."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["status"] == "ok"
    assert "subsystems" in data
    assert data["subsystems"]["api"] == "healthy"
    assert data["subsystems"]["storage"] == "healthy"
    assert data["subsystems"]["job_queue"] == "healthy"
    assert data["subsystems"]["model_registry"] == "healthy"


def test_system_diagnostics_api_endpoint():
    """Verify /system/diagnostics returns real operational metrics snapshot."""
    res = client.get("/api/v1/system/diagnostics")
    assert res.status_code == 200
    data = res.json()["data"]
    assert "total_requests" in data
    assert "avg_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "active_jobs_count" in data
    assert "recent_failures" in data
    assert isinstance(data["recent_failures"], list)


def test_security_secrets_redaction():
    """Verify secrets and auth tokens are not exposed in system info or error responses."""
    res_sys = client.get("/api/v1/system/info")
    assert res_sys.status_code == 200
    content = res_sys.text
    assert "api_key" not in content.lower()
    assert "secret" not in content.lower()
    assert "password" not in content.lower()
