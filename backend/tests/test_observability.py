"""Automated Test Suite for VisionForge Observability, Health Probes, Metrics, and Error Handling."""

import pytest
from fastapi.testclient import TestClient

from visionforge.core.exceptions import (
    DatasetNotFoundException,
    DependencyUnavailableException,
    JobNotFoundException,
    ModelNotFoundException,
    TrainingJobFailedException,
)
from visionforge.core.telemetry import JobStatus, get_metrics_collector
from visionforge.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_liveness_health_endpoint(client):
    """Verify /health and /api/v1/health return healthy status and subsystem breakdown."""
    # Root health probe
    res_root = client.get("/health")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["status"] == "ok"
    assert "version" in data_root

    # API v1 detailed health
    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    data_v1 = res_v1.json()["data"]
    assert data_v1["status"] == "ok"
    assert "subsystems" in data_v1
    assert data_v1["subsystems"]["api"] == "healthy"
    assert data_v1["subsystems"]["storage"] in ("healthy", "degraded")
    assert data_v1["subsystems"]["job_queue"] == "healthy"
    assert data_v1["subsystems"]["model_registry"] in ("healthy", "degraded")


def test_readiness_probe_endpoint(client):
    """Verify /ready and /api/v1/ready return operational readiness."""
    res_root = client.get("/ready")
    assert res_root.status_code == 200
    assert res_root.json()["ready"] is True

    res_v1 = client.get("/api/v1/ready")
    assert res_v1.status_code == 200
    data = res_v1.json()["data"]
    assert data["ready"] is True
    assert data["status"] == "ready"
    assert "checks" in data
    assert data["checks"]["storage_writable"] is True


def test_dependency_health_matrix(client):
    """Verify /api/v1/health/dependencies reports core and optional dependencies with graceful degradation."""
    res = client.get("/api/v1/health/dependencies")
    assert res.status_code == 200
    report = res.json()["data"]

    assert report["overall_status"] in ("healthy", "degraded")
    assert "dependencies" in report
    deps = report["dependencies"]

    # Core dependencies must exist
    assert "storage" in deps
    assert "model_registry" in deps
    assert "job_queue" in deps
    assert "visual_memory" in deps

    # Optional integrations must be reported as disabled or healthy without crashing
    assert "database" in deps
    assert deps["database"]["status"] in ("healthy", "disabled")
    assert "redis" in deps
    assert deps["redis"]["status"] in ("healthy", "disabled")
    assert "qdrant" in deps
    assert deps["qdrant"]["status"] in ("healthy", "disabled")
    assert "mlflow" in deps
    assert deps["mlflow"]["status"] in ("healthy", "disabled")


def test_request_id_correlation_and_timing(client):
    """Verify X-Request-ID is generated or preserved and X-Process-Time is attached."""
    # Test auto-generated request ID
    res = client.get("/api/v1/system/info")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    assert "X-Process-Time" in res.headers

    # Test custom request ID header propagation
    custom_id = "req_custom_tracer_999"
    res_custom = client.get("/api/v1/system/info", headers={"X-Request-ID": custom_id})
    assert res_custom.status_code == 200
    assert res_custom.headers["X-Request-ID"] == custom_id


def test_domain_exception_codes_and_responses():
    """Verify domain exceptions format standardized error codes and HTTP statuses."""
    ds_exc = DatasetNotFoundException("dataset_nonexistent")
    assert ds_exc.code == "DATASET_NOT_FOUND"
    assert ds_exc.status_code == 404

    mod_exc = ModelNotFoundException("model_nonexistent")
    assert mod_exc.code == "MODEL_NOT_FOUND"
    assert mod_exc.status_code == 404

    job_exc = JobNotFoundException("job_nonexistent")
    assert job_exc.code == "JOB_NOT_FOUND"
    assert job_exc.status_code == 404

    train_exc = TrainingJobFailedException("CUDA out of memory")
    assert train_exc.code == "TRAINING_JOB_FAILED"
    assert train_exc.status_code == 500

    dep_exc = DependencyUnavailableException("redis")
    assert dep_exc.code == "DEPENDENCY_UNAVAILABLE"
    assert dep_exc.status_code == 503


def test_job_observability_lifecycle(client):
    """Verify background job registration, progress, completion, and failure tracking."""
    collector = get_metrics_collector()
    test_job_id = "test_job_obs_001"

    # 1. Register Job
    job = collector.register_job(
        job_id=test_job_id,
        job_type="training",
        name="Test Training Job",
        request_id="req_test_001",
        metadata={"epochs": 2},
    )
    assert job.status == JobStatus.QUEUED
    assert job.progress_pct == 0.0

    # 2. Start Job
    collector.start_job(test_job_id)
    assert collector.get_job(test_job_id).status == JobStatus.RUNNING
    assert collector.get_job(test_job_id).started_at is not None

    # 3. Update Progress
    collector.update_job_progress(test_job_id, 45.0)
    assert collector.get_job(test_job_id).progress_pct == 45.0

    # 4. Complete Job
    collector.complete_job(test_job_id, metadata={"accuracy": 0.95})
    completed_job = collector.get_job(test_job_id)
    assert completed_job.status == JobStatus.COMPLETED
    assert completed_job.progress_pct == 100.0
    assert completed_job.duration_seconds is not None
    assert completed_job.metadata["accuracy"] == 0.95

    # 5. Query via API
    res = client.get("/api/v1/system/jobs")
    assert res.status_code == 200
    jobs_list = res.json()["data"]
    matching = [j for j in jobs_list if j["job_id"] == test_job_id]
    assert len(matching) == 1
    assert matching[0]["status"] == "COMPLETED"

    # Query single job API
    res_single = client.get(f"/api/v1/system/jobs/{test_job_id}")
    assert res_single.status_code == 200
    assert res_single.json()["data"]["job_id"] == test_job_id


def test_job_failure_tracking_and_errors_api(client):
    """Verify job failure records diagnostic error and is retrievable via /system/errors."""
    collector = get_metrics_collector()
    fail_job_id = "test_job_fail_002"

    collector.register_job(
        job_id=fail_job_id,
        job_type="evaluation",
        name="Failed Eval Job",
    )
    collector.start_job(fail_job_id)
    collector.fail_job(
        job_id=fail_job_id,
        error_code="INVALID_ANNOTATIONS",
        error_summary="Ground truth bounding box coordinates out of bounds",
        details={"bad_box": [1.2, 0.5, 0.4, 0.4]},
    )

    failed_job = collector.get_job(fail_job_id)
    assert failed_job.status == JobStatus.FAILED
    assert failed_job.error_code == "INVALID_ANNOTATIONS"

    # Verify errors API
    res_errors = client.get("/api/v1/system/errors")
    assert res_errors.status_code == 200
    failures = res_errors.json()["data"]
    assert any(f["job_id"] == fail_job_id for f in failures)


def test_prometheus_metrics_exposition(client):
    """Verify /metrics and /api/v1/system/metrics export valid Prometheus metrics."""
    res_root = client.get("/metrics")
    assert res_root.status_code == 200
    assert "text/plain" in res_root.headers["content-type"]
    root_text = res_root.text
    assert "visionforge_uptime_seconds" in root_text
    assert "visionforge_http_requests_total" in root_text
    assert "visionforge_jobs_active" in root_text
    assert "visionforge_cv_inferences_total" in root_text

    res_v1 = client.get("/api/v1/system/metrics")
    assert res_v1.status_code == 200
    assert "visionforge_uptime_seconds" in res_v1.text


def test_cv_operational_telemetry(client):
    """Verify CV inference and search metrics are recorded and exposed in diagnostics."""
    collector = get_metrics_collector()
    collector.record_inference(model_name="yolo11n.pt", duration_ms=12.5)
    collector.record_search(duration_ms=4.2, result_count=5)
    collector.record_video_frames(frame_count=30, duration_ms=20.0)

    res = client.get("/api/v1/system/diagnostics")
    assert res.status_code == 200
    diag = res.json()["data"]

    assert "cv_metrics" in diag
    cv = diag["cv_metrics"]
    assert cv["total_inferences"] >= 1
    assert cv["total_search_queries"] >= 1
    assert cv["total_video_frames_processed"] >= 30
