"""Unit and Integration Tests for Active Learning Retraining Loop & Performance Verdict Engine."""

import pytest
from fastapi.testclient import TestClient

from visionforge.active_learning.loop import (
    ActiveLearningLoopError,
    compute_metric_delta,
    execute_active_learning_loop_iteration,
)
from visionforge.active_learning.schemas import (
    ImprovementVerdict,
    ReviewStatus,
)
from visionforge.active_learning.service import (
    ActiveLearningService,
)
from visionforge.main import app

client = TestClient(app)


def test_compute_metric_delta():
    """Verify metric delta calculation logic."""
    md = compute_metric_delta(0.80, 0.88)
    assert md.baseline_val == 0.80
    assert md.retrained_val == 0.88
    assert md.delta == 0.08
    assert md.percent_change == 10.0


def test_execute_active_learning_loop_iteration():
    """Verify end-to-end active learning loop iteration and verdict determination."""
    service = ActiveLearningService()
    run = service.create_run(
        dataset_id="safety_v2",
        model_id="yolo11s.pt",
        top_k=5,
    )

    # Accept first 2 samples
    run.selected_samples[0].review_status = ReviewStatus.ACCEPTED
    run.selected_samples[1].review_status = ReviewStatus.ACCEPTED
    service.save_to_disk()

    iteration = execute_active_learning_loop_iteration(run, new_version_tag="v2.1")
    assert iteration.iteration_id.startswith("iter_")
    assert iteration.baseline_dataset_id == "safety_v2"
    assert iteration.new_dataset_version == "v2.1"
    assert iteration.reviewed_samples_count == 2
    assert iteration.verdict == ImprovementVerdict.IMPROVED
    assert iteration.map50_delta.delta > 0.0
    assert "PERFORMANCE IMPROVED" in iteration.verdict_summary


def test_loop_iteration_requires_accepted_samples():
    """Verify exception raised when no human review decisions have been accepted."""
    service = ActiveLearningService()
    run = service.create_run(
        dataset_id="safety_v2",
        model_id="yolo11s.pt",
        top_k=5,
    )
    # All samples remain UNREVIEWED

    with pytest.raises(ActiveLearningLoopError) as exc_info:
        execute_active_learning_loop_iteration(run)

    assert "contains NO accepted human-reviewed samples" in str(exc_info.value)


def test_service_execute_loop_persistence():
    """Verify service execute_loop, get_iteration, and list_iterations."""
    service = ActiveLearningService()
    run = service.create_run(
        dataset_id="safety_v2",
        model_id="yolo11s.pt",
        top_k=5,
    )
    run.selected_samples[0].review_status = ReviewStatus.ACCEPTED
    service.save_to_disk()

    iteration = service.execute_loop(run.run_id, "v2.1")
    assert iteration.iteration_id in [i.iteration_id for i in service.list_iterations()]

    fetched = service.get_iteration(iteration.iteration_id)
    assert fetched.iteration_id == iteration.iteration_id
    assert fetched.verdict == ImprovementVerdict.IMPROVED


def test_api_active_learning_loop_endpoints():
    """Test POST /api/v1/active-learning/loop and GET /api/v1/active-learning/iterations."""
    # 1. Create run
    res_run = client.post(
        "/api/v1/active-learning/runs",
        json={
            "dataset_id": "safety_v2",
            "model_id": "yolo11s.pt",
            "top_k": 5,
        },
    )
    run_id = res_run.json()["run_id"]
    sample_id = res_run.json()["selected_samples"][0]["image_id"]

    # 2. Accept sample
    client.post(
        "/api/v1/active-learning/review",
        json={
            "run_id": run_id,
            "image_id": sample_id,
            "status": "ACCEPTED",
            "notes": "Accepted sample for loop execution test",
        },
    )

    # 3. Execute Loop via API
    res_loop = client.post(
        "/api/v1/active-learning/loop",
        json={
            "baseline_dataset_id": "safety_v2",
            "baseline_model_id": "yolo11s.pt",
            "active_learning_run_id": run_id,
            "new_version_tag": "v2.1",
        },
    )
    assert res_loop.status_code == 201
    iter_data = res_loop.json()
    iter_id = iter_data["iteration_id"]
    assert iter_data["verdict"] == "IMPROVED"
    assert iter_data["map50_delta"]["delta"] > 0.0

    # 4. List Iterations via API
    res_list = client.get("/api/v1/active-learning/iterations")
    assert res_list.status_code == 200
    assert isinstance(res_list.json(), list)

    # 5. Get Iteration Detail via API
    res_get = client.get(f"/api/v1/active-learning/iterations/{iter_id}")
    assert res_get.status_code == 200
    assert res_get.json()["iteration_id"] == iter_id
