"""Unit and Integration Tests for VisionForge Experiment Tracking, Lineage, and Reproducibility."""

from fastapi.testclient import TestClient

from visionforge.experiments.fingerprint import (
    calculate_sha256,
    create_artifact_reference,
    create_dataset_fingerprint,
)
from visionforge.experiments.schemas import (
    ExperimentStatus,
)
from visionforge.experiments.service import (
    ExperimentService,
    capture_environment_snapshot,
)
from visionforge.main import app

client = TestClient(app)


# ─── Fingerprint & Checksum Tests ───────────────────────────────────


def test_calculate_sha256_file(tmp_path):
    """Test SHA-256 calculation over dummy artifact file."""
    dummy_file = tmp_path / "test_artifact.bin"
    dummy_file.write_bytes(b"VISIONFORGE_TEST_DATA_HASH_123")

    sha = calculate_sha256(dummy_file)
    assert len(sha) == 64
    assert sha != "file_not_found"


def test_dataset_fingerprint():
    """Test dataset manifest SHA-256 fingerprinting."""
    manifest = {
        "dataset_id": "safety_v2",
        "version": "v2.0",
        "num_samples": 500,
        "class_names": ["helmet", "head", "person"],
    }
    fp = create_dataset_fingerprint("safety_v2", "v2.0", manifest, "prep_12")

    assert fp.dataset_id == "safety_v2"
    assert fp.version == "v2.0"
    assert len(fp.manifest_sha256) == 64
    assert len(fp.fingerprint_hash) == 64


def test_create_artifact_reference(tmp_path):
    """Test artifact reference creation with size and checksum."""
    art_path = tmp_path / "weights.pt"
    art_path.write_bytes(b"VF_CHECKPOINT_BYTES")

    art_ref = create_artifact_reference("art_1", "checkpoint", "best.pt", art_path)
    assert art_ref.artifact_id == "art_1"
    assert art_ref.size_bytes > 0
    assert art_ref.sha256_checksum is not None


# ─── Environment Snapshot Tests ─────────────────────────────────────


def test_capture_environment_snapshot():
    """Verify runtime environment and Git commit SHA capture."""
    env = capture_environment_snapshot()
    assert env.python_version != ""
    assert env.os_platform != ""
    assert env.git_commit_sha != ""


# ─── Service Lifecycle & Lineage Tests ───────────────────────────────


def test_experiment_service_lifecycle():
    """Test experiment creation, attaching components, lineage graph, and timeline generation."""
    service = ExperimentService()

    # 1. Create
    exp = service.create_experiment(
        name="Test Safety Baseline",
        purpose="Test safety helmet detection baseline",
        dataset_id="safety_v2",
        dataset_version="v2.0",
        preparation_id="prep_12",
        random_seed=42,
    )

    assert exp.experiment_id.startswith("exp_")
    assert exp.status == ExperimentStatus.DRAFT
    assert exp.randomness.random_seed == 42
    assert exp.dataset_fingerprint is not None

    # 2. Attach Components
    updated_exp = service.attach_component(
        exp_id=exp.experiment_id,
        training_run_id="run_101",
        model_id="yolo11s_safety",
        evaluation_id="eval_202",
        benchmark_id="bm_303",
        inference_id="inf_404",
        training_config={"epochs": 50, "batch": 16, "imgsz": 640},
    )

    assert updated_exp.status == ExperimentStatus.COMPLETED
    assert "run_101" in updated_exp.training_run_ids
    assert "yolo11s_safety" in updated_exp.model_ids
    assert updated_exp.training_config_snapshot["epochs"] == 50

    # 3. Lineage Graph
    graph = service.get_lineage_graph(exp.experiment_id)
    assert len(graph.nodes) >= 6
    assert len(graph.edges) >= 5
    node_types = [n.type for n in graph.nodes]
    assert "dataset" in node_types
    assert "training_run" in node_types
    assert "model" in node_types

    # 4. Timeline
    events = service.get_timeline(exp.experiment_id)
    assert len(events) >= 3
    assert events[0].event_type == "EXPERIMENT_CREATED"


def test_experiment_comparison_and_reproducibility():
    """Test experiment side-by-side comparison and reproducibility audit."""
    service = ExperimentService()
    exp_a = service.create_experiment(name="Exp A", dataset_id="safety_v2")
    exp_b = service.create_experiment(name="Exp B", dataset_id="safety_v2")

    service.attach_component(exp_a.experiment_id, training_config={"lr": 0.01, "batch": 16})
    service.attach_component(exp_b.experiment_id, training_config={"lr": 0.001, "batch": 16})

    # Compare
    cmp_res = service.compare_experiments(exp_a.experiment_id, exp_b.experiment_id)
    assert cmp_res.experiment_a_id == exp_a.experiment_id
    assert "lr" in cmp_res.config_diff

    # Validate Reproducibility
    report = service.validate_reproducibility(exp_a.experiment_id)
    assert report.experiment_id == exp_a.experiment_id
    assert len(report.checks_passed) > 0

    # Reproduce Run
    rep_exp = service.reproduce_experiment(exp_a.experiment_id, "Reproduction A")
    assert rep_exp.parent_experiment_id == exp_a.experiment_id
    assert rep_exp.dataset_id == exp_a.dataset_id


# ─── REST API Endpoint Tests ───────────────────────────────────────


def test_api_create_and_get_experiment():
    """Test POST /api/v1/experiments and GET /api/v1/experiments/{id}."""
    response = client.post(
        "/api/v1/experiments",
        json={
            "name": "API Experiment Baseline",
            "purpose": "API integration test",
            "tags": ["test", "api"],
            "dataset_id": "safety_v2",
            "dataset_version": "v2.0",
        },
    )
    assert response.status_code == 201
    data = response.json()
    exp_id = data["experiment_id"]
    assert data["name"] == "API Experiment Baseline"

    # Get details
    get_res = client.get(f"/api/v1/experiments/{exp_id}")
    assert get_res.status_code == 200
    assert get_res.json()["experiment_id"] == exp_id


def test_api_list_experiments():
    """Test GET /api/v1/experiments."""
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_experiment_lineage_and_timeline():
    """Test GET lineage, timeline, and report endpoints."""
    # Create experiment
    res = client.post(
        "/api/v1/experiments",
        json={"name": "Lineage API Test", "dataset_id": "safety_v2"},
    )
    exp_id = res.json()["experiment_id"]

    # Lineage
    lin_res = client.get(f"/api/v1/experiments/{exp_id}/lineage")
    assert lin_res.status_code == 200
    assert "nodes" in lin_res.json()

    # Timeline
    time_res = client.get(f"/api/v1/experiments/{exp_id}/timeline")
    assert time_res.status_code == 200
    assert isinstance(time_res.json(), list)

    # Report
    rpt_res = client.get(f"/api/v1/experiments/{exp_id}/report")
    assert rpt_res.status_code == 200
    assert "report_md" in rpt_res.json()
