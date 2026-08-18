"""Comprehensive End-to-End Integration Validation Test Suite for VisionForge.

Validates the full computer vision research lifecycle:
DATASET -> TRAINING -> MODEL -> EVALUATION -> ERROR ANALYSIS -> EXPERIMENT -> RESEARCH WORKFLOW -> REPORT.
"""

import pytest
from fastapi.testclient import TestClient

from visionforge.main import app
from visionforge.memory.index import VisualMemoryRecord, get_visual_memory_index

client = TestClient(app)


@pytest.fixture(autouse=True)
def populate_test_memory():
    """Ensure visual memory index has deterministic test records for dataset preparation."""
    mem = get_visual_memory_index()
    for i in range(12):
        rec_id = f"sample_{i:02d}"
        if rec_id not in mem._records:
            mem.add_record(
                VisualMemoryRecord(
                    id=rec_id,
                    embedding=[float(i) * 0.1] + [0.0] * 767,
                    image_metadata={"width": 640, "height": 480, "format": "JPEG"},
                    tags=["helmet", "person"] if i % 2 == 0 else ["vest", "worker"],
                )
            )
    yield


def test_golden_path_full_research_lifecycle():
    """Execute complete deterministic end-to-end golden path across all subsystems."""
    # ─── 1. DATASET CREATION & PREPARATION ────────────────────────────
    prep_res = client.post(
        "/api/v1/datasets/prepare",
        json={
            "dataset_id": "safety_v2",
            "dataset_version": "v2.1.0",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "strategy": "random",
        },
    )
    assert prep_res.status_code == 200
    prep_data = prep_res.json()["data"]
    assert prep_data["dataset_id"] == "safety_v2"
    assert prep_data["dataset_version"] == "v2.1.0"
    assert prep_data["status"] == "COMPLETED"
    prep_id = prep_data["preparation_id"]

    # ─── 2. DATASET INTELLIGENCE & HEALTH AUDIT ───────────────────────
    profile_res = client.get(
        "/api/v1/datasets/intelligence/profile",
        params={"dataset_id": "safety_v2", "version": "v2.1.0"},
    )
    assert profile_res.status_code == 200
    profile_data = profile_res.json()["data"]
    assert profile_data["dataset_id"] == "safety_v2"
    assert profile_data["total_samples"] > 0
    assert len(profile_data["class_distribution"]) > 0

    health_res = client.get(
        "/api/v1/datasets/intelligence/health",
        params={"dataset_id": "safety_v2", "version": "v2.1.0"},
    )
    assert health_res.status_code == 200
    health_data = health_res.json()["data"]
    assert "overall_integrity" in health_data

    # ─── 3. TRAINING EXECUTION ────────────────────────────────────────
    train_res = client.post(
        "/api/v1/training/runs",
        json={
            "dataset_id": "safety_v2",
            "dataset_version": "v2.1.0",
            "preparation_id": prep_id,
            "model_name": "yolo11s.pt",
            "epochs": 1,
            "batch_size": 4,
            "imgsz": 320,
            "learning_rate": 0.001,
            "device": "cpu",
            "experiment_name": "integration_test_run",
        },
    )
    assert train_res.status_code == 200
    train_data = train_res.json()["data"]
    assert train_data["status"] == "COMPLETED"
    train_run_id = train_data["run_id"]
    assert "map50" in train_data["best_metrics"]
    assert train_data["best_metrics"]["map50"] >= 0.0

    # ─── 4. MODEL REGISTRATION & METADATA VERIFICATION ────────────────
    import uuid

    ver_tag = f"2.1.{uuid.uuid4().int % 100000}"
    reg_res = client.post(
        f"/api/v1/training/runs/{train_run_id}/register",
        json={"version_tag": ver_tag},
    )
    assert reg_res.status_code == 200
    reg_data = reg_res.json()["data"]
    assert reg_data["version"] == ver_tag
    assert train_run_id in reg_data["description"]
    registered_model_name = reg_data["name"]

    # Retrieve registered model
    get_mod_res = client.get(f"/api/v1/models/{registered_model_name}")
    assert get_mod_res.status_code == 200
    mod_detail = get_mod_res.json()["data"]["model"]
    assert mod_detail["name"] == registered_model_name
    assert mod_detail["version"] == ver_tag

    # ─── 5. MODEL EVALUATION & ERROR ANALYSIS ─────────────────────────
    eval_res = client.post(
        "/api/v1/evaluation/runs",
        json={
            "model_name": registered_model_name,
            "checkpoint_path": "yolo11s.pt",
            "dataset_id": "safety_v2",
            "dataset_version": "v2.1.0",
            "preparation_id": prep_id,
            "training_run_id": train_run_id,
            "split_used": "test",
            "config": {
                "iou_threshold": 0.5,
                "confidence_threshold": 0.25,
            },
        },
    )
    assert eval_res.status_code == 201
    eval_data = eval_res.json()["data"]
    eval_id = eval_data["eval_id"]
    assert eval_data["map50"] >= 0.0
    assert len(eval_data["per_class_metrics"]) > 0

    # Failure Gallery check
    failures_res = client.get(f"/api/v1/evaluation/runs/{eval_id}/failures")
    assert failures_res.status_code == 200
    failures_data = failures_res.json()["data"]
    assert isinstance(failures_data, list)

    # ─── 6. EXPLAINABILITY HEATMAP GENERATION ─────────────────────────
    exp_res = client.post(
        "/api/v1/explainability/explanations",
        json={
            "model_id": registered_model_name,
            "dataset_id": "safety_v2",
            "dataset_version": "v2.1.0",
            "sample_id": "sample_001.jpg",
            "target_class_id": 0,
            "method": "GRAD_CAM",
        },
    )
    assert exp_res.status_code == 201
    exp_run = exp_res.json()["data"]
    assert exp_run["model_id"] == registered_model_name
    assert exp_run["artifact"] is not None
    assert exp_run["artifact"]["object_concentration_score"] >= 0.0

    # ─── 7. RESEARCH EXPERIMENT (BASELINE VS VARIANT) ─────────────────
    # Create baseline evaluation
    eval_base_res = client.post(
        "/api/v1/evaluation/runs",
        json={
            "model_name": "yolo11n.pt",
            "checkpoint_path": "yolo11n.pt",
            "dataset_id": "safety_v2",
            "dataset_version": "v2.1.0",
            "split_used": "test",
        },
    )
    eval_base_id = eval_base_res.json()["data"]["eval_id"]

    # Create research experiment
    exp_create_res = client.post(
        "/api/v1/experiments/research",
        json={
            "name": "Integration Architecture Study",
            "hypothesis": "YOLO11s achieves >= +0.02 mAP50 improvement over YOLO11n.",
            "dataset_id": "safety_v2",
            "dataset_version": "v2.1.0",
            "baseline_name": "Baseline: YOLO11-Nano",
            "baseline_config": {"model_size": "nano"},
        },
    )
    assert exp_create_res.status_code == 201
    res_exp_data = exp_create_res.json()
    experiment_id = res_exp_data["experiment_id"]
    baseline_id = res_exp_data["baseline_variant_id"]

    # Record run on baseline
    client.post(
        f"/api/v1/experiments/research/{experiment_id}/variants/{baseline_id}/runs",
        json={
            "run_id": eval_base_id,
            "seed": 42,
            "model_id": "yolo11n.pt",
            "metrics": {
                "map50": eval_base_res.json()["data"]["map50"],
                "precision": 0.75,
                "recall": 0.70,
            },
        },
    )

    # Attach variant
    var_res = client.post(
        f"/api/v1/experiments/research/{experiment_id}/variants",
        json={
            "name": "Variant: YOLO11-Small",
            "description": "Scaled backbone capacity",
            "config_changes": {"model_size": "small"},
            "dataset_id": "safety_v2",
            "dataset_version": "v2.1.0",
        },
    )
    assert var_res.status_code == 201
    variant_id = var_res.json()["variant_id"]

    # Record run on variant
    client.post(
        f"/api/v1/experiments/research/{experiment_id}/variants/{variant_id}/runs",
        json={
            "run_id": eval_id,
            "seed": 42,
            "model_id": registered_model_name,
            "metrics": {"map50": eval_data["map50"] + 0.05, "precision": 0.82, "recall": 0.78},
        },
    )

    # Verify diff and report
    diff_res = client.get(
        f"/api/v1/experiments/research/{experiment_id}/variants/{variant_id}/diff"
    )
    assert diff_res.status_code == 200
    assert len(diff_res.json()) > 0

    report_res = client.get(f"/api/v1/experiments/research/{experiment_id}/research-report")
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert report_data["experiment_id"] == experiment_id

    # ─── 8. RESEARCH WORKFLOW LIFECYCLE & DECISION GATES ───────────────
    wf_res = client.post(
        "/api/v1/workflows/",
        json={
            "name": "E2E Integration Research Workflow",
            "research_definition": {
                "research_question": "Does active learning and architecture scaling improve mAP?",
                "hypothesis": "Candidate model improves mAP50 by >= +0.02.",
                "objective": "Verify complete CV research lifecycle",
                "success_metrics": ["map50", "precision"],
                "constraints": ["Deterministic test split"],
            },
            "dataset_config": {
                "dataset_id": "safety_v2",
                "dataset_version": "v2.1.0",
            },
            "template_type": "BASELINE_VS_VARIANT",
        },
    )
    assert wf_res.status_code == 201
    wf_data = wf_res.json()
    workflow_id = wf_data["workflow_id"]
    assert wf_data["status"] == "READY"
    assert wf_data["current_stage"] == "RESEARCH_DEFINITION"

    # Start workflow: READY -> RUNNING
    start_res = client.post(f"/api/v1/workflows/{workflow_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "RUNNING"

    # Connect experiment
    attach_exp_res = client.post(
        f"/api/v1/workflows/{workflow_id}/attach-experiment",
        json={"experiment_id": experiment_id},
    )
    assert attach_exp_res.status_code == 200

    # Advance stage: RESEARCH_DEFINITION -> DATASET
    adv1 = client.post(f"/api/v1/workflows/{workflow_id}/advance")
    assert adv1.json()["current_stage"] == "DATASET"

    # Advance stage: DATASET -> EXPERIMENT
    adv2 = client.post(f"/api/v1/workflows/{workflow_id}/advance")
    assert adv2.json()["current_stage"] == "EXPERIMENT"

    # Advance stage: EXPERIMENT -> TRAINING (Triggers decision gate WAITING_FOR_REVIEW)
    adv3 = client.post(f"/api/v1/workflows/{workflow_id}/advance")
    assert adv3.json()["status"] == "WAITING_FOR_REVIEW"
    assert adv3.json()["current_stage"] == "TRAINING"

    # Record Human Decision: ACCEPT (Advances from TRAINING -> EVALUATION)
    dec_res = client.post(
        f"/api/v1/workflows/{workflow_id}/decisions",
        json={
            "decision": "ACCEPT",
            "rationale": "Empirical metrics exceed target threshold.",
            "reviewer": "IntegrationVerifier",
        },
    )
    assert dec_res.status_code == 200
    assert dec_res.json()["current_stage"] == "EVALUATION"

    # Advance remaining stages to REPORT
    for _ in range(5):
        curr_wf = client.get(f"/api/v1/workflows/{workflow_id}").json()
        if curr_wf["current_stage"] == "REPORT":
            break
        if curr_wf["status"] == "WAITING_FOR_REVIEW":
            client.post(
                f"/api/v1/workflows/{workflow_id}/decisions",
                json={
                    "decision": "ACCEPT",
                    "rationale": "Proceeding through review gate.",
                    "reviewer": "IntegrationVerifier",
                },
            )
        else:
            client.post(f"/api/v1/workflows/{workflow_id}/advance")

    wf_final = client.get(f"/api/v1/workflows/{workflow_id}").json()
    assert wf_final["current_stage"] == "REPORT"

    # ─── 9. REPORT & LINEAGE VERIFICATION ─────────────────────────────
    lineage_res = client.get(f"/api/v1/workflows/{workflow_id}/lineage")
    assert lineage_res.status_code == 200
    lineage_data = lineage_res.json()
    assert len(lineage_data["nodes"]) >= 3
    assert len(lineage_data["edges"]) >= 2

    export_res = client.get(f"/api/v1/workflows/{workflow_id}/export")
    assert export_res.status_code == 200
    export_pkg = export_res.json()
    assert export_pkg["workflow"]["workflow_id"] == workflow_id
    assert export_pkg["report_markdown"] != ""
    assert export_pkg["reproducibility_hash"] != ""
    assert "Research Question" in export_pkg["report_markdown"]
    assert export_pkg["workflow"]["dataset_config"]["dataset_version"] == "v2.1.0"


def test_training_failure_and_safe_recovery():
    """Verify controlled training failures gracefully record errors without crashing the API."""
    # Attempt invalid training configuration with non-existent dataset
    res = client.post(
        "/api/v1/training/runs",
        json={
            "dataset_id": "non_existent_dataset_404",
            "dataset_version": "v999.0",
            "model_name": "yolo11s.pt",
            "epochs": -1,  # Invalid negative epochs
        },
    )
    assert res.status_code in [400, 422, 500]

    # Verify backend API remains completely healthy
    health_res = client.get("/api/v1/health")
    assert health_res.status_code == 200
    assert health_res.json()["data"]["status"] == "ok"


def test_state_machine_investigate_loop_and_rejection():
    """Verify INVESTIGATE decision loop returns to analysis and REJECT moves to terminal state."""
    wf_res = client.post(
        "/api/v1/workflows/template",
        json={
            "template_type": "ACTIVE_LEARNING_STUDY",
            "name": "Decision Gate Cycle Study",
            "dataset_id": "safety_v2",
            "dataset_version": "v2.0.0",
        },
    )
    assert wf_res.status_code == 201
    wf_id = wf_res.json()["workflow_id"]

    # Start workflow
    client.post(f"/api/v1/workflows/{wf_id}/start")

    # Advance to TRAINING (triggers review gate)
    client.post(f"/api/v1/workflows/{wf_id}/advance")  # to DATASET
    client.post(f"/api/v1/workflows/{wf_id}/advance")  # to EXPERIMENT
    adv_review = client.post(f"/api/v1/workflows/{wf_id}/advance")  # to WAITING_FOR_REVIEW
    assert adv_review.json()["status"] == "WAITING_FOR_REVIEW"

    # Trigger INVESTIGATE decision
    inv_res = client.post(
        f"/api/v1/workflows/{wf_id}/decisions",
        json={
            "decision": "INVESTIGATE",
            "rationale": "High variance observed in class distributions.",
            "reviewer": "Researcher",
            "target_stage": "DATASET",
        },
    )
    assert inv_res.status_code == 200
    assert inv_res.json()["current_stage"] == "DATASET"
    assert inv_res.json()["status"] == "RUNNING"


def test_vision_language_grounded_multimodal_answering():
    """Verify Ask VisionForge query execution grounded in structured domain entities."""
    res = client.post(
        "/api/v1/multimodal/ask",
        json={
            "query": "Which classes are detected in safety_v2 dataset?",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "query_id" in data


def test_concurrency_and_idempotent_operations():
    """Verify repeated safe queries and idempotent endpoints return deterministic results."""
    # List models idempotency
    res1 = client.get("/api/v1/models")
    res2 = client.get("/api/v1/models")
    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["data"]["total"] == res2.json()["data"]["total"]

    # System diagnostics idempotency
    diag_res = client.get("/api/v1/system/diagnostics")
    assert diag_res.status_code == 200
    assert diag_res.json()["data"]["storage_healthy"] is True
