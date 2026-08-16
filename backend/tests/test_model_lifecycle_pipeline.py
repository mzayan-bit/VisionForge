"""Unit and Integration Tests for VisionForge End-to-End Model Lifecycle Pipeline."""

from fastapi.testclient import TestClient

from visionforge.main import app
from visionforge.models.lifecycle_schemas import (
    CreatePipelineRequest,
    DeployModelRequest,
    PipelineStage,
    PipelineStatus,
    StageExecutionState,
)
from visionforge.models.lifecycle_service import get_model_lifecycle_service

client = TestClient(app)


def test_create_and_execute_full_pipeline():
    """Verify autonomous execution of all 9 stages in model lifecycle pipeline."""
    service = get_model_lifecycle_service()

    req = CreatePipelineRequest(
        name="Automated Test Pipeline",
        dataset_id="safety_v2",
        dataset_version="v1.0.0",
        base_model="yolo11s.pt",
        target_model_name="yolo11s_safety_test",
        epochs=50,
        batch_size=16,
        auto_advance=True,
    )

    pipeline = service.create_pipeline(req)

    assert pipeline.pipeline_id.startswith("pipe_")
    assert pipeline.status == PipelineStatus.COMPLETED
    assert pipeline.is_deployed is True
    assert pipeline.deployment_endpoint is not None
    assert len(pipeline.stages) == 9

    # Verify all 9 stages completed
    for st in PipelineStage:
        stage_record = pipeline.stages.get(st.value)
        assert stage_record is not None
        assert stage_record.status == StageExecutionState.COMPLETED
        assert stage_record.summary != ""
        assert len(stage_record.metrics) > 0


def test_stage_by_stage_stepwise_advancement():
    """Verify manual stepping through individual pipeline stages."""
    service = get_model_lifecycle_service()

    req = CreatePipelineRequest(
        name="Stepwise Test Pipeline",
        dataset_id="safety_v2",
        dataset_version="v1.0.0",
        base_model="yolo11s.pt",
        target_model_name="yolo11s_safety_stepwise",
        auto_advance=False,  # Paused at stage 1
    )

    pipeline = service.create_pipeline(req)
    assert pipeline.status == PipelineStatus.PENDING
    assert pipeline.current_stage == PipelineStage.DATASET_VERSION

    # Step 1 -> Step 2
    p2 = service.advance_pipeline(pipeline.pipeline_id)
    assert p2.current_stage == PipelineStage.TRAINING_CONFIG
    assert p2.stages[PipelineStage.TRAINING_CONFIG.value].status == StageExecutionState.COMPLETED

    # Step 2 -> Step 3
    p3 = service.advance_pipeline(pipeline.pipeline_id)
    assert p3.current_stage == PipelineStage.TRAINING_RUN
    assert p3.stages[PipelineStage.TRAINING_RUN.value].status == StageExecutionState.COMPLETED


def test_deploy_pipeline_model():
    """Verify deploying model checkpoint to active runtime."""
    service = get_model_lifecycle_service()

    req = CreatePipelineRequest(
        name="Deployment Test Pipeline",
        dataset_id="safety_v2",
        base_model="yolo11s.pt",
        target_model_name="yolo11s_safety_deploy",
        auto_advance=True,
    )
    pipeline = service.create_pipeline(req)

    deployed = service.deploy_pipeline_model(
        pipeline.pipeline_id,
        DeployModelRequest(environment="production", device="auto", warm_up_runs=5),
    )

    assert deployed.is_deployed is True
    assert "/api/v1/inference/yolo11s_safety_deploy" in deployed.deployment_endpoint
    stage_9 = deployed.stages[PipelineStage.DEPLOYMENT.value]
    assert stage_9.status == StageExecutionState.COMPLETED
    assert stage_9.metrics["environment"] == "production"


def test_pipeline_lineage_dag():
    """Verify 9-node DAG lineage graph generation."""
    service = get_model_lifecycle_service()

    req = CreatePipelineRequest(
        name="Lineage Test Pipeline",
        dataset_id="safety_v2",
        base_model="yolo11s.pt",
        target_model_name="yolo11s_safety_lineage",
        auto_advance=True,
    )
    pipeline = service.create_pipeline(req)

    nodes = service.get_pipeline_lineage(pipeline.pipeline_id)
    assert len(nodes) == 9

    # First node is dataset
    assert nodes[0].stage == PipelineStage.DATASET_VERSION
    assert nodes[0].parent_node_ids == []

    # Last node is deployment
    assert nodes[8].stage == PipelineStage.DEPLOYMENT
    assert len(nodes[8].parent_node_ids) == 1


def test_lifecycle_api_endpoints():
    """Verify REST API endpoints for Model Lifecycle Pipeline."""
    # 1. Create Pipeline
    res_create = client.post(
        "/api/v1/lifecycle/pipelines",
        json={
            "name": "API Test Pipeline",
            "dataset_id": "safety_v2",
            "dataset_version": "v1.0.0",
            "base_model": "yolo11s.pt",
            "target_model_name": "yolo11s_api_test",
            "epochs": 10,
            "auto_advance": True,
        },
    )
    assert res_create.status_code == 201
    pipe_data = res_create.json()["data"]
    pid = pipe_data["pipeline_id"]
    assert pipe_data["status"] == "COMPLETED"

    # 2. Get Pipeline
    res_get = client.get(f"/api/v1/lifecycle/pipelines/{pid}")
    assert res_get.status_code == 200
    assert res_get.json()["data"]["pipeline_id"] == pid

    # 3. List Pipelines
    res_list = client.get("/api/v1/lifecycle/pipelines")
    assert res_list.status_code == 200
    assert len(res_list.json()["data"]) >= 1

    # 4. Get Lineage
    res_lin = client.get(f"/api/v1/lifecycle/pipelines/{pid}/lineage")
    assert res_lin.status_code == 200
    assert len(res_lin.json()["data"]) == 9

    # 5. Deploy Pipeline Model
    res_dep = client.post(
        f"/api/v1/lifecycle/pipelines/{pid}/deploy",
        json={"environment": "production", "device": "auto", "warm_up_runs": 3},
    )
    assert res_dep.status_code == 200
    assert res_dep.json()["data"]["is_deployed"] is True
