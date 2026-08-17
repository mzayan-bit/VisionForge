"""Deterministic Unit & Lifecycle Tests for VisionForge End-to-End Research Workflows."""

import pytest
from fastapi.testclient import TestClient

from visionforge.main import app
from visionforge.workflows.schemas import (
    DatasetConfig,
    DecisionType,
    ResearchDefinition,
    WorkflowStage,
    WorkflowStatus,
    WorkflowTemplateType,
)
from visionforge.workflows.service import (
    InvalidStateTransitionError,
    ResearchWorkflowService,
)

client = TestClient(app)


def test_workflow_creation_and_templates(tmp_path):
    """Verify custom workflow creation and template instantiation."""
    service = ResearchWorkflowService(storage_dir=tmp_path)

    # 1. Custom workflow
    defn = ResearchDefinition(
        research_question="Does dataset cleaning improve small-object mAP?",
        hypothesis="Removing blurred bounding boxes increases precision by >= +0.03.",
        objective="Quantify dataset cleaning impact",
        success_metrics=["precision", "map50"],
        constraints=["Fixed test split"],
    )
    wf_custom = service.create_workflow(
        name="Dataset Cleaning Study",
        research_definition=defn,
        dataset_config=DatasetConfig(dataset_id="safety_v2", dataset_version="v2.1.0"),
    )
    assert wf_custom.workflow_id.startswith("wf_")
    assert wf_custom.status == WorkflowStatus.READY
    assert wf_custom.current_stage == WorkflowStage.RESEARCH_DEFINITION
    assert wf_custom.dataset_config.dataset_id == "safety_v2"
    assert wf_custom.dataset_config.is_locked is True

    # 2. Template workflow (Active Learning)
    wf_al = service.create_from_template(
        template_type=WorkflowTemplateType.ACTIVE_LEARNING_STUDY,
        name="Active Learning Benchmark",
    )
    assert wf_al.template_type == WorkflowTemplateType.ACTIVE_LEARNING_STUDY
    assert "active learning" in wf_al.research_definition.research_question.lower()


def test_state_machine_transitions_and_rejections(tmp_path):
    """Verify valid state transitions and rejection of invalid state transitions."""
    service = ResearchWorkflowService(storage_dir=tmp_path)
    wf = service.create_from_template(WorkflowTemplateType.BASELINE_VS_VARIANT)

    assert wf.status == WorkflowStatus.READY

    # Start: READY -> RUNNING
    wf = service.start_workflow(wf.workflow_id)
    assert wf.status == WorkflowStatus.RUNNING

    # Pause: RUNNING -> PAUSED
    wf = service.pause_workflow(wf.workflow_id)
    assert wf.status == WorkflowStatus.PAUSED

    # Resume: PAUSED -> RUNNING
    wf = service.resume_workflow(wf.workflow_id)
    assert wf.status == WorkflowStatus.RUNNING

    # Set Review Gate: RUNNING -> WAITING_FOR_REVIEW
    wf = service.update_status(wf.workflow_id, WorkflowStatus.WAITING_FOR_REVIEW)
    assert wf.status == WorkflowStatus.WAITING_FOR_REVIEW

    # Complete: WAITING_FOR_REVIEW -> COMPLETED
    wf = service.update_status(wf.workflow_id, WorkflowStatus.COMPLETED)
    assert wf.status == WorkflowStatus.COMPLETED

    # Illegal transition: COMPLETED -> RUNNING (Must raise InvalidStateTransitionError)
    with pytest.raises(InvalidStateTransitionError):
        service.update_status(wf.workflow_id, WorkflowStatus.RUNNING)


def test_stage_progression_and_decision_gates(tmp_path):
    """Verify sequential 8-stage advancement and human decision gate handling."""
    service = ResearchWorkflowService(storage_dir=tmp_path)
    wf = service.create_from_template(WorkflowTemplateType.ACTIVE_LEARNING_STUDY)

    # 1. Advance: RESEARCH_DEFINITION -> DATASET
    wf = service.advance_stage(wf.workflow_id)
    assert wf.current_stage == WorkflowStage.DATASET

    # 2. Advance: DATASET -> EXPERIMENT
    wf = service.advance_stage(wf.workflow_id)
    assert wf.current_stage == WorkflowStage.EXPERIMENT

    # 3. Advance: EXPERIMENT -> TRAINING (Triggers WAITING_FOR_REVIEW)
    wf = service.advance_stage(wf.workflow_id)
    assert wf.current_stage == WorkflowStage.TRAINING
    assert wf.status == WorkflowStatus.WAITING_FOR_REVIEW

    # 4. Advance: TRAINING -> EVALUATION
    wf = service.advance_stage(wf.workflow_id)
    assert wf.current_stage == WorkflowStage.EVALUATION

    # 5. Advance: EVALUATION -> ERROR_ANALYSIS
    wf = service.advance_stage(wf.workflow_id)
    assert wf.current_stage == WorkflowStage.ERROR_ANALYSIS

    # 6. Advance: ERROR_ANALYSIS -> COMPARISON
    wf = service.advance_stage(wf.workflow_id)
    assert wf.current_stage == WorkflowStage.COMPARISON

    # 7. Decision Gate: INVESTIGATE Loop -> Returns to ERROR_ANALYSIS and increments iteration
    assert wf.current_iteration == 1
    wf = service.record_decision(
        wf_id=wf.workflow_id,
        decision=DecisionType.INVESTIGATE,
        rationale="Small helmets in dark illumination show false negatives; need hard sample review.",
        target_stage=WorkflowStage.ERROR_ANALYSIS,
    )
    assert wf.current_iteration == 2
    assert wf.current_stage == WorkflowStage.ERROR_ANALYSIS
    assert len(wf.decision_history) == 1
    assert wf.decision_history[0].decision == DecisionType.INVESTIGATE

    # 8. Decision Gate: ACCEPT -> Advances to REPORT stage
    wf = service.advance_stage(wf.workflow_id)  # to COMPARISON
    wf = service.record_decision(
        wf_id=wf.workflow_id,
        decision=DecisionType.ACCEPT,
        rationale="Hypothesis confirmed: +0.062 mAP with 50% fewer samples.",
    )
    assert wf.current_stage == WorkflowStage.REPORT
    assert wf.generated_report_markdown is not None
    assert "Active Learning" in wf.generated_report_markdown


def test_stage_notes_and_lineage_graph(tmp_path):
    """Verify attaching researcher qualitative notes and generating lineage DAG."""
    service = ResearchWorkflowService(storage_dir=tmp_path)
    wf = service.create_from_template(WorkflowTemplateType.BASELINE_VS_VARIANT)
    wf.baseline_run_id = "run_base_640"
    wf.evaluation_ids = ["eval_test_640"]
    service.save_to_disk()

    # Add Note
    note = service.add_stage_note(
        wf_id=wf.workflow_id,
        stage=WorkflowStage.DATASET,
        text="Verified train/test split has 0 leakage.",
        author="Dr. Lead",
    )
    assert note.author == "Dr. Lead"
    assert len(wf.stage_notes) == 1

    # Get Lineage DAG
    lineage = service.get_lineage_graph(wf.workflow_id)
    assert len(lineage.nodes) >= 4
    assert len(lineage.edges) >= 3
    node_types = {n.entity_type for n in lineage.nodes}
    assert "research" in node_types
    assert "dataset" in node_types
    assert "training_run" in node_types
    assert "evaluation" in node_types


def test_workflow_export_package(tmp_path):
    """Verify self-contained workflow package export with SHA-256 reproducibility hash."""
    service = ResearchWorkflowService(storage_dir=tmp_path)
    wf = service.create_from_template(WorkflowTemplateType.MODEL_ARCHITECTURE_COMPARISON)

    pkg = service.export_workflow_package(wf.workflow_id)
    assert pkg.workflow.workflow_id == wf.workflow_id
    assert len(pkg.reproducibility_hash) == 16
    assert "Research Question" in pkg.report_markdown


def test_workflow_rest_api_endpoints():
    """Verify REST API routes for research workflow operations."""
    # 1. Create from template
    res_create = client.post(
        "/api/v1/workflows/template",
        json={
            "template_type": "ACTIVE_LEARNING_STUDY",
            "name": "API Active Learning Test",
            "dataset_id": "safety_v2",
            "dataset_version": "v2.0.0",
        },
    )
    assert res_create.status_code == 201
    wf_data = res_create.json()
    wf_id = wf_data["workflow_id"]

    # 2. Get workflow
    res_get = client.get(f"/api/v1/workflows/{wf_id}")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "API Active Learning Test"

    # 3. Start workflow
    res_start = client.post(f"/api/v1/workflows/{wf_id}/start")
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "RUNNING"

    # 4. Advance stage
    res_adv = client.post(f"/api/v1/workflows/{wf_id}/advance")
    assert res_adv.status_code == 200
    assert res_adv.json()["current_stage"] == "DATASET"

    # 5. Add Note
    res_note = client.post(
        f"/api/v1/workflows/{wf_id}/notes",
        json={"stage": "DATASET", "text": "API note verification", "author": "Tester"},
    )
    assert res_note.status_code == 201

    # 6. Record Decision
    res_dec = client.post(
        f"/api/v1/workflows/{wf_id}/decisions",
        json={"decision": "ACCEPT", "rationale": "Validated via API test"},
    )
    assert res_dec.status_code == 200

    # 7. Get Lineage
    res_lin = client.get(f"/api/v1/workflows/{wf_id}/lineage")
    assert res_lin.status_code == 200
    assert "nodes" in res_lin.json()

    # 8. Export Package
    res_exp = client.get(f"/api/v1/workflows/{wf_id}/export")
    assert res_exp.status_code == 200
    assert "reproducibility_hash" in res_exp.json()
