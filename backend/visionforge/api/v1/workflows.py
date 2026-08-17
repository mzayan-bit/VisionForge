"""VisionForge Research Workflow API Routes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.workflows.schemas import (
    DatasetConfig,
    DecisionType,
    ResearchDefinition,
    ResearchWorkflow,
    StageNote,
    WorkflowExportPackage,
    WorkflowLineageGraph,
    WorkflowStage,
    WorkflowStatus,
    WorkflowTemplateType,
)
from visionforge.workflows.service import (
    InvalidStateTransitionError,
    WorkflowNotFoundError,
    get_research_workflow_service,
)

logger = logging.getLogger("visionforge.api.v1.workflows")

router = APIRouter(prefix="/workflows", tags=["Research Workflows"])


class CreateWorkflowRequest(BaseModel):
    name: str = Field(description="Workflow title")
    research_definition: ResearchDefinition
    dataset_config: DatasetConfig | None = None
    template_type: WorkflowTemplateType = WorkflowTemplateType.CUSTOM
    description: str = Field(default="")


class CreateFromTemplateRequest(BaseModel):
    template_type: WorkflowTemplateType
    name: str | None = None
    dataset_id: str = "safety_v2"
    dataset_version: str = "v2.0.0"


class RecordDecisionRequest(BaseModel):
    decision: DecisionType
    rationale: str
    reviewer: str = "Researcher"
    target_stage: WorkflowStage | None = None


class AddStageNoteRequest(BaseModel):
    stage: WorkflowStage
    text: str
    author: str = "Researcher"


class AttachExperimentRequest(BaseModel):
    experiment_id: str


class StatusTransitionRequest(BaseModel):
    status: WorkflowStatus
    reason: str = ""


@router.post(
    "",
    response_model=ResearchWorkflow,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Research Workflow",
)
def create_workflow(payload: CreateWorkflowRequest) -> ResearchWorkflow:
    """Create a new custom research workflow with locked evaluation stages."""
    service = get_research_workflow_service()
    return service.create_workflow(
        name=payload.name,
        research_definition=payload.research_definition,
        dataset_config=payload.dataset_config,
        template_type=payload.template_type,
        description=payload.description,
    )


@router.post(
    "/template",
    response_model=ResearchWorkflow,
    status_code=status.HTTP_201_CREATED,
    summary="Create a workflow from a vetted template",
)
def create_from_template(payload: CreateFromTemplateRequest) -> ResearchWorkflow:
    """Instantiate a structured research study from a standardized template."""
    service = get_research_workflow_service()
    return service.create_from_template(
        template_type=payload.template_type,
        name=payload.name,
        dataset_id=payload.dataset_id,
        dataset_version=payload.dataset_version,
    )


@router.get(
    "",
    response_model=list[ResearchWorkflow],
    summary="List all research workflows",
)
def list_workflows() -> list[ResearchWorkflow]:
    """Retrieve all research workflows sorted chronologically."""
    service = get_research_workflow_service()
    return service.list_workflows()


@router.get(
    "/{wf_id}",
    response_model=ResearchWorkflow,
    summary="Get single research workflow details",
)
def get_workflow(wf_id: str) -> ResearchWorkflow:
    """Retrieve complete workflow status, stage progress, and decision history."""
    service = get_research_workflow_service()
    try:
        return service.get_workflow(wf_id)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{wf_id}/start",
    response_model=ResearchWorkflow,
    summary="Start workflow execution",
)
def start_workflow(wf_id: str) -> ResearchWorkflow:
    """Transition workflow to RUNNING status."""
    service = get_research_workflow_service()
    try:
        return service.start_workflow(wf_id)
    except (WorkflowNotFoundError, InvalidStateTransitionError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/{wf_id}/pause",
    response_model=ResearchWorkflow,
    summary="Pause active workflow",
)
def pause_workflow(wf_id: str) -> ResearchWorkflow:
    """Pause workflow safely."""
    service = get_research_workflow_service()
    try:
        return service.pause_workflow(wf_id)
    except (WorkflowNotFoundError, InvalidStateTransitionError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/{wf_id}/resume",
    response_model=ResearchWorkflow,
    summary="Resume paused workflow",
)
def resume_workflow(wf_id: str) -> ResearchWorkflow:
    """Resume workflow from last valid stage."""
    service = get_research_workflow_service()
    try:
        return service.resume_workflow(wf_id)
    except (WorkflowNotFoundError, InvalidStateTransitionError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/{wf_id}/cancel",
    response_model=ResearchWorkflow,
    summary="Cancel workflow",
)
def cancel_workflow(wf_id: str, reason: str = Query("Researcher cancelled")) -> ResearchWorkflow:
    """Cancel workflow execution."""
    service = get_research_workflow_service()
    try:
        return service.cancel_workflow(wf_id, reason=reason)
    except (WorkflowNotFoundError, InvalidStateTransitionError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/{wf_id}/advance",
    response_model=ResearchWorkflow,
    summary="Advance to next workflow stage",
)
def advance_stage(wf_id: str) -> ResearchWorkflow:
    """Progress to next sequential stage in the 8-stage pipeline."""
    service = get_research_workflow_service()
    try:
        return service.advance_stage(wf_id)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{wf_id}/decisions",
    response_model=ResearchWorkflow,
    summary="Record human decision gate result",
)
def record_decision(wf_id: str, payload: RecordDecisionRequest) -> ResearchWorkflow:
    """Process researcher decision at review gates (ACCEPT, REJECT, or INVESTIGATE loop)."""
    service = get_research_workflow_service()
    try:
        return service.record_decision(
            wf_id=wf_id,
            decision=payload.decision,
            rationale=payload.rationale,
            reviewer=payload.reviewer,
            target_stage=payload.target_stage,
        )
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{wf_id}/notes",
    response_model=StageNote,
    status_code=status.HTTP_201_CREATED,
    summary="Add researcher observational note",
)
def add_stage_note(wf_id: str, payload: AddStageNoteRequest) -> StageNote:
    """Attach qualitative observation note to a stage."""
    service = get_research_workflow_service()
    try:
        return service.add_stage_note(
            wf_id=wf_id,
            stage=payload.stage,
            text=payload.text,
            author=payload.author,
        )
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{wf_id}/attach-experiment",
    response_model=ResearchWorkflow,
    summary="Attach ResearchExperiment to workflow",
)
def attach_experiment(wf_id: str, payload: AttachExperimentRequest) -> ResearchWorkflow:
    """Link a ResearchExperiment record to this workflow."""
    service = get_research_workflow_service()
    try:
        return service.attach_experiment(wf_id=wf_id, experiment_id=payload.experiment_id)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{wf_id}/lineage",
    response_model=WorkflowLineageGraph,
    summary="Get workflow lineage DAG",
)
def get_workflow_lineage(wf_id: str) -> WorkflowLineageGraph:
    """Construct complete directed lineage graph linking research entities."""
    service = get_research_workflow_service()
    try:
        return service.get_lineage_graph(wf_id)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{wf_id}/report",
    response_model=dict[str, str],
    summary="Generate workflow markdown research report",
)
def get_workflow_report(wf_id: str) -> dict[str, str]:
    """Synthesize complete traceable research workflow markdown report."""
    service = get_research_workflow_service()
    try:
        report_md = service.generate_workflow_report(wf_id)
        return {"workflow_id": wf_id, "markdown_report": report_md}
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{wf_id}/export",
    response_model=WorkflowExportPackage,
    summary="Export self-contained research workflow package",
)
def export_workflow_package(wf_id: str) -> WorkflowExportPackage:
    """Export self-contained JSON package with metadata, lineage, report, and hash."""
    service = get_research_workflow_service()
    try:
        return service.export_workflow_package(wf_id)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
