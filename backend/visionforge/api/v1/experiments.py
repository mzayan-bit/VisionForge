"""VisionForge Experiment Tracking and Reproducibility API Routes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.experiments.schemas import (
    Experiment,
    ExperimentComparison,
    ExperimentStatus,
    LineageGraph,
    ReproducibilityReport,
    TimelineEvent,
)
from visionforge.experiments.service import (
    ExperimentNotFoundError,
    get_experiment_service,
)

logger = logging.getLogger("visionforge.api.v1.experiments")

router = APIRouter(prefix="/experiments", tags=["Experiment Tracking"])


class CreateExperimentRequest(BaseModel):
    name: str = Field(description="Experiment display title")
    description: str = Field(default="", description="Description")
    purpose: str = Field(default="", description="Research goal")
    hypothesis: str | None = Field(default=None, description="Hypothesis")
    tags: list[str] = Field(default_factory=lambda: ["baseline"])
    dataset_id: str | None = Field(default=None)
    dataset_version: str | None = Field(default=None)
    preparation_id: str | None = Field(default=None)
    random_seed: int = Field(default=42)


class UpdateNotesRequest(BaseModel):
    hypothesis: str | None = None
    observations: str | None = None
    conclusions: str | None = None


class AddTagsRequest(BaseModel):
    tags: list[str]


class AttachComponentRequest(BaseModel):
    training_run_id: str | None = None
    model_id: str | None = None
    evaluation_id: str | None = None
    benchmark_id: str | None = None
    inference_id: str | None = None
    training_config: dict | None = None


class CompareExperimentsRequest(BaseModel):
    experiment_a_id: str
    experiment_b_id: str


@router.post(
    "",
    response_model=Experiment,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new research experiment",
)
def create_experiment(payload: CreateExperimentRequest) -> Experiment:
    """Create a new research experiment with automated environment snapshotting and dataset fingerprinting."""
    service = get_experiment_service()
    return service.create_experiment(
        name=payload.name,
        description=payload.description,
        purpose=payload.purpose,
        hypothesis=payload.hypothesis,
        tags=payload.tags,
        dataset_id=payload.dataset_id,
        dataset_version=payload.dataset_version,
        preparation_id=payload.preparation_id,
        random_seed=payload.random_seed,
    )


@router.get(
    "",
    response_model=list[Experiment],
    summary="List research experiments",
)
def list_experiments(
    status: ExperimentStatus | None = None,
    tag: str | None = None,
    dataset_id: str | None = None,
    query: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Experiment]:
    """Retrieve paginated list of research experiments with filtering."""
    service = get_experiment_service()
    return service.list_experiments(
        status=status,
        tag=tag,
        dataset_id=dataset_id,
        query=query,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{exp_id}",
    response_model=Experiment,
    summary="Get single experiment detail",
)
def get_experiment(exp_id: str) -> Experiment:
    """Retrieve complete metadata for a specific experiment ID."""
    service = get_experiment_service()
    try:
        return service.get_experiment(exp_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.patch(
    "/{exp_id}/notes",
    response_model=Experiment,
    summary="Update researcher notes and findings",
)
def update_experiment_notes(exp_id: str, payload: UpdateNotesRequest) -> Experiment:
    """Update hypothesis, observations, or conclusions for an experiment."""
    service = get_experiment_service()
    try:
        return service.update_notes(
            exp_id=exp_id,
            hypothesis=payload.hypothesis,
            observations=payload.observations,
            conclusions=payload.conclusions,
        )
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{exp_id}/tags",
    response_model=Experiment,
    summary="Add tags to experiment",
)
def add_experiment_tags(exp_id: str, payload: AddTagsRequest) -> Experiment:
    """Attach tags to an experiment."""
    service = get_experiment_service()
    try:
        return service.add_tags(exp_id=exp_id, tags=payload.tags)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{exp_id}/attach",
    response_model=Experiment,
    summary="Attach resource to experiment",
)
def attach_component(exp_id: str, payload: AttachComponentRequest) -> Experiment:
    """Link a training run, model, evaluation, benchmark, or inference run to an experiment."""
    service = get_experiment_service()
    try:
        return service.attach_component(
            exp_id=exp_id,
            training_run_id=payload.training_run_id,
            model_id=payload.model_id,
            evaluation_id=payload.evaluation_id,
            benchmark_id=payload.benchmark_id,
            inference_id=payload.inference_id,
            training_config=payload.training_config,
        )
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{exp_id}/lineage",
    response_model=LineageGraph,
    summary="Get experiment lineage graph",
)
def get_experiment_lineage(exp_id: str) -> LineageGraph:
    """Construct directed lineage graph linking Dataset -> Prep -> Run -> Model -> Eval -> Benchmark -> Inference."""
    service = get_experiment_service()
    try:
        return service.get_lineage_graph(exp_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{exp_id}/timeline",
    response_model=list[TimelineEvent],
    summary="Get experiment timeline",
)
def get_experiment_timeline(exp_id: str) -> list[TimelineEvent]:
    """Retrieve chronological event timeline for an experiment."""
    service = get_experiment_service()
    try:
        return service.get_timeline(exp_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/compare",
    response_model=ExperimentComparison,
    summary="Compare two experiments and config diff",
)
def compare_experiments(payload: CompareExperimentsRequest) -> ExperimentComparison:
    """Generate side-by-side telemetry comparison and config diff between two experiments."""
    service = get_experiment_service()
    try:
        return service.compare_experiments(payload.experiment_a_id, payload.experiment_b_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{exp_id}/validate",
    response_model=ReproducibilityReport,
    summary="Run reproducibility validation audit",
)
def validate_reproducibility(exp_id: str) -> ReproducibilityReport:
    """Validate dataset fingerprints, checkpoints, snapshots, and git commit references for reproducibility."""
    service = get_experiment_service()
    try:
        return service.validate_reproducibility(exp_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{exp_id}/reproduce",
    response_model=Experiment,
    status_code=status.HTTP_201_CREATED,
    summary="Spawn reproduction attempt run",
)
def reproduce_experiment(exp_id: str, new_name: str | None = None) -> Experiment:
    """Spawn a pre-filled reproduction attempt experiment pre-populated with parent configuration."""
    service = get_experiment_service()
    try:
        return service.reproduce_experiment(exp_id, new_name)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{exp_id}/report",
    response_model=dict[str, str],
    summary="Get structured experiment report markdown",
)
def get_experiment_report(exp_id: str) -> dict[str, str]:
    """Generate markdown research report summarizing experiment lineage and telemetry."""
    service = get_experiment_service()
    try:
        report_md = service.generate_experiment_report(exp_id)
        return {"experiment_id": exp_id, "report_md": report_md}
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
