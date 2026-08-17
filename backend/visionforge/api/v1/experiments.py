"""VisionForge Experiment Tracking and Reproducibility API Routes."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.experiments.schemas import (
    AblationStudy,
    EvaluationProtocol,
    Experiment,
    ExperimentComparison,
    ExperimentRunRecord,
    ExperimentStatus,
    ExperimentVariant,
    LineageGraph,
    ReproducibilityReport,
    ResearchExperiment,
    ResearchReport,
    TimelineEvent,
    VariableDiffItem,
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


# ─── Research Benchmark & Ablation Lab Endpoints ───────────────────


class CreateResearchExperimentRequest(BaseModel):
    name: str = Field(description="Experiment title")
    hypothesis: str = Field(description="Researcher-provided hypothesis text")
    dataset_id: str = Field(default="safety_v2", description="Dataset identifier")
    dataset_version: str = Field(default="v2.0.0", description="Dataset version")
    baseline_name: str = Field(default="Baseline", description="Baseline branch display name")
    baseline_config: dict[str, Any] | None = Field(
        default=None, description="Baseline configuration parameters"
    )
    protocol: EvaluationProtocol | None = Field(default=None, description="Evaluation protocol")
    description: str = Field(default="", description="Abstract or context")


class AddVariantRequest(BaseModel):
    name: str = Field(description="Variant display title")
    config_changes: dict[str, Any] = Field(
        description="Explicit parameters changed relative to baseline"
    )
    description: str = Field(default="", description="Summary of changes")
    dataset_id: str | None = Field(default=None, description="Dataset ID if dataset ablation")
    dataset_version: str | None = Field(
        default=None, description="Dataset version if dataset ablation"
    )
    label_count: int | None = Field(default=None, description="Annotated label count")
    label_percentage: float | None = Field(default=None, description="Dataset budget percentage")


class RecordRunRequest(BaseModel):
    run_id: str = Field(description="Target training/evaluation run ID")
    seed: int = Field(description="Random seed tested")
    model_id: str = Field(default="yolo11s.pt", description="Model checkpoint name")
    metrics: dict[str, float] = Field(
        description="Evaluation scalar metrics (e.g. map50, precision, recall)"
    )
    per_class_metrics: dict[str, float] = Field(
        default_factory=dict, description="Class-level metric scores"
    )
    error_counts: dict[str, int] = Field(default_factory=dict, description="Error taxonomy counts")
    training_time_sec: float | None = Field(
        default=None, description="Training duration in seconds"
    )
    gpu_hours: float | None = Field(default=None, description="GPU hours measured")


@router.post(
    "/research",
    response_model=ResearchExperiment,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Research Experiment",
)
def create_research_experiment(payload: CreateResearchExperimentRequest) -> ResearchExperiment:
    """Create a new formal ResearchExperiment with locked evaluation protocol and baseline branch."""
    service = get_experiment_service()
    return service.create_research_experiment(
        name=payload.name,
        hypothesis=payload.hypothesis,
        dataset_id=payload.dataset_id,
        dataset_version=payload.dataset_version,
        baseline_name=payload.baseline_name,
        baseline_config=payload.baseline_config,
        protocol=payload.protocol,
        description=payload.description,
    )


@router.get(
    "/research",
    response_model=list[ResearchExperiment],
    summary="List all Research Experiments",
)
def list_research_experiments() -> list[ResearchExperiment]:
    """Retrieve all Research Experiments sorted chronologically."""
    service = get_experiment_service()
    return service.list_research_experiments()


@router.get(
    "/research/{exp_id}",
    response_model=ResearchExperiment,
    summary="Get single Research Experiment details",
)
def get_research_experiment(exp_id: str) -> ResearchExperiment:
    """Retrieve complete ResearchExperiment details with variants and multi-seed stats."""
    service = get_experiment_service()
    try:
        return service.get_research_experiment(exp_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/research/{exp_id}/variants",
    response_model=ExperimentVariant,
    status_code=status.HTTP_201_CREATED,
    summary="Add an experimental variant branch",
)
def add_research_variant(exp_id: str, payload: AddVariantRequest) -> ExperimentVariant:
    """Add a controlled experimental branch to a ResearchExperiment."""
    service = get_experiment_service()
    try:
        return service.add_variant(
            exp_id=exp_id,
            name=payload.name,
            config_changes=payload.config_changes,
            description=payload.description,
            dataset_id=payload.dataset_id,
            dataset_version=payload.dataset_version,
            label_count=payload.label_count,
            label_percentage=payload.label_percentage,
        )
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/research/{exp_id}/variants/{variant_id}/runs",
    response_model=ExperimentVariant,
    summary="Record a multi-seed evaluation trial run",
)
def record_variant_run(
    exp_id: str, variant_id: str, payload: RecordRunRequest
) -> ExperimentVariant:
    """Attach a seed trial to a variant and recompute aggregated statistics."""
    service = get_experiment_service()
    try:
        run_record = ExperimentRunRecord(
            run_id=payload.run_id,
            seed=payload.seed,
            model_id=payload.model_id,
            metrics=payload.metrics,
            per_class_metrics=payload.per_class_metrics,
            error_counts=payload.error_counts,
            training_time_sec=payload.training_time_sec,
            gpu_hours=payload.gpu_hours,
        )
        return service.record_run(exp_id=exp_id, variant_id=variant_id, run_record=run_record)
    except (ExperimentNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/research/{exp_id}/variants/{variant_id}/diff",
    response_model=list[VariableDiffItem],
    summary="Get configuration parameter diff table",
)
def get_configuration_diff(exp_id: str, variant_id: str) -> list[VariableDiffItem]:
    """Generate parameter-level diff comparing Baseline vs Variant."""
    service = get_experiment_service()
    try:
        return service.compute_configuration_diff(exp_id=exp_id, variant_id=variant_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/research/{exp_id}/ablation",
    response_model=AblationStudy,
    summary="Get component ablation study matrix",
)
def get_ablation_matrix(exp_id: str) -> AblationStudy:
    """Generate structured component ablation matrix with measured effect deltas."""
    service = get_experiment_service()
    try:
        return service.compute_ablation_matrix(exp_id=exp_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/research/{exp_id}/research-report",
    response_model=ResearchReport,
    summary="Generate grounded research report",
)
def get_research_report(exp_id: str) -> ResearchReport:
    """Synthesize formal research report with grounded conclusions and statistical summaries."""
    service = get_experiment_service()
    try:
        return service.generate_research_report(exp_id=exp_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
