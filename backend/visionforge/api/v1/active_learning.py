"""VisionForge Active Learning & Intelligent Sample Selection API Routes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.active_learning.loop import ActiveLearningLoopError
from visionforge.active_learning.schemas import (
    ActiveLearningCycle,
    ActiveLearningCycleHistoryItem,
    ActiveLearningIteration,
    CandidateSampleDetail,
    ExecuteLoopRequest,
    ReviewDecisionRequest,
    ReviewDecisionType,
    ReviewerDecisionRecord,
    SampleReviewConsensus,
    SelectionBiasReport,
    SelectionStrategy,
    SignalWeights,
    StrategyComparisonRequest,
    StrategyComparisonResult,
)
from visionforge.active_learning.service import (
    ActiveLearningRunNotFoundError,
    TestSetProtectionError,
    get_active_learning_service,
)
from visionforge.core.responses import APIResponse, success_response

logger = logging.getLogger("visionforge.api.v1.active_learning")

router = APIRouter(prefix="/active-learning", tags=["Active Learning"])


class CreateCycleRequest(BaseModel):
    name: str = Field(default="Active Learning Curation Cycle", description="Cycle display name")
    dataset_id: str = Field(default="safety_v2", description="Target dataset identifier")
    dataset_version: str = Field(default="v1.0.0", description="Source dataset version tag")
    model_id: str = Field(default="yolo11s.pt", description="Target model identifier")
    model_version: str = Field(default="1.0.0", description="Model version tag")
    candidate_pool_id: str = Field(default="pool_unlabeled_site_cctv", description="Candidate pool ID")
    strategy: SelectionStrategy = Field(
        default=SelectionStrategy.HYBRID, description="Selection strategy"
    )
    budget: int = Field(default=50, ge=1, le=500, description="Exact human review sample budget")
    weights: SignalWeights | None = Field(default=None, description="Signal combination weights")


class SelectCandidatesRequest(BaseModel):
    budget: int = Field(default=50, ge=1, le=500, description="Sample selection budget")
    strategy: SelectionStrategy | None = Field(default=None, description="Strategy to execute")
    weights: SignalWeights | None = Field(default=None, description="Custom weights")


class CommitDatasetVersionRequest(BaseModel):
    new_version_tag: str = Field(default="v2.1.0", description="Target dataset version tag")
    changes_summary: str | None = Field(default=None, description="Curation changes summary")


class ResolveDisagreementRequest(BaseModel):
    sample_id: str = Field(description="Sample identifier")
    final_decision: ReviewDecisionType = Field(description="Final resolved decision")
    resolved_by: str = Field(default="Lead Researcher", description="Admin reviewer name")


def _get_service():
    return get_active_learning_service()


# ─── Active Learning Cycles ────────────────────────────────────────────

@router.post(
    "/cycles",
    response_model=APIResponse[ActiveLearningCycle],
    status_code=status.HTTP_201_CREATED,
    summary="Create Active Learning Cycle",
)
def create_active_learning_cycle(payload: CreateCycleRequest) -> APIResponse[ActiveLearningCycle]:
    """Create a new active learning cycle and select initial prioritized candidate batch."""
    service = _get_service()
    try:
        cycle = service.create_cycle(
            name=payload.name,
            dataset_id=payload.dataset_id,
            dataset_version=payload.dataset_version,
            model_id=payload.model_id,
            model_version=payload.model_version,
            candidate_pool_id=payload.candidate_pool_id,
            strategy=payload.strategy,
            budget=payload.budget,
            weights=payload.weights,
        )
        return success_response(data=cycle, message=f"Created active learning cycle '{cycle.cycle_id}' with {len(cycle.selected_samples)} candidates")
    except TestSetProtectionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/cycles",
    response_model=APIResponse[list[ActiveLearningCycle]],
    summary="List Active Learning Cycles",
)
def list_active_learning_cycles(
    dataset_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[ActiveLearningCycle]]:
    """Retrieve list of active learning cycles."""
    service = _get_service()
    cycles = service.list_cycles(dataset_id=dataset_id, limit=limit, offset=offset)
    return success_response(data=cycles, message=f"Retrieved {len(cycles)} active learning cycle(s)")


@router.get(
    "/cycles/history",
    response_model=APIResponse[list[ActiveLearningCycleHistoryItem]],
    summary="Get Active Learning Cycle History & Progression",
)
def get_cycle_history(
    dataset_id: str = Query(default="safety_v2"),
) -> APIResponse[list[ActiveLearningCycleHistoryItem]]:
    """Retrieve longitudinal cycle history tracking before/after mAP progression."""
    service = _get_service()
    history = service.get_cycle_history(dataset_id=dataset_id)
    return success_response(data=history, message=f"Retrieved {len(history)} active learning cycle milestone(s)")


@router.get(
    "/cycles/{cycle_id}",
    response_model=APIResponse[ActiveLearningCycle],
    summary="Get Single Active Learning Cycle",
)
def get_active_learning_cycle(cycle_id: str) -> APIResponse[ActiveLearningCycle]:
    """Retrieve detailed cycle record with candidate sample queue."""
    service = _get_service()
    try:
        cycle = service.get_cycle(cycle_id)
        return success_response(data=cycle, message=f"Retrieved cycle '{cycle_id}'")
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/cycles/{cycle_id}/select",
    response_model=APIResponse[ActiveLearningCycle],
    summary="Execute Candidate Sample Selection",
)
def select_candidates_for_cycle(
    cycle_id: str, payload: SelectCandidatesRequest
) -> APIResponse[ActiveLearningCycle]:
    """Execute candidate selection for a cycle respecting exact budget."""
    service = _get_service()
    try:
        cycle = service.select_candidates_for_cycle(
            cycle_id=cycle_id,
            budget=payload.budget,
            strategy=payload.strategy,
            weights=payload.weights,
        )
        return success_response(data=cycle, message=f"Selected {len(cycle.selected_samples)} samples under {cycle.strategy.value} strategy")
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/cycles/{cycle_id}/review",
    response_model=APIResponse[ReviewerDecisionRecord],
    summary="Submit Human Review Decision",
)
def submit_review_decision(
    cycle_id: str, payload: ReviewDecisionRequest
) -> APIResponse[ReviewerDecisionRecord]:
    """Record human review decision for a sample within a cycle."""
    service = _get_service()
    try:
        rec = service.record_review_decision(
            cycle_id=cycle_id,
            sample_id=payload.image_id,
            decision=payload.decision,
            reviewer_id=payload.reviewer_id,
            ground_truth_class=payload.ground_truth_class,
            notes=payload.notes,
            bbox_corrections=payload.bbox_corrections,
        )
        return success_response(data=rec, message=f"Recorded review decision '{payload.decision.value}' for {payload.image_id}")
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/cycles/{cycle_id}/consensus/{sample_id}",
    response_model=APIResponse[SampleReviewConsensus],
    summary="Get Multi-Reviewer Consensus for Sample",
)
def get_sample_consensus(cycle_id: str, sample_id: str) -> APIResponse[SampleReviewConsensus]:
    """Evaluate multi-reviewer agreement for a specific sample."""
    service = _get_service()
    consensus = service.get_sample_consensus(sample_id)
    return success_response(data=consensus, message=f"Consensus evaluated: {consensus.consensus_status.value}")


@router.post(
    "/cycles/{cycle_id}/commit-version",
    response_model=APIResponse[ActiveLearningCycle],
    summary="Commit Curated Dataset Version",
)
def commit_cycle_dataset_version(
    cycle_id: str, payload: CommitDatasetVersionRequest
) -> APIResponse[ActiveLearningCycle]:
    """Explicit confirmation to commit approved reviewed samples into a new immutable dataset version."""
    service = _get_service()
    try:
        cycle = service.commit_cycle_dataset_version(
            cycle_id=cycle_id,
            new_version_tag=payload.new_version_tag,
            changes_summary=payload.changes_summary,
        )
        return success_response(data=cycle, message=f"Committed new dataset version '{payload.new_version_tag}'")
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


class CreateRunRequest(BaseModel):
    dataset_id: str = Field(description="Target dataset identifier")
    model_id: str = Field(default="yolo11s.pt", description="Target model identifier")
    candidate_paths: list[str] | None = Field(default=None, description="Candidate image file paths")
    strategy: SelectionStrategy = Field(
        default=SelectionStrategy.UNCERTAINTY_DIVERSITY, description="Selection strategy"
    )
    weights: SignalWeights | None = Field(default=None, description="Signal combination weights")
    top_k: int = Field(default=25, ge=1, le=500, description="Top-K sample recommendations")
    experiment_id: str | None = Field(default=None, description="Optional experiment ID link")


@router.post(
    "/compare",
    response_model=StrategyComparisonResult,
    summary="Compare Two Selection Strategies",
)
def compare_strategies(payload: StrategyComparisonRequest) -> StrategyComparisonResult:
    """Generate comparative overlap analysis between two active learning selection strategies."""
    service = _get_service()
    return service.compare_strategies(payload)


# ─── Backward-Compatibility Run Routes ───────────────────────────────

@router.post(
    "/runs",
    response_model=ActiveLearningCycle,
    status_code=status.HTTP_201_CREATED,
    summary="Create Active Learning run (legacy)",
)
def create_active_learning_run_legacy(payload: CreateRunRequest) -> ActiveLearningCycle:
    service = _get_service()
    try:
        return service.create_run(
            dataset_id=payload.dataset_id,
            model_id=payload.model_id,
            candidate_paths=payload.candidate_paths,
            strategy=payload.strategy,
            weights=payload.weights,
            top_k=payload.top_k,
            experiment_id=payload.experiment_id,
        )
    except TestSetProtectionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/review",
    response_model=ActiveLearningCycle,
    summary="Submit human review decision (legacy)",
)
def submit_review_decision_legacy(payload: ReviewDecisionRequest) -> ActiveLearningCycle:
    service = _get_service()
    try:
        return service.submit_review_decision(payload)
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs",
    response_model=list[ActiveLearningCycle],
    summary="List active learning runs (legacy)",
)
def list_active_learning_runs_legacy(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[ActiveLearningCycle]:
    return _get_service().list_runs(limit=limit, offset=offset)


@router.get(
    "/runs/{run_id}",
    response_model=ActiveLearningCycle,
    summary="Get single active learning run (legacy)",
)
def get_active_learning_run_legacy(run_id: str) -> ActiveLearningCycle:
    try:
        return _get_service().get_run(run_id)
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/queue",
    response_model=list[CandidateSampleDetail],
    summary="Get human review queue (legacy)",
)
def get_review_queue_legacy(run_id: str) -> list[CandidateSampleDetail]:
    try:
        run = _get_service().get_run(run_id)
        return run.selected_samples
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/bias",
    response_model=SelectionBiasReport,
    summary="Get selection bias analysis report (legacy)",
)
def analyze_selection_bias_legacy(run_id: str) -> SelectionBiasReport:
    try:
        return _get_service().generate_bias_report(run_id)
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/loop",
    response_model=ActiveLearningIteration,
    status_code=status.HTTP_201_CREATED,
    summary="Execute retraining loop (legacy)",
)
def execute_active_learning_loop_legacy(payload: ExecuteLoopRequest) -> ActiveLearningIteration:
    try:
        return _get_service().execute_loop(
            active_learning_run_id=payload.active_learning_run_id,
            new_version_tag=payload.new_version_tag,
        )
    except (ActiveLearningRunNotFoundError, ActiveLearningLoopError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/iterations",
    response_model=list[ActiveLearningIteration],
    summary="List loop iterations (legacy)",
)
def list_active_learning_iterations_legacy(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[ActiveLearningIteration]:
    return _get_service().list_iterations(limit=limit, offset=offset)


@router.get(
    "/iterations/{iteration_id}",
    response_model=ActiveLearningIteration,
    summary="Get loop iteration (legacy)",
)
def get_active_learning_iteration_legacy(iteration_id: str) -> ActiveLearningIteration:
    try:
        return _get_service().get_iteration(iteration_id)
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
