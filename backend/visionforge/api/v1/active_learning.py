"""VisionForge Active Learning & Intelligent Sample Selection API Routes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.active_learning.loop import ActiveLearningLoopError
from visionforge.active_learning.schemas import (
    ActiveLearningIteration,
    ActiveLearningRun,
    ExecuteLoopRequest,
    RankedSample,
    ReviewDecisionRequest,
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

logger = logging.getLogger("visionforge.api.v1.active_learning")

router = APIRouter(prefix="/active-learning", tags=["Active Learning"])


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
    "/runs",
    response_model=ActiveLearningRun,
    status_code=status.HTTP_201_CREATED,
    summary="Create Active Learning run and generate recommendations",
)
def create_active_learning_run(payload: CreateRunRequest) -> ActiveLearningRun:
    """Create Active Learning run, validate test-set protection, and generate ranked recommendations."""
    service = get_active_learning_service()
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


@router.get(
    "/runs",
    response_model=list[ActiveLearningRun],
    summary="List active learning runs",
)
def list_active_learning_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[ActiveLearningRun]:
    """Retrieve paginated list of historical active learning runs."""
    service = get_active_learning_service()
    return service.list_runs(limit=limit, offset=offset)


@router.get(
    "/runs/{run_id}",
    response_model=ActiveLearningRun,
    summary="Get single active learning run detail",
)
def get_active_learning_run(run_id: str) -> ActiveLearningRun:
    """Retrieve complete metadata and selected samples for an active learning run."""
    service = get_active_learning_service()
    try:
        return service.get_run(run_id)
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/review",
    response_model=ActiveLearningRun,
    summary="Submit human review decision on recommended sample",
)
def submit_review_decision(payload: ReviewDecisionRequest) -> ActiveLearningRun:
    """Submit human review decision (accept, reject, skip, mark for labeling) on a sample."""
    service = get_active_learning_service()
    try:
        return service.submit_review_decision(payload)
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/queue",
    response_model=list[RankedSample],
    summary="Get human review queue for an active learning run",
)
def get_review_queue(run_id: str) -> list[RankedSample]:
    """Retrieve ranked candidate sample review queue for a specific run."""
    service = get_active_learning_service()
    try:
        run = service.get_run(run_id)
        return run.selected_samples
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/bias",
    response_model=SelectionBiasReport,
    summary="Get selection bias analysis report",
)
def analyze_selection_bias(run_id: str) -> SelectionBiasReport:
    """Analyze potential selection bias in the recommended candidate set."""
    service = get_active_learning_service()
    try:
        return service.analyze_selection_bias(run_id)
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/compare",
    response_model=StrategyComparisonResult,
    summary="Compare two active learning selection strategies",
)
def compare_strategies(payload: StrategyComparisonRequest) -> StrategyComparisonResult:
    """Generate comparative analysis between two active learning selection strategies."""
    service = get_active_learning_service()
    return service.compare_strategies(
        dataset_id=payload.dataset_id,
        model_id=payload.model_id,
        strategy_a=payload.strategy_a,
        strategy_b=payload.strategy_b,
        top_k=payload.top_k,
    )


@router.post(
    "/loop",
    response_model=ActiveLearningIteration,
    status_code=status.HTTP_201_CREATED,
    summary="Execute end-to-end active learning retraining loop and compute metric delta",
)
def execute_active_learning_loop(payload: ExecuteLoopRequest) -> ActiveLearningIteration:
    """Execute complete closed-loop retraining iteration and measure performance improvement on untouched test split."""
    service = get_active_learning_service()
    try:
        return service.execute_loop(
            active_learning_run_id=payload.active_learning_run_id,
            new_version_tag=payload.new_version_tag,
        )
    except (ActiveLearningRunNotFoundError, ActiveLearningLoopError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/iterations",
    response_model=list[ActiveLearningIteration],
    summary="List active learning retraining loop iterations",
)
def list_active_learning_iterations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[ActiveLearningIteration]:
    """Retrieve paginated list of completed active learning retraining loop iterations."""
    service = get_active_learning_service()
    return service.list_iterations(limit=limit, offset=offset)


@router.get(
    "/iterations/{iteration_id}",
    response_model=ActiveLearningIteration,
    summary="Get single active learning iteration detail and performance verdict",
)
def get_active_learning_iteration(iteration_id: str) -> ActiveLearningIteration:
    """Retrieve complete metadata and metric delta for a specific active learning iteration."""
    service = get_active_learning_service()
    try:
        return service.get_iteration(iteration_id)
    except ActiveLearningRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

