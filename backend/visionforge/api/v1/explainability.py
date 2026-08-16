"""REST API Endpoints for VisionForge Model Explainability & Visual Diagnostics."""

import logging

from fastapi import APIRouter, HTTPException, Query, status

from visionforge.core.responses import APIResponse, success_response
from visionforge.explainability.schemas import (
    AddResearcherNoteRequest,
    AttributionArtifact,
    CompareExplanationsRequest,
    CreateExplanationRequest,
    ExplanationComparison,
    ExplanationMethod,
    ExplanationRun,
    ReviewExplanationRequest,
    ReviewRating,
)
from visionforge.explainability.service import (
    ExplanationNotFoundError,
    get_explainability_service,
)

logger = logging.getLogger("visionforge.api.v1.explainability")

router = APIRouter(prefix="/explainability", tags=["Model Explainability & Visual Diagnostics"])


def _get_service():
    return get_explainability_service()


@router.post(
    "/explanations",
    response_model=APIResponse[ExplanationRun],
    status_code=status.HTTP_201_CREATED,
    summary="Create or Retrieve Visual Attribution Explanation",
)
def create_explanation(payload: CreateExplanationRequest) -> APIResponse[ExplanationRun]:
    """Generate a spatial attribution heatmap (Grad-CAM, Layer-CAM, Attention Maps) or serve from cache."""
    service = _get_service()
    run = service.create_explanation(payload)
    msg = (
        f"Served cached explanation '{run.explanation_id}'"
        if run.cache_hit
        else f"Generated explanation '{run.explanation_id}' (Status: {run.status.value})"
    )
    return success_response(data=run, message=msg)


@router.get(
    "/explanations",
    response_model=APIResponse[list[ExplanationRun]],
    summary="List Explanation Runs",
)
def list_explanations(
    model_id: str | None = Query(default=None),
    dataset_id: str | None = Query(default=None),
    class_name: str | None = Query(default=None),
    is_correct: bool | None = Query(default=None),
    method: ExplanationMethod | None = Query(default=None),
    review_status: ReviewRating | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[ExplanationRun]]:
    """List historical explanation runs with optional filtering."""
    service = _get_service()
    runs = service.list_explanations(
        model_id=model_id,
        dataset_id=dataset_id,
        class_name=class_name,
        is_correct=is_correct,
        method=method,
        review_status=review_status,
        limit=limit,
        offset=offset,
    )
    return success_response(data=runs, message=f"Retrieved {len(runs)} explanation run(s)")


@router.get(
    "/explanations/{explanation_id}",
    response_model=APIResponse[ExplanationRun],
    summary="Get Single Explanation Record",
)
def get_explanation(explanation_id: str) -> APIResponse[ExplanationRun]:
    """Retrieve full explanation run descriptor and telemetry."""
    service = _get_service()
    try:
        run = service.get_explanation(explanation_id)
        return success_response(data=run, message=f"Retrieved explanation '{explanation_id}'")
    except ExplanationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/explanations/{explanation_id}/status",
    response_model=APIResponse[dict[str, str]],
    summary="Get Explanation Execution Status",
)
def get_explanation_status(explanation_id: str) -> APIResponse[dict[str, str]]:
    """Retrieve explanation status and error diagnostics if failed."""
    service = _get_service()
    try:
        run = service.get_explanation(explanation_id)
        return success_response(
            data={
                "explanation_id": run.explanation_id,
                "status": run.status.value,
                "error_message": run.error_message or "",
            },
            message="Retrieved status",
        )
    except ExplanationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/explanations/{explanation_id}/artifact",
    response_model=APIResponse[AttributionArtifact],
    summary="Get Raw Attribution Matrix Artifact",
)
def get_explanation_artifact(explanation_id: str) -> APIResponse[AttributionArtifact]:
    """Retrieve the raw 2D spatial heatmap matrix and concentration statistics."""
    service = _get_service()
    try:
        run = service.get_explanation(explanation_id)
        if not run.artifact:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No attribution artifact generated for explanation '{explanation_id}' (Status: {run.status.value})",
            )
        return success_response(data=run.artifact, message="Retrieved attribution artifact")
    except ExplanationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/explanations/{explanation_id}/review",
    response_model=APIResponse[ExplanationRun],
    summary="Record Human Review Rating",
)
def review_explanation(
    explanation_id: str, payload: ReviewExplanationRequest
) -> APIResponse[ExplanationRun]:
    """Submit human assessment rating ('USEFUL', 'NOT_USEFUL', 'UNCLEAR', 'NEEDS_INVESTIGATION')."""
    service = _get_service()
    try:
        run = service.review_explanation(explanation_id, payload)
        return success_response(
            data=run, message=f"Recorded review rating '{payload.rating.value}'"
        )
    except ExplanationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/explanations/{explanation_id}/notes",
    response_model=APIResponse[ExplanationRun],
    summary="Append Researcher Observation Note",
)
def add_researcher_note(
    explanation_id: str, payload: AddResearcherNoteRequest
) -> APIResponse[ExplanationRun]:
    """Attach a researcher diagnostic observation note to an explanation record."""
    service = _get_service()
    try:
        run = service.add_researcher_note(explanation_id, payload)
        return success_response(data=run, message="Researcher note appended")
    except ExplanationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/compare",
    response_model=APIResponse[ExplanationComparison],
    summary="Compare Two Explanations Side-by-Side",
)
def compare_explanations(payload: CompareExplanationsRequest) -> APIResponse[ExplanationComparison]:
    """Compute spatial attribution difference between two explanation runs."""
    service = _get_service()
    try:
        cmp_res = service.compare_explanations(
            payload.explanation_id_a, payload.explanation_id_b
        )
        return success_response(
            data=cmp_res, message="Explanation comparison completed successfully"
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
