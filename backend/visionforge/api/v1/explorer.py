"""Embedding Explorer REST API Endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException

from visionforge.core.responses import APIResponse, success_response
from visionforge.explorer.schemas import ExplorerDatasetPayload, ProjectionRequest
from visionforge.explorer.service import (
    EmbeddingExplorerService,
    get_embedding_explorer_service,
)

router = APIRouter(tags=["Embedding Explorer"])


def _get_service() -> EmbeddingExplorerService:
    return get_embedding_explorer_service()


@router.get(
    "/explorer/stats",
    response_model=APIResponse[dict[str, Any]],
    summary="Get Embedding Space Explorer Statistics",
    description=(
        "Returns telemetry metrics regarding dataset size, vector dimensions, and "
        "supported projection algorithms."
    ),
)
async def get_explorer_stats() -> APIResponse[dict[str, Any]]:
    """Return dataset telemetry for embedding explorer."""
    svc = _get_service()
    stats = svc.get_explorer_stats()
    return success_response(data=stats, message="Embedding space statistics retrieved successfully")


@router.post(
    "/explorer/project",
    response_model=APIResponse[ExplorerDatasetPayload],
    summary="Generate Embedding Space Projection & Clustering",
    description=(
        "Computes 2D/3D PCA or t-SNE projection, K-Means clustering, and anomaly "
        "outlier scores over Visual Memory vectors."
    ),
)
async def generate_projection(
    req: ProjectionRequest,
) -> APIResponse[ExplorerDatasetPayload]:
    """Execute dimensionality reduction projection, clustering, and outlier detection."""
    svc = _get_service()
    try:
        payload = svc.generate_projection(req)
        method_str = payload.reduction_meta.method.value.upper()
        msg = f"Projected {payload.total_points} point(s) using {method_str}"
        return success_response(data=payload, message=msg)
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc
