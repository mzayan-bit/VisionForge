"""Visual Search REST API Endpoints."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.search.engine import SearchResponsePayload, SimilarityMetric
from visionforge.search.history import SearchHistoryRecord
from visionforge.search.service import VisualSearchService, get_visual_search_service

router = APIRouter(tags=["Visual Search"])

MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB limit


class RecordSearchRequest(BaseModel):
    """Request payload for searching using an existing Visual Memory record ID."""

    record_id: str = Field(description="Record ID of the visual memory query item")
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum top matches (1..100)")
    metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Distance metric for ranking"
    )
    threshold: float = Field(
        default=0.0, ge=0.0, le=0.95, description="Minimum similarity threshold (0.0..0.95)"
    )


class VectorSearchRequest(BaseModel):
    """Request payload for searching directly via raw query vector."""

    vector: list[float] = Field(description="Dense 768-dimensional query vector")
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum top matches (1..100)")
    metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Distance metric for ranking"
    )
    threshold: float = Field(
        default=0.0, ge=0.0, le=0.95, description="Minimum similarity threshold (0.0..0.95)"
    )


def _get_service() -> VisualSearchService:
    return get_visual_search_service()


@router.post(
    "/search/image",
    response_model=APIResponse[SearchResponsePayload],
    summary="Visual Search by Uploaded Image",
    description=(
        "Uploads a query image, extracts embedding via SigLIP, and performs "
        "similarity search over Visual Memory."
    ),
)
async def search_by_image(
    file: UploadFile = File(..., description="Query image file"),
    top_k: int = Form(5, ge=1, le=100, description="Top K nearest matches"),
    metric: SimilarityMetric = Form(SimilarityMetric.COSINE, description="Similarity metric"),
    threshold: float = Form(0.0, ge=0.0, le=0.95, description="Minimum similarity threshold"),
) -> APIResponse[SearchResponsePayload]:
    """Execute similarity search using query image upload."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content-type '{file.content_type}'. Must be an image file.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded query image file is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded image exceeds 20MB limit.")

    svc = _get_service()
    try:
        results = await svc.search_by_image(
            image_bytes=image_bytes,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
        )
        msg = f"Found {results.returned_count} match(es) in {results.candidate_count} candidates"
        return success_response(data=results, message=msg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(exc)}") from exc


@router.post(
    "/search/record",
    response_model=APIResponse[SearchResponsePayload],
    summary="Visual Search by Visual Memory Record ID",
    description="Performs similarity search using an existing Visual Memory record ID as query.",
)
async def search_by_record(
    req: RecordSearchRequest,
) -> APIResponse[SearchResponsePayload]:
    """Execute similarity search using existing Visual Memory record ID."""
    svc = _get_service()
    try:
        results = svc.search_by_record(
            record_id=req.record_id,
            top_k=req.top_k,
            metric=req.metric,
            threshold=req.threshold,
        )
        msg = f"Found {results.returned_count} match(es) in {results.candidate_count} candidates"
        return success_response(data=results, message=msg)
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post(
    "/search/vector",
    response_model=APIResponse[SearchResponsePayload],
    summary="Visual Search by Vector",
    description="Performs similarity search over Visual Memory using a 768D query vector.",
)
async def search_by_vector(
    req: VectorSearchRequest,
) -> APIResponse[SearchResponsePayload]:
    """Execute similarity search using direct 768D query vector."""
    if len(req.vector) != 768:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid query vector dimension {len(req.vector)}. Expected 768D.",
        )

    svc = _get_service()
    try:
        results = svc.search_by_vector(
            query_vector=req.vector,
            top_k=req.top_k,
            metric=req.metric,
            threshold=req.threshold,
        )
        msg = f"Found {results.returned_count} match(es) in {results.candidate_count} candidates"
        return success_response(data=results, message=msg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(exc)}") from exc


@router.get(
    "/search/history",
    response_model=APIResponse[list[SearchHistoryRecord]],
    summary="Get Visual Search History Logs",
    description="Returns historical visual search transaction execution logs.",
)
async def get_search_history(
    limit: int = 50, offset: int = 0
) -> APIResponse[list[SearchHistoryRecord]]:
    """Return paginated search history records."""
    svc = _get_service()
    history = svc.get_search_history(limit=limit, offset=offset)
    return success_response(
        data=history, message=f"Retrieved {len(history)} search history log(s)"
    )
