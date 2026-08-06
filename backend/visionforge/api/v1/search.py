"""Visual Search REST API Endpoints."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.search.engine import (
    SearchResponsePayload,
    SimilarityMetric,
    VisualSearchEngine,
    get_visual_search_engine,
)

router = APIRouter(tags=["Visual Search"])


class VectorSearchRequest(BaseModel):
    """Request payload for searching directly via raw query vector."""

    vector: list[float] = Field(description="Dense 768-dimensional query vector")
    top_k: int = Field(default=5, description="Maximum top nearest matches to return")
    metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Distance metric for ranking"
    )
    threshold: float = Field(
        default=0.0, description="Minimum similarity threshold cutoff (0.0 to 1.0)"
    )


def _get_search_engine() -> VisualSearchEngine:
    return get_visual_search_engine()


@router.post(
    "/search/image",
    response_model=APIResponse[SearchResponsePayload],
    summary="Visual Search by Image",
    description=(
        "Uploads a query image, extracts embedding via SigLIP, and performs "
        "vectorized similarity search over Visual Memory."
    ),
)
async def search_by_image(
    file: UploadFile = File(..., description="Query image file"),
    top_k: int = Form(5, description="Top K nearest matches"),
    metric: SimilarityMetric = Form(SimilarityMetric.COSINE, description="Similarity metric"),
    threshold: float = Form(0.0, description="Minimum similarity threshold"),
) -> APIResponse[SearchResponsePayload]:
    """Execute similarity search using query image upload."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Query file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded query file is empty.")

    engine = _get_search_engine()
    try:
        results = await engine.search_by_image(
            image_bytes=image_bytes,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
        )
        msg = f"Found {len(results.results)} match(es) across {results.candidate_count} candidates"
        return success_response(data=results, message=msg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(exc)}") from exc


@router.post(
    "/search/vector",
    response_model=APIResponse[SearchResponsePayload],
    summary="Visual Search by Vector",
    description="Performs vectorized similarity search over Visual Memory using query vector.",
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

    engine = _get_search_engine()
    results = engine.search_by_vector(
        query_vector=req.vector,
        top_k=req.top_k,
        metric=req.metric,
        threshold=req.threshold,
    )
    msg = f"Found {len(results.results)} match(es) across {results.candidate_count} candidates"
    return success_response(data=results, message=msg)
