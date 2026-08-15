"""Visual Search REST API Endpoints."""

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.search.engine import SearchResponsePayload, SimilarityMetric
from visionforge.search.history import SearchHistoryRecord
from visionforge.search.schemas import (
    NearDuplicateResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
    VisualAsset,
    VisualAssetType,
)
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


class IndexRunRequest(BaseModel):
    """Request payload to index all visual assets from a video inference run."""

    run_id: str = Field(description="Target VideoInferenceRun ID to index")


class FrameSearchRequest(BaseModel):
    """Request payload to search similar assets to a video frame."""

    video_id: str = Field(description="Target video ID")
    timestamp_sec: float = Field(default=0.0, description="Timestamp in video")
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.0, ge=0.0, le=0.98)
    filter_asset_types: list[VisualAssetType] | None = None


class ObjectSearchRequest(BaseModel):
    """Request payload to search similar assets to a track object crop."""

    run_id: str = Field(description="Target VideoInferenceRun ID")
    track_id: int = Field(description="Target Track ID")
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.0, ge=0.0, le=0.98)
    filter_asset_types: list[VisualAssetType] | None = None


class EventSearchRequest(BaseModel):
    """Request payload to search similar assets to an event moment."""

    event_id: str = Field(description="Target TemporalEvent ID")
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.0, ge=0.0, le=0.98)


def _get_service() -> VisualSearchService:
    return get_visual_search_service()


# ─── Unified Visual Search Routes ──────────────────────────────────────


@router.post(
    "/search/unified",
    response_model=APIResponse[UnifiedSearchResponse],
    summary="Unified Visual Search",
    description="Execute multi-modal visual similarity search across images, frames, object crops, moments, and dataset samples.",
)
async def search_unified(
    req: UnifiedSearchRequest,
) -> APIResponse[UnifiedSearchResponse]:
    """Execute unified visual search."""
    svc = _get_service()
    try:
        results = await svc.search_unified(req=req)
        msg = f"Found {results.returned_count} match(es) across {results.candidate_count} indexed visual assets"
        return success_response(data=results, message=msg)
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post(
    "/search/frame",
    response_model=APIResponse[UnifiedSearchResponse],
    summary="Search by Video Frame",
    description="Search for visually similar frames, images, and samples to a specific video timestamp.",
)
async def search_by_frame(
    req: FrameSearchRequest,
) -> APIResponse[UnifiedSearchResponse]:
    """Search similar visual assets using video frame timestamp."""
    svc = _get_service()
    u_req = UnifiedSearchRequest(
        query_type=VisualAssetType.FRAME,
        video_id=req.video_id,
        timestamp_sec=req.timestamp_sec,
        top_k=req.top_k,
        threshold=req.threshold,
        filter_asset_types=req.filter_asset_types,
    )
    try:
        results = await svc.search_unified(req=u_req)
        return success_response(data=results, message=f"Found {results.returned_count} matches for frame @ {req.timestamp_sec:.1f}s")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/search/object",
    response_model=APIResponse[UnifiedSearchResponse],
    summary="Search by Object Crop",
    description="Search for visually similar object appearances to a detected track.",
)
async def search_by_object(
    req: ObjectSearchRequest,
) -> APIResponse[UnifiedSearchResponse]:
    """Search similar visual assets using a track object appearance."""
    svc = _get_service()
    u_req = UnifiedSearchRequest(
        query_type=VisualAssetType.OBJECT_CROP,
        run_id=req.run_id,
        track_id=req.track_id,
        top_k=req.top_k,
        threshold=req.threshold,
        filter_asset_types=req.filter_asset_types,
    )
    try:
        results = await svc.search_unified(req=u_req)
        return success_response(data=results, message=f"Found {results.returned_count} matches for Track #{req.track_id}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/search/event",
    response_model=APIResponse[UnifiedSearchResponse],
    summary="Search by Event Moment",
    description="Search for visually similar moments/events across videos using event evidence.",
)
async def search_by_event(
    req: EventSearchRequest,
) -> APIResponse[UnifiedSearchResponse]:
    """Search similar video moments using event evidence frame."""
    svc = _get_service()
    u_req = UnifiedSearchRequest(
        query_type=VisualAssetType.EVENT_FRAME,
        event_id=req.event_id,
        top_k=req.top_k,
        threshold=req.threshold,
    )
    try:
        results = await svc.search_unified(req=u_req)
        return success_response(data=results, message=f"Found {results.returned_count} matches for Event {req.event_id}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/search/duplicates",
    response_model=APIResponse[NearDuplicateResponse],
    summary="Find Near-Duplicate Candidates",
    description="Identifies pairs of visually near-identical assets (similarity >= threshold).",
)
def find_near_duplicates(
    threshold: float = Query(default=0.95, ge=0.80, le=0.99),
    asset_type: VisualAssetType | None = Query(default=None),
    dataset_id: str | None = Query(default=None),
) -> APIResponse[NearDuplicateResponse]:
    """Discover near-duplicate candidate pairs."""
    svc = _get_service()
    resp = svc.find_near_duplicates(
        threshold=threshold,
        filter_asset_type=asset_type,
        filter_dataset_id=dataset_id,
    )
    return success_response(
        data=resp,
        message=f"Discovered {resp.duplicate_pairs_found} candidate near-duplicate pair(s)",
    )


@router.get(
    "/search/assets",
    response_model=APIResponse[list[VisualAsset]],
    summary="List Searchable Visual Assets",
    description="Returns registered searchable visual assets catalog.",
)
def list_searchable_assets(
    asset_type: VisualAssetType | None = Query(default=None),
    video_id: str | None = Query(default=None),
    dataset_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[VisualAsset]]:
    """List searchable assets."""
    svc = _get_service()
    assets = svc.list_assets(
        asset_type=asset_type,
        video_id=video_id,
        dataset_id=dataset_id,
        limit=limit,
        offset=offset,
    )
    return success_response(data=assets, message=f"Retrieved {len(assets)} visual asset(s)")


@router.post(
    "/search/index-run",
    response_model=APIResponse[dict[str, int]],
    summary="Index Video Run Assets",
    description="Indexes frames, track object appearances, and event evidence into visual search memory.",
)
def index_video_run(req: IndexRunRequest) -> APIResponse[dict[str, int]]:
    """Index all visual assets from a video inference run."""
    svc = _get_service()
    try:
        cnt = svc.index_video_run_assets(req.run_id)
        return success_response(data={"indexed_assets": cnt}, message=f"Indexed {cnt} visual assets from run '{req.run_id}'")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ─── Backward-Compatible Existing Endpoints ───────────────────────────


@router.post(
    "/search/image",
    response_model=APIResponse[SearchResponsePayload],
    summary="Visual Search by Uploaded Image",
    description="Uploads a query image, extracts embedding via SigLIP, and performs similarity search.",
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
