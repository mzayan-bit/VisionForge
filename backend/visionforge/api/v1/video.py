"""VisionForge Video Intelligence API Routes."""

import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from visionforge.video.schemas import (
    FrameSamplingMode,
    TemporalAnalytics,
    Track,
    VideoInferenceRun,
    VideoMetadata,
)
from visionforge.video.service import (
    VideoRunNotFoundError,
    VideoValidationError,
    get_video_intelligence_service,
)

logger = logging.getLogger("visionforge.api.v1.video")

router = APIRouter(prefix="/video", tags=["Video Intelligence"])


class CreateVideoRunRequest(BaseModel):
    video_id: str = Field(description="Target video asset ID")
    model_id: str = Field(default="yolo11s.pt", description="Object detection model ID")
    sampling_mode: FrameSamplingMode = Field(
        default=FrameSamplingMode.EVERY_2ND_FRAME, description="Frame sampling stride mode"
    )
    custom_stride: int = Field(default=2, ge=1, le=60, description="Custom sampling stride interval")


@router.post(
    "/upload",
    response_model=VideoMetadata,
    status_code=status.HTTP_201_CREATED,
    summary="Upload video asset and extract metadata",
)
async def upload_video(file: UploadFile = File(...)) -> VideoMetadata:
    """Upload video file, perform validation checks, and extract video metadata telemetry."""
    service = get_video_intelligence_service()
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename required.")

    # Save uploaded file to temp directory
    temp_dir = service._storage_dir / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    content = await file.read()
    temp_path.write_bytes(content)

    try:
        return service.register_video(str(temp_path))
    except VideoValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/metadata/{video_id}",
    response_model=VideoMetadata,
    summary="Get video asset metadata",
)
def get_video_metadata(video_id: str) -> VideoMetadata:
    """Retrieve metadata telemetry for a specific video asset ID."""
    service = get_video_intelligence_service()
    return service.get_video_metadata(video_id)


@router.post(
    "/runs",
    response_model=VideoInferenceRun,
    status_code=status.HTTP_201_CREATED,
    summary="Create & execute video detection and ByteTrack tracking run",
)
def create_video_run(payload: CreateVideoRunRequest) -> VideoInferenceRun:
    """Execute video object detection, ByteTrack tracking, and temporal analytics pipeline."""
    service = get_video_intelligence_service()
    try:
        return service.execute_video_inference(
            video_id=payload.video_id,
            model_id=payload.model_id,
            sampling_mode=payload.sampling_mode,
            custom_stride=payload.custom_stride,
        )
    except VideoValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/runs",
    response_model=list[VideoInferenceRun],
    summary="List video inference tracking runs",
)
def list_video_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[VideoInferenceRun]:
    """Retrieve paginated list of video inference runs."""
    service = get_video_intelligence_service()
    return service.list_runs(limit=limit, offset=offset)


@router.get(
    "/runs/{run_id}",
    response_model=VideoInferenceRun,
    summary="Get single video run detail",
)
def get_video_run(run_id: str) -> VideoInferenceRun:
    """Retrieve complete telemetry, tracks, and analytics for a video inference run."""
    service = get_video_intelligence_service()
    try:
        return service.get_run(run_id)
    except VideoRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/tracks",
    response_model=list[Track],
    summary="Query tracks for a video run",
)
def get_video_tracks(
    run_id: str,
    class_name: str | None = Query(default=None, description="Filter tracks by class name"),
    min_duration_sec: float | None = Query(
        default=None, ge=0.0, description="Minimum track duration"
    ),
) -> list[Track]:
    """Retrieve persistent tracks for a video run with filtering."""
    service = get_video_intelligence_service()
    try:
        run = service.get_run(run_id)
        tracks = run.tracks

        if class_name:
            tracks = [t for t in tracks if t.class_name.lower() == class_name.lower()]
        if min_duration_sec is not None:
            tracks = [t for t in tracks if t.visibility_duration_sec >= min_duration_sec]

        return tracks
    except VideoRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/analytics",
    response_model=TemporalAnalytics,
    summary="Get temporal analytics for a video run",
)
def get_video_analytics(run_id: str) -> TemporalAnalytics:
    """Retrieve temporal analytics time series and aggregate statistics."""
    service = get_video_intelligence_service()
    try:
        run = service.get_run(run_id)
        return run.analytics
    except VideoRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/export",
    summary="Export track trajectories as CSV",
)
def export_video_run_csv(run_id: str) -> dict[str, str]:
    """Export all trajectory points for a video run in CSV format."""
    service = get_video_intelligence_service()
    try:
        csv_data = service.export_run_csv(run_id)
        return {"run_id": run_id, "format": "csv", "data": csv_data}
    except VideoRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
