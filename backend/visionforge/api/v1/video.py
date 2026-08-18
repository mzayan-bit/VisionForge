"""VisionForge Video Intelligence API Routes."""

import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from visionforge.video.schemas import (
    FrameSamplingConfig,
    FrameSamplingMode,
    TemporalAnalytics,
    Track,
    TrajectoryPoint,
    VideoComparisonResult,
    VideoInferenceRun,
    VideoMetadata,
    VideoSession,
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
    custom_stride: int = Field(
        default=2, ge=1, le=60, description="Custom sampling stride interval"
    )


class CreateVideoSessionRequest(BaseModel):
    video_id: str = Field(description="Target video asset ID")
    model_version: str = Field(default="1.0.0", description="Model version tag")


class CompareVideosRequest(BaseModel):
    video_a_id: str = Field(description="First video ID")
    video_b_id: str = Field(description="Second video ID")


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


@router.get(
    "/videos",
    response_model=list[VideoMetadata],
    summary="List registered video assets",
)
def list_videos() -> list[VideoMetadata]:
    """List all registered video assets."""
    service = get_video_intelligence_service()
    return service.list_videos()


@router.get(
    "/stream/{video_id}",
    summary="Stream raw video asset file",
)
def stream_video(video_id: str):
    """Stream raw video asset file for native browser video playback."""
    service = get_video_intelligence_service()
    if video_id not in service._videos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video asset '{video_id}' not found.",
        )
    meta = service._videos[video_id]

    # 1. Check uploads directory
    upload_path = service._storage_dir / "uploads" / meta.filename
    if upload_path.is_file():
        return FileResponse(str(upload_path), media_type="video/mp4", filename=meta.filename)

    # 2. Check direct storage directory
    direct_path = service._storage_dir / meta.filename
    if direct_path.is_file():
        return FileResponse(str(direct_path), media_type="video/mp4", filename=meta.filename)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Physical video file for '{video_id}' not found on server.",
    )


@router.post(
    "/sessions",
    response_model=VideoSession,
    status_code=status.HTTP_201_CREATED,
    summary="Create video analysis session",
)
def create_video_session(payload: CreateVideoSessionRequest) -> VideoSession:
    """Create a tracked video session with lineage tracking."""
    service = get_video_intelligence_service()
    try:
        return service.create_video_session(
            video_id=payload.video_id,
            model_version=payload.model_version,
        )
    except VideoValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/sessions",
    response_model=list[VideoSession],
    summary="List video analysis sessions",
)
def list_video_sessions() -> list[VideoSession]:
    """List all recorded video sessions."""
    service = get_video_intelligence_service()
    return service.list_video_sessions()


@router.get(
    "/sessions/{session_id}",
    response_model=VideoSession,
    summary="Get video session detail",
)
def get_video_session(session_id: str) -> VideoSession:
    """Retrieve a single video session record."""
    service = get_video_intelligence_service()
    s = service.get_video_session(session_id)
    if not s:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found"
        )
    return s


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
        cfg = FrameSamplingConfig(
            mode=payload.sampling_mode,
            sample_interval=payload.custom_stride,
        )
        return service.run_video_tracking(
            video_id=payload.video_id,
            model_id=payload.model_id,
            sampling_config=cfg,
        )
    except VideoValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/runs",
    response_model=list[VideoInferenceRun],
    summary="List video inference tracking runs",
)
def list_video_runs(
    video_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[VideoInferenceRun]:
    """List historical video inference tracking runs."""
    service = get_video_intelligence_service()
    runs = service.list_runs(video_id=video_id)
    return runs[offset : offset + limit]


@router.get(
    "/runs/{run_id}",
    response_model=VideoInferenceRun,
    summary="Get single video tracking run record",
)
def get_video_run(run_id: str) -> VideoInferenceRun:
    """Retrieve complete video run descriptor, tracks, and analytics."""
    service = get_video_intelligence_service()
    try:
        return service.get_run(run_id)
    except VideoRunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/tracks",
    response_model=list[Track],
    summary="Get tracked objects for a video run",
)
def get_run_tracks(
    run_id: str,
    class_name: str | None = Query(default=None),
    min_duration: float | None = Query(default=None),
) -> list[Track]:
    """Retrieve list of Track records identified in a video tracking run."""
    service = get_video_intelligence_service()
    run = service.get_run(run_id)
    tracks = run.tracks
    if class_name:
        tracks = [t for t in tracks if t.class_name.lower() == class_name.lower()]
    if min_duration is not None:
        tracks = [t for t in tracks if t.visibility_duration_sec >= min_duration]
    return tracks


@router.get(
    "/runs/{run_id}/tracks/{track_id}",
    response_model=Track,
    summary="Get single track detail and full trajectory",
)
def get_track_detail(run_id: str, track_id: int) -> Track:
    """Retrieve specific track details, trajectory points, and region interactions."""
    service = get_video_intelligence_service()
    run = service.get_run(run_id)
    for t in run.tracks:
        if t.track_id == track_id:
            return t
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Track #{track_id} not found in run '{run_id}'.",
    )


@router.get(
    "/runs/{run_id}/trajectories",
    response_model=dict[str, list[TrajectoryPoint]],
    summary="Get all trajectories for a video run",
)
def get_run_trajectories(run_id: str) -> dict[str, list[TrajectoryPoint]]:
    """Retrieve mapping of track ID to chronological trajectory points."""
    service = get_video_intelligence_service()
    run = service.get_run(run_id)
    return {str(t.track_id): t.trajectory for t in run.tracks}


@router.get(
    "/runs/{run_id}/analytics",
    response_model=TemporalAnalytics,
    summary="Get temporal analytics telemetry",
)
def get_run_analytics(run_id: str) -> TemporalAnalytics:
    """Retrieve aggregate temporal metrics, track duration distributions, and dwell times."""
    service = get_video_intelligence_service()
    run = service.get_run(run_id)
    return run.analytics


@router.get(
    "/runs/{run_id}/export",
    summary="Export track trajectories as CSV",
)
def export_run_csv(run_id: str) -> dict[str, Any]:
    """Export all track trajectories for a video run as CSV text."""
    service = get_video_intelligence_service()
    run = service.get_run(run_id)
    lines = [
        "run_id,video_id,track_id,class_name,frame_index,timestamp_sec,x_center_px,y_center_px,width_px,height_px"
    ]
    for t in run.tracks:
        for pt in t.trajectory:
            lines.append(
                f"{run.run_id},{run.video_id},{t.track_id},{t.class_name},{pt.frame_index},{pt.timestamp_sec},{pt.x_center_px},{pt.y_center_px},{pt.width_px},{pt.height_px}"
            )
    csv_text = "\n".join(lines)
    return {"data": csv_text, "filename": f"{run_id}_trajectories.csv"}


@router.post(
    "/compare",
    response_model=VideoComparisonResult,
    summary="Compare two video analysis runs side-by-side",
)
def compare_videos(payload: CompareVideosRequest) -> VideoComparisonResult:
    """Compare tracks, detections, and duration statistics between two video runs."""
    service = get_video_intelligence_service()
    try:
        return service.compare_videos(payload.video_a_id, payload.video_b_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
