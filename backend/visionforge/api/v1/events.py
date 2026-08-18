"""VisionForge Temporal Event Intelligence API Routes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.events.schemas import (
    CoordinateSystem,
    EventAnalytics,
    EventEvidence,
    EventRuleConfig,
    EventType,
    RegionOfInterest,
    RegionShape,
    SceneSummary,
    TemporalEvent,
)
from visionforge.events.service import (
    EventNotFoundError,
    RegionNotFoundError,
    get_temporal_event_service,
)

logger = logging.getLogger("visionforge.api.v1.events")

router = APIRouter(prefix="/events", tags=["Temporal Event Intelligence"])


class CreateRegionRequest(BaseModel):
    video_id: str = Field(description="Target video asset ID")
    name: str = Field(description="Human-readable region name")
    coordinates: list[list[float]] = Field(description="Vertices coordinates")
    shape_type: RegionShape = Field(
        default=RegionShape.RECTANGLE, description="Region geometry shape"
    )
    coordinate_system: CoordinateSystem = Field(
        default=CoordinateSystem.PIXEL, description="Coordinate system"
    )
    color: str = Field(default="#3b82f6", description="Color stroke hex code")


class UpdateRegionRequest(BaseModel):
    name: str | None = Field(default=None, description="Updated human-readable region name")
    coordinates: list[list[float]] | None = Field(
        default=None, description="Updated vertices coordinates"
    )
    shape_type: RegionShape | None = Field(
        default=None, description="Updated shape geometry format"
    )
    color: str | None = Field(default=None, description="Updated color stroke hex code")


class GenerateEventsRequest(BaseModel):
    run_id: str = Field(description="Target VideoInferenceRun ID")
    config: EventRuleConfig | None = Field(
        default=None, description="Optional rule configuration thresholds"
    )


@router.post(
    "/regions",
    response_model=RegionOfInterest,
    status_code=status.HTTP_201_CREATED,
    summary="Create Region of Interest (ROI)",
)
def create_region(payload: CreateRegionRequest) -> RegionOfInterest:
    """Create and store a spatial Region of Interest for a video asset."""
    service = get_temporal_event_service()
    return service.create_region(
        video_id=payload.video_id,
        name=payload.name,
        coordinates=payload.coordinates,
        shape_type=payload.shape_type,
        coordinate_system=payload.coordinate_system,
        color=payload.color,
    )


@router.get(
    "/regions",
    response_model=list[RegionOfInterest],
    summary="List Regions of Interest",
)
def list_regions(
    video_id: str | None = Query(default=None, description="Filter regions by video asset ID"),
) -> list[RegionOfInterest]:
    """Retrieve list of defined regions of interest."""
    service = get_temporal_event_service()
    return service.list_regions(video_id=video_id)


@router.get(
    "/regions/{region_id}",
    response_model=RegionOfInterest,
    summary="Get Region of Interest detail",
)
def get_region(region_id: str) -> RegionOfInterest:
    """Get single region of interest by ID."""
    service = get_temporal_event_service()
    try:
        return service.get_region(region_id)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put(
    "/regions/{region_id}",
    response_model=RegionOfInterest,
    summary="Update Region of Interest",
)
def update_region(region_id: str, payload: UpdateRegionRequest) -> RegionOfInterest:
    """Update geometry, name, or styling of an existing region."""
    service = get_temporal_event_service()
    try:
        return service.update_region(
            region_id=region_id,
            name=payload.name,
            coordinates=payload.coordinates,
            shape_type=payload.shape_type,
            color=payload.color,
        )
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/regions/{region_id}/duplicate",
    response_model=RegionOfInterest,
    status_code=status.HTTP_201_CREATED,
    summary="Duplicate Region of Interest",
)
def duplicate_region(
    region_id: str,
    offset_px: float = Query(default=30.0, description="Pixel offset applied to duplicated region"),
) -> RegionOfInterest:
    """Duplicate an existing region with a distinct ID, name, and spatial offset."""
    service = get_temporal_event_service()
    try:
        return service.duplicate_region(region_id=region_id, offset_px=offset_px)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete(
    "/regions/{region_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Region of Interest",
)
def delete_region(region_id: str) -> None:
    """Delete a region of interest."""
    service = get_temporal_event_service()
    try:
        service.delete_region(region_id)
    except RegionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/generate",
    response_model=list[TemporalEvent],
    status_code=status.HTTP_201_CREATED,
    summary="Detect and generate temporal events for a video run",
)
def generate_events(payload: GenerateEventsRequest) -> list[TemporalEvent]:
    """Execute rule-based temporal event detection pipeline on pre-computed video run tracks."""
    service = get_temporal_event_service()
    try:
        return service.generate_events_for_run(run_id=payload.run_id, config=payload.config)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}",
    response_model=list[TemporalEvent],
    summary="Query temporal events for a video run",
)
def get_events_for_run(
    run_id: str,
    event_type: EventType | None = Query(default=None, description="Filter by EventType"),
    track_id: int | None = Query(default=None, description="Filter by track ID"),
    region_id: str | None = Query(default=None, description="Filter by region ID"),
) -> list[TemporalEvent]:
    """Retrieve chronological event stream for a video run with filtering."""
    service = get_temporal_event_service()
    return service.get_events_for_run(
        run_id=run_id, event_type=event_type, track_id=track_id, region_id=region_id
    )


@router.get(
    "/{event_id}",
    response_model=TemporalEvent,
    summary="Get single temporal event detail",
)
def get_event_detail(event_id: str) -> TemporalEvent:
    """Retrieve complete metadata and parameters for a single event."""
    service = get_temporal_event_service()
    try:
        return service.get_event_detail(event_id)
    except EventNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{event_id}/evidence",
    response_model=EventEvidence,
    summary="Get visual verification evidence frames for an event",
)
def get_event_evidence(event_id: str) -> EventEvidence:
    """Retrieve 3-frame evidence indices and highlight parameters for visual verification."""
    service = get_temporal_event_service()
    try:
        return service.get_event_evidence(event_id)
    except EventNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/analytics",
    response_model=EventAnalytics,
    summary="Get event analytics and chronological timeline",
)
def get_event_analytics(run_id: str) -> EventAnalytics:
    """Retrieve aggregate event breakdown statistics and timeline."""
    service = get_temporal_event_service()
    return service.get_event_analytics(run_id)


@router.get(
    "/runs/{run_id}/summary",
    response_model=SceneSummary,
    summary="Get deterministic scene summary",
)
def get_scene_summary(run_id: str) -> SceneSummary:
    """Retrieve structured scene summary for a video run."""
    service = get_temporal_event_service()
    return service.get_scene_summary(run_id)


@router.get(
    "/runs/{run_id}/export",
    summary="Export event stream as CSV",
)
def export_events_csv(run_id: str) -> dict[str, str]:
    """Export all events for a video run in CSV format."""
    service = get_temporal_event_service()
    csv_data = service.export_events_csv(run_id)
    return {"run_id": run_id, "format": "csv", "data": csv_data}
