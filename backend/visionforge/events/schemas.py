"""VisionForge Temporal Event Intelligence Data Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Supported observable temporal event types."""

    TRACK_STARTED = "TRACK_STARTED"
    TRACK_ENDED = "TRACK_ENDED"
    OBJECT_ENTERED_REGION = "OBJECT_ENTERED_REGION"
    OBJECT_LEFT_REGION = "OBJECT_LEFT_REGION"
    OBJECT_STAYED_IN_REGION = "OBJECT_STAYED_IN_REGION"
    OBJECT_DWELLED = "OBJECT_DWELLED"
    OBJECT_APPEARED = "OBJECT_APPEARED"
    OBJECT_DISAPPEARED = "OBJECT_DISAPPEARED"
    PROLONGED_PROXIMITY = "PROLONGED_PROXIMITY"
    TRACK_CROSSING_REGION = "TRACK_CROSSING_REGION"
    OBJECT_STOPPED = "OBJECT_STOPPED"
    OBJECT_MOVED = "OBJECT_MOVED"
    OBJECT_COUNT_CHANGED = "OBJECT_COUNT_CHANGED"
    OBJECTS_BECAME_CLOSE = "OBJECTS_BECAME_CLOSE"
    OBJECTS_MOVED_APART = "OBJECTS_MOVED_APART"


class EventReliability(StrEnum):
    """Derivation reliability level based on trajectory density and observation persistence."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RegionShape(StrEnum):
    """Shape geometry of Region of Interest (ROI)."""

    RECTANGLE = "RECTANGLE"
    POLYGON = "POLYGON"


class CoordinateSystem(StrEnum):
    """Spatial coordinate reference system for Region of Interest."""

    PIXEL = "PIXEL"
    NORMALIZED = "NORMALIZED"


class RegionOfInterest(BaseModel):
    """Spatial Region of Interest (ROI) defined on video canvas."""

    region_id: str = Field(description="Unique region identifier ('reg_...')")
    video_id: str = Field(description="Source video asset ID")
    name: str = Field(description="Human-readable region name (e.g. 'Loading Zone A', 'Restricted Corridor')")
    shape_type: RegionShape = Field(default=RegionShape.RECTANGLE, description="Shape geometry format")
    coordinates: list[list[float]] = Field(
        description="Vertex coordinates [[x1, y1], [x2, y2], ...] or [[x_min, y_min], [x_max, y_max]]"
    )
    coordinate_system: CoordinateSystem = Field(
        default=CoordinateSystem.PIXEL, description="Pixel or normalized coordinate space"
    )
    color: str = Field(default="#3b82f6", description="UI overlay stroke color")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Creation timestamp"
    )


class EventRuleConfig(BaseModel):
    """Configurable threshold parameters governing temporal event detection."""

    dwell_threshold_sec: float = Field(
        default=3.0, ge=0.5, le=60.0, description="Dwell duration threshold in seconds"
    )
    stopped_speed_threshold_px_s: float = Field(
        default=15.0, ge=1.0, le=100.0, description="Speed threshold below which object is considered STOPPED (px/s)"
    )
    proximity_threshold_px: float = Field(
        default=100.0, ge=10.0, le=500.0, description="Proximity distance threshold between objects (px)"
    )
    separation_threshold_px: float = Field(
        default=180.0, ge=20.0, le=800.0, description="Separation distance threshold between objects (px)"
    )
    debounce_frames: int = Field(
        default=3, ge=1, le=15, description="Debounce frame hysteresis window to prevent jitter"
    )


class EventEvidence(BaseModel):
    """Visual verification evidence frames for a detected temporal event."""

    event_id: str = Field(description="Target event ID")
    frame_before_idx: int = Field(description="Frame index immediately before event onset")
    event_frame_idx: int = Field(description="Key event occurrence frame index")
    frame_after_idx: int = Field(description="Frame index immediately after event completion")
    start_timestamp_sec: float = Field(default=0.0, description="Exact start time in seconds")
    representative_timestamp_sec: float = Field(default=0.0, description="Peak event timestamp")
    end_timestamp_sec: float = Field(default=0.0, description="Exact end time in seconds")
    highlight_track_ids: list[int] = Field(description="Track IDs to highlight on evidence canvas")
    highlight_region_id: str | None = Field(default=None, description="Associated Region ROI ID")
    trigger_rule: str = Field(default="", description="Deterministic rule or condition that triggered event")
    snapshot_notes: str = Field(description="Explanatory visual verification summary")
    evidence_thumbnail_uri: str | None = Field(default=None, description="Artifact image path")


class TemporalEvent(BaseModel):
    """Clean, explainable temporal event derived from observable visual trajectories."""

    event_id: str = Field(description="Unique event identifier ('evt_...')")
    run_id: str = Field(description="Source VideoInferenceRun ID")
    video_id: str = Field(description="Source video asset ID")
    event_type: EventType = Field(description="Observable event classification")
    start_timestamp_sec: float = Field(description="Event start timestamp in seconds")
    end_timestamp_sec: float = Field(description="Event end timestamp in seconds")
    duration_sec: float = Field(description="Event duration in seconds")
    source_track_ids: list[int] = Field(description="List of persistent Track IDs involved")
    source_frame_range: list[int] = Field(description="[start_frame, end_frame] index range")
    reliability: EventReliability = Field(default=EventReliability.HIGH, description="Event derivation reliability")
    event_params: dict[str, Any] = Field(
        default_factory=dict, description="Structured parameters (region_id, distance_px, speed_px_s, etc.)"
    )
    description: str = Field(description="Human-readable explainable event narrative")
    trigger_rule: str = Field(
        default="", description="Deterministic rule or evidence basis for event generation"
    )
    evidence: EventEvidence | None = Field(
        default=None, description="Linked visual evidence frame metadata"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Detection ISO timestamp"
    )


class EventAnalytics(BaseModel):
    """Summary telemetry and breakdown of temporal events across a video run."""

    total_events: int = Field(description="Total events detected")
    events_by_type: dict[str, int] = Field(
        default_factory=dict, description="Event count breakdown by EventType"
    )
    events_by_class: dict[str, int] = Field(
        default_factory=dict, description="Event count breakdown by object class"
    )
    events_by_region: dict[str, int] = Field(
        default_factory=dict, description="Event count breakdown by Region ROI"
    )
    avg_dwell_time_sec: float = Field(description="Average dwell event duration in seconds")
    longest_dwell_sec: float = Field(description="Longest single dwell duration in seconds")
    proximity_events_count: int = Field(description="Total proximity interaction events count")
    timeline: list[dict[str, Any]] = Field(
        default_factory=list, description="Chronological event stream entries"
    )


class SceneSummary(BaseModel):
    """Structured deterministic summary of a video scene's temporal events."""

    video_id: str = Field(description="Video asset ID")
    run_id: str = Field(description="VideoInferenceRun ID")
    duration_sec: float = Field(description="Total video duration in seconds")
    total_tracks: int = Field(description="Total objects tracked")
    total_regions: int = Field(description="Total active ROI regions")
    total_events: int = Field(description="Total events detected")
    most_active_region: str = Field(description="Region ROI with highest event density")
    longest_dwell_event: dict[str, Any] | None = Field(
        default=None, description="Details of the longest single dwell event"
    )
