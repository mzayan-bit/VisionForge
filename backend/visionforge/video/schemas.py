"""VisionForge Video Intelligence Data Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FrameSamplingMode(StrEnum):
    """Supported frame sampling modes."""

    EVERY_FRAME = "EVERY_FRAME"
    EVERY_2ND_FRAME = "EVERY_2ND_FRAME"
    EVERY_5TH_FRAME = "EVERY_5TH_FRAME"
    EVERY_10TH_FRAME = "EVERY_10TH_FRAME"
    TARGET_FPS = "TARGET_FPS"


class TrackStatus(StrEnum):
    """Lifecycle state of a multi-object track."""

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    TERMINATED = "TERMINATED"


class VideoSessionStatus(StrEnum):
    """Lifecycle execution status of a video processing session / job."""

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class VideoMetadata(BaseModel):
    """Metadata describing a video file asset."""

    video_id: str = Field(description="Unique video identifier ('vid_...')")
    filename: str = Field(description="Original video file name")
    duration_sec: float = Field(description="Video duration in seconds")
    fps: float = Field(description="Video frames per second rate")
    frame_count: int = Field(description="Total frame count in video")
    width: int = Field(description="Frame pixel width")
    height: int = Field(description="Frame pixel height")
    codec: str = Field(default="h264", description="Video codec format")
    size_bytes: int = Field(description="File size in bytes")
    video_fingerprint: str | None = Field(default=None, description="SHA-256 asset hash")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Upload ISO timestamp"
    )


class FrameSamplingConfig(BaseModel):
    """Configuration governing frame sampling rate during video processing."""

    mode: FrameSamplingMode = Field(
        default=FrameSamplingMode.EVERY_2ND_FRAME, description="Sampling mode strategy"
    )
    sample_interval: int = Field(
        default=2,
        ge=1,
        le=60,
        description="Sampling stride interval (e.g. 2 = process every 2nd frame)",
    )
    target_fps: float | None = Field(default=None, description="Optional target sampling FPS")
    total_sampled_frames: int = Field(
        default=0, description="Count of frames sampled for inference"
    )


class FrameRef(BaseModel):
    """Clean representation of an extracted video frame."""

    frame_index: int = Field(description="0-indexed frame number in video")
    timestamp_sec: float = Field(description="Exact frame timestamp in seconds")
    width: int = Field(description="Frame width")
    height: int = Field(description="Frame height")
    video_id: str = Field(description="Source video ID")
    image_path: str | None = Field(default=None, description="Path to extracted frame image")


class TrajectoryPoint(BaseModel):
    """Single spatial trajectory point along a tracked object's motion path."""

    frame_index: int = Field(description="Frame index of observation")
    timestamp_sec: float = Field(description="Timestamp in seconds")
    x_center_px: float = Field(description="Bounding box center X coordinate in pixels")
    y_center_px: float = Field(description="Bounding box center Y coordinate in pixels")
    norm_x: float = Field(description="Normalized center X [0.0, 1.0]")
    norm_y: float = Field(description="Normalized center Y [0.0, 1.0]")
    width_px: float = Field(description="Bounding box width in pixels")
    height_px: float = Field(description="Bounding box height in pixels")
    bbox: list[float] = Field(description="Bounding box [x_min, y_min, x_max, y_max]")
    instantaneous_speed_px_s: float | None = Field(
        default=None, description="Image-space velocity (pixels/second) from previous point"
    )


class RegionVisit(BaseModel):
    """Spatial zone interaction record for a track."""

    region_id: str = Field(description="Region of Interest ID")
    region_name: str = Field(description="Human readable zone name")
    entered_sec: float = Field(description="Timestamp when object entered region")
    exited_sec: float | None = Field(
        default=None, description="Timestamp when object exited region"
    )
    dwell_duration_sec: float = Field(default=0.0, description="Total dwell duration in seconds")
    visit_count: int = Field(default=1, description="Number of discrete visits to this zone")


class Track(BaseModel):
    """Standard representation of a tracked object maintaining identity across time."""

    track_id: int = Field(description="Persistent integer Track ID (e.g. 4 for Track #4)")
    class_name: str = Field(description="Detected object class (e.g. person, helmet, car)")
    first_frame: int = Field(description="Frame index when object was first detected")
    last_frame: int = Field(description="Frame index when object was last observed")
    first_timestamp_sec: float = Field(description="First observation timestamp in seconds")
    last_timestamp_sec: float = Field(description="Last observation timestamp in seconds")
    visibility_duration_sec: float = Field(description="Total track duration in seconds")
    avg_confidence: float = Field(description="Average detection confidence score across track")
    min_confidence: float = Field(description="Minimum detection confidence score")
    max_confidence: float = Field(description="Maximum detection confidence score")
    total_distance_px: float = Field(description="Total trajectory distance traversed in pixels")
    avg_speed_px_per_sec: float = Field(
        description="Average image-space velocity (pixels / second)"
    )
    image_space_velocity_px_s: float = Field(
        default=0.0, description="Explicitly labeled image-space displacement velocity"
    )
    observation_count: int = Field(default=0, description="Total frames where object was detected")
    gap_count: int = Field(default=0, description="Observation gaps/interpolations along track")
    status: TrackStatus = Field(default=TrackStatus.TERMINATED, description="Track lifecycle state")
    trajectory: list[TrajectoryPoint] = Field(
        default_factory=list, description="Chronological sequence of trajectory points"
    )
    detections_count: int = Field(default=0, description="Total detection count for this track")
    regions_visited: list[RegionVisit] = Field(
        default_factory=list, description="List of spatial zones interacted with"
    )
    associated_events: list[str] = Field(
        default_factory=list, description="List of TemporalEvent IDs triggered by this track"
    )
    associated_detections: list[dict[str, Any]] = Field(
        default_factory=list, description="Linked detection metadata"
    )


class TemporalAnalytics(BaseModel):
    """Aggregate temporal statistics for a video inference tracking run."""

    total_tracks: int = Field(description="Total persistent objects tracked")
    tracks_by_class: dict[str, int] = Field(
        default_factory=dict, description="Count of unique tracks per object class"
    )
    avg_track_duration_sec: float = Field(
        description="Average track visibility duration in seconds"
    )
    longest_track_duration_sec: float = Field(
        description="Longest single track duration in seconds"
    )
    avg_pixel_movement_px: float = Field(description="Average total distance traversed in pixels")
    total_region_visits: int = Field(default=0, description="Total zone entry events")
    avg_dwell_time_sec: float = Field(
        default=0.0, description="Average dwell duration across zones"
    )
    median_dwell_time_sec: float = Field(
        default=0.0, description="Median dwell duration in seconds"
    )
    events_per_minute: float = Field(default=0.0, description="Temporal event frequency")
    active_objects_over_time: list[dict[str, Any]] = Field(
        default_factory=list, description="Time series of active objects count per second"
    )
    detections_over_time: list[dict[str, Any]] = Field(
        default_factory=list, description="Time series of total detections per second"
    )


class VideoSession(BaseModel):
    """Long-running video session record with full lineage and metadata."""

    session_id: str = Field(description="Unique session identifier ('vses_...')")
    video_id: str = Field(description="Target video asset ID")
    video_source: str = Field(description="File URI or stream endpoint source")
    duration_sec: float = Field(description="Duration in seconds")
    fps: float = Field(description="Video frame rate")
    width: int = Field(description="Resolution width")
    height: int = Field(description="Resolution height")
    frame_count: int = Field(description="Total frame count")
    codec: str = Field(default="h264", description="Video compression codec")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    processing_config: dict[str, Any] = Field(default_factory=dict)
    model_version: str = Field(default="1.0.0", description="Detection model version used")
    tracking_config: dict[str, Any] = Field(default_factory=dict)
    status: VideoSessionStatus = Field(
        default=VideoSessionStatus.COMPLETED, description="Execution status"
    )
    video_fingerprint: str = Field(
        default="sha256_mock_video", description="Cryptographic asset hash"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = Field(default=None)
    lineage: dict[str, Any] = Field(
        default_factory=dict, description="Dataset/model provenance lineage"
    )


class VideoInferenceRun(BaseModel):
    """Complete execution record of a video detection & tracking pipeline run."""

    run_id: str = Field(description="Unique video inference run ID ('vrun_...')")
    video_id: str = Field(description="Target video asset ID")
    model_id: str = Field(description="Object detection model ID used for inference")
    tracker_name: str = Field(default="ByteTrack", description="Multi-object tracking algorithm")
    sampling_config: FrameSamplingConfig = Field(description="Frame sampling configuration used")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Execution ISO timestamp"
    )
    status: str = Field(default="COMPLETED", description="Run execution status")
    duration_sec: float = Field(description="Total video duration in seconds")
    processed_frames: int = Field(description="Number of frames sampled and processed")
    total_detections: int = Field(description="Total detections produced across all frames")
    total_tracks: int = Field(description="Total persistent tracks identified")
    tracks: list[Track] = Field(default_factory=list, description="List of tracked object entities")
    analytics: TemporalAnalytics = Field(description="Temporal statistics and time series data")
    processing_fps: float = Field(description="Video pipeline processing speed in frames/sec")
    inference_latency_ms: float = Field(
        description="Average per-frame model inference latency in ms"
    )
    tracking_latency_ms: float = Field(description="Average per-frame tracker update latency in ms")


class VideoComparisonResult(BaseModel):
    """Side-by-side comparative analysis between two video analysis runs."""

    comparison_id: str = Field(description="Unique comparison ID ('vcmp_...')")
    video_a_id: str = Field(description="First video asset ID")
    video_b_id: str = Field(description="Second video asset ID")
    track_count_delta: int = Field(description="Difference in total tracks (B - A)")
    event_count_delta: int = Field(description="Difference in total events (B - A)")
    avg_dwell_delta_sec: float = Field(description="Difference in average dwell duration (B - A)")
    tracks_by_class_delta: dict[str, int] = Field(default_factory=dict)
    summary_findings: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class VideoBenchmarkResult(BaseModel):
    """Performance telemetry benchmark for video processing."""

    run_id: str = Field(description="Target video run ID")
    video_id: str = Field(description="Video ID")
    model_id: str = Field(description="Model ID")
    tracker_name: str = Field(description="Tracker name")
    processing_fps: float = Field(description="End-to-end processing FPS")
    avg_inference_latency_ms: float = Field(description="Average model inference latency (ms)")
    p95_inference_latency_ms: float = Field(description="p95 model inference latency (ms)")
    tracking_latency_ms: float = Field(description="Tracking update latency (ms)")
    total_runtime_sec: float = Field(description="Total execution runtime in seconds")
