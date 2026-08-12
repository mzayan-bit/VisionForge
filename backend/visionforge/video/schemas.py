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
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Upload ISO timestamp"
    )


class FrameSamplingConfig(BaseModel):
    """Configuration governing frame sampling rate during video processing."""

    mode: FrameSamplingMode = Field(
        default=FrameSamplingMode.EVERY_2ND_FRAME, description="Sampling mode strategy"
    )
    sample_interval: int = Field(
        default=2, ge=1, le=60, description="Sampling stride interval (e.g. 2 = process every 2nd frame)"
    )
    target_fps: float | None = Field(default=None, description="Optional target sampling FPS")
    total_sampled_frames: int = Field(default=0, description="Count of frames sampled for inference")


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
    avg_speed_px_per_sec: float = Field(description="Average pixel speed (pixels / second)")
    status: TrackStatus = Field(default=TrackStatus.TERMINATED, description="Track lifecycle state")
    trajectory: list[TrajectoryPoint] = Field(
        default_factory=list, description="Chronological sequence of trajectory points"
    )
    detections_count: int = Field(default=0, description="Total detection count for this track")


class TemporalAnalytics(BaseModel):
    """Aggregate temporal statistics for a video inference tracking run."""

    total_tracks: int = Field(description="Total persistent objects tracked")
    tracks_by_class: dict[str, int] = Field(
        default_factory=dict, description="Count of unique tracks per object class"
    )
    avg_track_duration_sec: float = Field(description="Average track visibility duration in seconds")
    longest_track_duration_sec: float = Field(description="Longest single track duration in seconds")
    avg_pixel_movement_px: float = Field(description="Average total distance traversed in pixels")
    active_objects_over_time: list[dict[str, Any]] = Field(
        default_factory=list, description="Time series of active objects count per second"
    )
    detections_over_time: list[dict[str, Any]] = Field(
        default_factory=list, description="Time series of total detections per second"
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
    inference_latency_ms: float = Field(description="Average per-frame model inference latency in ms")
    tracking_latency_ms: float = Field(description="Average per-frame tracker update latency in ms")


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
