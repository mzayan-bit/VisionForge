"""VisionForge Inference Engine Schemas and Data Models."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ModelLifecycleState(StrEnum):
    """Lifecycle state of an inference model in memory."""

    NOT_LOADED = "NOT_LOADED"
    LOADING = "LOADING"
    READY = "READY"
    FAILED = "FAILED"
    UNLOADING = "UNLOADING"


class InferenceConfig(BaseModel):
    """Strongly typed configuration for model inference."""

    model_id: str = Field(description="Target model identifier or checkpoint path")
    confidence_threshold: float = Field(
        default=0.25, ge=0.01, le=1.0, description="Minimum prediction confidence threshold"
    )
    iou_threshold: float = Field(
        default=0.45, ge=0.01, le=1.0, description="NMS Intersection over Union threshold"
    )
    imgsz: int = Field(default=640, ge=32, le=2048, description="Target image dimension in pixels")
    device: str = Field(
        default="auto", description="Compute device target ('auto', 'cpu', 'cuda', 'mps')"
    )


class NormalizedBoundingBox(BaseModel):
    """Bounding box representation normalized to [0, 1] coordinate space."""

    x_center: float = Field(description="Normalized X center coordinate [0, 1]")
    y_center: float = Field(description="Normalized Y center coordinate [0, 1]")
    width: float = Field(description="Normalized box width [0, 1]")
    height: float = Field(description="Normalized box height [0, 1]")
    pixel_coords: list[float] | None = Field(
        default=None, description="Pixel coordinate bounding box [x1, y1, x2, y2]"
    )


class StandardPrediction(BaseModel):
    """Framework-independent object detection prediction representation."""

    prediction_id: str = Field(description="Unique prediction identifier")
    class_id: int = Field(description="Numerical category class index")
    class_name: str = Field(description="Human readable category label string")
    confidence: float = Field(description="Confidence probability score [0.0, 1.0]")
    bbox: NormalizedBoundingBox = Field(description="Normalized bounding box coordinates")
    model_id: str = Field(description="Source model identifier")
    model_version: str = Field(default="1.0.0", description="Source model version tag")


class PredictionSummary(BaseModel):
    """Aggregated prediction statistics for an inference execution."""

    total_detections: int = Field(default=0, description="Total detected bounding boxes")
    classes_detected: list[str] = Field(
        default_factory=list, description="Unique class labels detected"
    )
    highest_confidence: float = Field(
        default=0.0, description="Maximum confidence score in result set"
    )
    average_confidence: float = Field(
        default=0.0, description="Mean confidence score across predictions"
    )
    inference_ms: float = Field(
        default=0.0, description="Inference execution latency in milliseconds"
    )
    model_id: str = Field(description="Evaluated model ID")
    image_width: int = Field(default=0, description="Original image width in pixels")
    image_height: int = Field(default=0, description="Original image height in pixels")


class InferenceResult(BaseModel):
    """Complete standardized result descriptor of an inference run."""

    inference_id: str = Field(description="Unique inference transaction ID ('inf_...')")
    image_path: str = Field(description="Path to processed source image file")
    image_id: str | None = Field(default=None, description="Associated Visual Memory image ID")
    model_id: str = Field(description="Model identifier")
    model_version: str = Field(default="1.0.0", description="Model version tag")
    predictions: list[StandardPrediction] = Field(
        default_factory=list, description="List of detected objects"
    )
    summary: PredictionSummary = Field(description="Summary metrics")
    config: InferenceConfig = Field(description="Inference configuration parameters used")
    visual_overlay_path: str | None = Field(
        default=None, description="Path to derived visualization image with overlay boxes"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="ISO timestamp"
    )


class InferenceModelDescriptor(BaseModel):
    """Descriptor for a registered or available model ready for inference."""

    model_id: str = Field(description="Model identifier key")
    name: str = Field(description="Display model name")
    version: str = Field(default="1.0.0", description="Model version tag")
    task: str = Field(default="detection", description="Computer Vision task category")
    framework: str = Field(default="Ultralytics/PyTorch", description="Model framework runtime")
    checkpoint_path: str = Field(description="Path to model weights file")
    status: ModelLifecycleState = Field(
        default=ModelLifecycleState.NOT_LOADED, description="In-memory lifecycle status"
    )
    training_run_id: str | None = Field(default=None, description="Associated training run ID")
    dataset_id: str | None = Field(default=None, description="Dataset used during training")
    map50: float | None = Field(default=None, description="Validation mAP@50 metric")
    precision: float | None = Field(default=None, description="Validation precision metric")
    recall: float | None = Field(default=None, description="Validation recall metric")
    is_available: bool = Field(default=True, description="Whether checkpoint is available on disk")
    unavailability_reason: str | None = Field(
        default=None, description="Reason if model checkpoint cannot be loaded"
    )


class ModelComparisonRequest(BaseModel):
    """Payload for comparing two models on a single image."""

    image_path: str | None = Field(default=None, description="Image file path")
    image_id: str | None = Field(default=None, description="Visual Memory image ID")
    model_a_id: str = Field(description="First model identifier")
    model_b_id: str = Field(description="Second model identifier")
    config_a: InferenceConfig | None = Field(default=None, description="Config for Model A")
    config_b: InferenceConfig | None = Field(default=None, description="Config for Model B")


class ModelComparisonResult(BaseModel):
    """Side-by-side model comparison result."""

    comparison_id: str = Field(description="Unique comparison transaction ID ('cmp_...')")
    image_path: str = Field(description="Evaluated image file path")
    image_width: int = Field(description="Original image width")
    image_height: int = Field(description="Original image height")
    model_a_result: InferenceResult = Field(description="Inference result for Model A")
    model_b_result: InferenceResult = Field(description="Inference result for Model B")
    notes: str = Field(description="Qualitative summary of latency and detection differences")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="ISO timestamp"
    )


class InferenceBenchmarkConfig(BaseModel):
    """Configuration for inference latency benchmarking."""

    model_id: str = Field(description="Model identifier")
    runs: int = Field(default=20, ge=1, le=500, description="Number of timed benchmark passes")
    warmup_runs: int = Field(default=3, ge=0, le=50, description="Number of warm-up iterations")
    batch_size: int = Field(default=1, ge=1, le=64, description="Inference batch size")
    imgsz: int = Field(default=640, ge=32, le=2048, description="Image resolution")
    device: str = Field(default="auto", description="Execution device target")


class InferenceBenchmarkResult(BaseModel):
    """Detailed telemetry results of an inference latency benchmark."""

    benchmark_id: str = Field(description="Unique benchmark ID ('bm_...')")
    model_id: str = Field(description="Evaluated model ID")
    model_version: str = Field(default="1.0.0", description="Model version tag")
    device: str = Field(description="Hardware execution device")
    runs: int = Field(description="Number of measured iterations")
    average_latency_ms: float = Field(description="Mean inference latency in ms")
    median_latency_ms: float = Field(description="50th percentile (median) latency in ms")
    p95_latency_ms: float = Field(description="95th percentile latency in ms")
    min_latency_ms: float = Field(description="Minimum latency in ms")
    max_latency_ms: float = Field(description="Maximum latency in ms")
    fps: float = Field(description="Calculated throughput frames per second")
    hardware_info: str = Field(description="Runtime hardware environment details")
    config: InferenceBenchmarkConfig = Field(description="Benchmark configuration parameters")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="ISO timestamp"
    )
