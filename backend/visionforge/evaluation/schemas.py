"""VisionForge Model Evaluation & Error Analysis Schemas."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EvaluationStatus(StrEnum):
    """Lifecycle state of an evaluation run."""
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ErrorCategory(StrEnum):
    """Scientific taxonomy of object detection errors."""
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    MISCLASSIFICATION = "MISCLASSIFICATION"
    POOR_LOCALIZATION = "POOR_LOCALIZATION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    BACKGROUND_DETECTION = "BACKGROUND_DETECTION"
    DUPLICATE_DETECTION = "DUPLICATE_DETECTION"


class PerClassMetrics(BaseModel):
    """Evaluation metrics for a specific class."""
    class_id: int
    class_name: str
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0


class ConfusionMetrics(BaseModel):
    """Raw counts of detections against ground truth."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


class ErrorPrediction(BaseModel):
    """Detailed error prediction record for diagnostic analysis."""
    image_id: str
    image_path: str
    ground_truth_class: str | None = None
    predicted_class: str | None = None
    confidence: float | None = None
    iou: float | None = None
    error_type: ErrorCategory
    gt_bbox: list[float] | None = None
    pred_bbox: list[float] | None = None


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""
    iou_threshold: float = Field(default=0.5, description="IoU threshold for considering a detection valid")
    confidence_threshold: float = Field(default=0.25, description="Confidence threshold for predictions")


class EvaluationRun(BaseModel):
    """Complete evaluation execution record."""
    eval_id: str = Field(description="Unique evaluation ID ('eval_...')")
    model_name: str
    model_version: str | None = None
    training_run_id: str | None = None
    dataset_id: str
    dataset_version: str
    preparation_id: str | None = None
    split_used: str = Field(default="test", description="Dataset split evaluated (e.g., test, val)")
    config: EvaluationConfig
    status: EvaluationStatus = EvaluationStatus.CREATED
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    device: str = "cpu"
    software_version: str = "0.1.0"

    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0

    per_class_metrics: list[PerClassMetrics] = Field(default_factory=list)

    # Artifact references
    prediction_artifact_path: str | None = None
    report_artifact_path: str | None = None
    error_message: str | None = None


class BenchmarkRun(BaseModel):
    """Comparison of multiple models against a strictly validated protocol."""
    benchmark_id: str = Field(description="Unique benchmark ID ('bench_...')")
    models: list[str] = Field(description="List of model references or eval_ids")
    dataset_id: str
    dataset_version: str
    test_split: str
    config: EvaluationConfig
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    hardware: str = "cpu"
    software_environment: str = "0.1.0"
    metrics_summary: dict[str, dict[str, float]] = Field(default_factory=dict, description="Model -> Metrics map")
