"""VisionForge Model Evaluation, Benchmark Lab, & Model Comparison Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

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


class RegressionStatus(StrEnum):
    """Regression determination status between candidate and baseline models."""

    IMPROVED = "IMPROVED"
    REGRESSION = "REGRESSION"
    NEUTRAL = "NEUTRAL"
    INCOMPARABLE = "INCOMPARABLE"


class PRCurvePoint(BaseModel):
    """Precision-Recall coordinate point on the operating curve."""

    recall: float = Field(description="Recall value in range [0.0, 1.0]")
    precision: float = Field(description="Precision value in range [0.0, 1.0]")


class PerClassMetrics(BaseModel):
    """Evaluation metrics for a specific class category."""

    class_id: int = Field(description="Zero-indexed class identifier")
    class_name: str = Field(description="Human-readable category class label")
    precision: float = Field(default=0.0, description="Precision score at default threshold")
    recall: float = Field(default=0.0, description="Recall score at default threshold")
    f1: float = Field(default=0.0, description="Harmonic mean F1 score")
    map50: float = Field(default=0.0, description="Average Precision at IoU=0.50 (AP@50)")
    map75: float = Field(default=0.0, description="Average Precision at IoU=0.75 (AP@75)")
    map50_95: float = Field(default=0.0, description="Mean Average Precision across IoU=[0.50:0.95] (AP@[.5:.95])")
    support: int = Field(default=0, description="Ground truth annotation count for this class")
    predictions_count: int = Field(default=0, description="Total predictions emitted for this class")
    true_positives: int = Field(default=0, description="True positive detection count")
    false_positives: int = Field(default=0, description="False positive detection count")
    false_negatives: int = Field(default=0, description="False negative missed object count")
    pr_curve_points: list[PRCurvePoint] = Field(
        default_factory=list, description="Interpolated Precision-Recall curve coordinates"
    )


class ThresholdPoint(BaseModel):
    """Metrics evaluated at a specific confidence threshold operating point."""

    confidence_threshold: float = Field(description="Confidence threshold evaluated (e.g. 0.10, 0.20, ...)")
    precision: float = Field(description="Precision at this threshold")
    recall: float = Field(description="Recall at this threshold")
    f1: float = Field(description="F1 score at this threshold")
    true_positives: int = Field(description="TP count")
    false_positives: int = Field(description="FP count")
    false_negatives: int = Field(description="FN count")


class DetectionMetrics(BaseModel):
    """Aggregate object detection metrics."""

    precision: float = Field(default=0.0, description="Aggregate Precision")
    recall: float = Field(default=0.0, description="Aggregate Recall")
    f1: float = Field(default=0.0, description="Aggregate F1 Score")
    mean_iou: float = Field(default=0.0, description="Mean IoU of True Positive matches")
    map50: float = Field(default=0.0, description="mAP at IoU=0.50")
    map75: float = Field(default=0.0, description="mAP at IoU=0.75")
    map50_95: float = Field(default=0.0, description="mAP at IoU=[0.50:0.95]")
    support_gt_count: int = Field(default=0, description="Total ground truth objects evaluated")
    total_predictions: int = Field(default=0, description="Total candidate detections evaluated")


class ConfusionMatrixData(BaseModel):
    """Raw counts of predictions against ground truth classes including background."""

    class_names: list[str] = Field(description="Class labels list (last item is 'background')")
    matrix: list[list[int]] = Field(description="2D integer confusion matrix [row=GT, col=Pred]")
    total_samples: int = Field(default=0, description="Total bounding box evaluations")


class RuntimeMetrics(BaseModel):
    """Comprehensive latency and throughput hardware execution profile."""

    warmup_iterations: int = Field(default=5, description="Number of warm-up iterations excluded from timing")
    evaluated_iterations: int = Field(default=30, description="Number of steady-state iterations benchmarked")
    preprocess_ms_mean: float = Field(default=0.0, description="Mean preprocessing latency in ms")
    preprocess_ms_p95: float = Field(default=0.0, description="95th percentile preprocessing latency in ms")
    inference_ms_mean: float = Field(default=0.0, description="Mean model forward pass latency in ms")
    inference_ms_median: float = Field(default=0.0, description="Median model forward pass latency in ms")
    inference_ms_p95: float = Field(default=0.0, description="95th percentile model forward pass latency in ms")
    postprocess_ms_mean: float = Field(default=0.0, description="Mean postprocessing/NMS latency in ms")
    postprocess_ms_p95: float = Field(default=0.0, description="95th percentile postprocessing/NMS latency in ms")
    total_latency_ms_mean: float = Field(default=0.0, description="Mean total end-to-end latency in ms")
    total_latency_ms_p95: float = Field(default=0.0, description="95th percentile total latency in ms")
    throughput_fps: float = Field(default=0.0, description="Steady-state throughput (Frames/Images Per Second)")
    model_parameters_m: float | None = Field(default=None, description="Model parameter count in Millions")
    model_size_mb: float | None = Field(default=None, description="Model checkpoint file size in MB")
    device: str = Field(default="cpu", description="Execution device (e.g. 'cpu', 'cuda:0', 'mps')")
    device_name: str = Field(default="Generic CPU", description="Human readable hardware descriptor")


class ErrorPrediction(BaseModel):
    """Detailed error prediction record for diagnostic analysis."""

    image_id: str = Field(description="Identifier of the evaluated image")
    image_path: str = Field(description="Relative or absolute path of the image")
    ground_truth_class: str | None = Field(default=None, description="Ground truth class if applicable")
    predicted_class: str | None = Field(default=None, description="Predicted class if applicable")
    confidence: float | None = Field(default=None, description="Prediction confidence score")
    iou: float | None = Field(default=None, description="Intersection over Union with nearest GT")
    error_type: ErrorCategory = Field(description="Taxonomy classification of the failure")
    gt_bbox: list[float] | None = Field(default=None, description="GT bbox [x1, y1, x2, y2] or [xc, yc, w, h]")
    pred_bbox: list[float] | None = Field(default=None, description="Pred bbox [x1, y1, x2, y2] or [xc, yc, w, h]")
    sample_link: str | None = Field(default=None, description="Link to dataset sample or visual viewer")


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation and benchmarking run."""

    iou_threshold: float = Field(default=0.5, ge=0.1, le=0.95, description="IoU threshold for considering detection valid")
    confidence_threshold: float = Field(default=0.25, ge=0.01, le=0.95, description="Confidence threshold for predictions")
    nms_iou_threshold: float = Field(default=0.45, ge=0.1, le=0.95, description="NMS IoU threshold")
    img_size: int = Field(default=640, description="Input image size (pixels)")
    batch_size: int = Field(default=1, ge=1, le=64, description="Evaluation batch size")
    device: str = Field(default="cpu", description="Compute device ('cpu', 'cuda', 'mps')")
    fp16: bool = Field(default=False, description="Half-precision floating point mode")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    warmup_iterations: int = Field(default=5, ge=0, le=50, description="Warm-up iterations before runtime profiling")


class BenchmarkDatasetSnapshot(BaseModel):
    """Immutable snapshot of the dataset evaluated during benchmarking."""

    dataset_id: str = Field(description="Dataset identifier")
    dataset_version: str = Field(description="Dataset version identifier")
    dataset_fingerprint: str = Field(description="Cryptographic SHA-256 fingerprint of dataset")
    split_used: str = Field(default="test", description="Dataset split evaluated ('test', 'val', 'train')")
    total_images: int = Field(default=0, description="Total image samples in evaluated split")
    total_annotations: int = Field(default=0, description="Total bounding box annotations")
    class_distribution: dict[str, int] = Field(default_factory=dict, description="Annotation count per class")


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
    """Comprehensive benchmark execution record with full metrics, runtime, and lineage."""

    benchmark_id: str = Field(description="Unique benchmark ID ('bench_...')")
    name: str = Field(description="Human readable benchmark name")
    description: str = Field(default="", description="Benchmark description or research objective")
    task: str = Field(default="OBJECT_DETECTION", description="Computer vision task evaluated")
    model_name: str = Field(description="Evaluated model name")
    model_version: str = Field(default="1.0.0", description="Model architecture or checkpoint version")
    checkpoint_path: str | None = Field(default=None, description="Path to evaluated model checkpoint file")
    is_baseline: bool = Field(default=False, description="Whether this benchmark serves as a baseline reference")
    baseline_benchmark_id: str | None = Field(default=None, description="ID of baseline benchmark if candidate")
    dataset_snapshot: BenchmarkDatasetSnapshot = Field(description="Immutable dataset snapshot evaluated")
    config: EvaluationConfig = Field(description="Evaluation hyperparameters and settings")
    status: EvaluationStatus = Field(default=EvaluationStatus.CREATED, description="Execution lifecycle status")
    metrics: DetectionMetrics = Field(default_factory=DetectionMetrics, description="Aggregate detection metrics")
    per_class_metrics: list[PerClassMetrics] = Field(default_factory=list, description="Per-class metric breakdowns")
    threshold_analysis: list[ThresholdPoint] = Field(
        default_factory=list, description="Confidence threshold operating points"
    )
    confusion_matrix: ConfusionMatrixData = Field(
        default_factory=lambda: ConfusionMatrixData(class_names=[], matrix=[], total_samples=0),
        description="Multi-class confusion matrix",
    )
    runtime_metrics: RuntimeMetrics = Field(
        default_factory=RuntimeMetrics, description="Latency and throughput profiling"
    )
    errors_summary: dict[str, int] = Field(
        default_factory=dict, description="Counts per diagnostic error category"
    )
    reproducibility: dict[str, Any] = Field(
        default_factory=dict, description="Environment snapshot, git commit SHA, and randomness seeds"
    )
    experiment_id: str | None = Field(default=None, description="Experiment reference ID if linked")
    artifacts: list[dict[str, Any]] = Field(
        default_factory=list, description="Generated artifacts (metrics.json, report.md, etc.)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Benchmark creation timestamp"
    )
    completed_at: str | None = Field(default=None, description="Benchmark completion timestamp")


class ModelComparisonResult(BaseModel):
    """Controlled scientific comparison between baseline and candidate models."""

    comparison_id: str = Field(description="Unique comparison ID ('cmp_...')")
    baseline_benchmark: BenchmarkRun = Field(description="Baseline model benchmark record")
    candidate_benchmark: BenchmarkRun = Field(description="Candidate model benchmark record")
    is_directly_comparable: bool = Field(
        description="Whether evaluation conditions (dataset, version, split, task) match"
    )
    incompatibility_reasons: list[str] = Field(
        default_factory=list, description="Reasons if comparison violates scientific control"
    )
    metric_deltas: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metric -> {'baseline': x, 'candidate': y, 'delta_abs': d, 'delta_rel_pct': p}"
    )
    per_class_deltas: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Class -> {'map50_delta': d, 'recall_delta': d, 'precision_delta': d}"
    )
    regression_status: RegressionStatus = Field(
        default=RegressionStatus.NEUTRAL, description="Automated regression determination"
    )
    regression_notes: list[str] = Field(
        default_factory=list, description="Detailed regression or improvement observations"
    )
    failure_transitions: dict[str, int] = Field(
        default_factory=dict, description="Fixed vs new vs persistent failure counts"
    )
    disagreement_samples: list[dict[str, Any]] = Field(
        default_factory=list, description="List of samples where model predictions significantly disagree"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Comparison timestamp"
    )


class BenchmarkHistoryItem(BaseModel):
    """Point in longitudinal model performance progression."""

    benchmark_id: str
    model_name: str
    model_version: str
    timestamp: str
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1: float
    throughput_fps: float
    total_latency_ms: float
    dataset_version: str
    is_baseline: bool
