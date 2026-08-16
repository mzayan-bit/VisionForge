"""VisionForge Model Evaluation, Benchmark Lab, Error Analysis, & Model Comparison Schemas."""

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
    """Scientific taxonomy of object detection diagnostic errors."""

    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    MISCLASSIFICATION = "MISCLASSIFICATION"
    WRONG_CLASS = "WRONG_CLASS"
    POOR_LOCALIZATION = "POOR_LOCALIZATION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    BACKGROUND_DETECTION = "BACKGROUND_DETECTION"
    DUPLICATE_DETECTION = "DUPLICATE_DETECTION"
    SMALL_OBJECT_FAILURE = "SMALL_OBJECT_FAILURE"


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

    confidence_threshold: float = Field(description="Confidence threshold evaluated (e.g. 0.20, 0.30, ...)")
    precision: float = Field(description="Precision at this threshold")
    recall: float = Field(description="Recall at this threshold")
    f1: float = Field(description="F1 score at this threshold")
    true_positives: int = Field(default=0, description="TP count")
    false_positives: int = Field(default=0, description="FP count")
    false_negatives: int = Field(default=0, description="FN count")


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


class ConfusionPair(BaseModel):
    """Measured classification confusion between ground truth and predicted class."""

    ground_truth_class: str = Field(description="Actual ground truth class label")
    predicted_class: str = Field(description="Incorrectly predicted class label")
    count: int = Field(default=0, description="Number of instances confused")
    mean_confidence: float = Field(default=0.0, description="Average confidence of confused predictions")
    mean_iou: float = Field(default=0.0, description="Average IoU of confused bounding boxes")
    sample_ids: list[str] = Field(default_factory=list, description="Associated image sample IDs")


class ConfusionMatrixData(BaseModel):
    """Raw counts of predictions against ground truth classes including background."""

    class_names: list[str] = Field(description="Class labels list (last item is 'background')")
    matrix: list[list[int]] = Field(description="2D integer confusion matrix [row=GT, col=Pred]")
    total_samples: int = Field(default=0, description="Total bounding box evaluations")
    confusion_pairs: list[ConfusionPair] = Field(
        default_factory=list, description="Aggregated top misclassification confusion pairs"
    )


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


class FailureSampleDetail(BaseModel):
    """Comprehensive diagnostic record for a single failure candidate."""

    sample_id: str = Field(description="Unique failure record identifier")
    eval_id: str = Field(description="Associated Evaluation / Benchmark ID")
    image_id: str = Field(description="Identifier of the evaluated image")
    image_path: str = Field(description="Path to image file")
    error_type: ErrorCategory = Field(description="Diagnostic failure category")
    ground_truth_class: str | None = Field(default=None, description="Expected GT class")
    predicted_class: str | None = Field(default=None, description="Model predicted class")
    confidence: float | None = Field(default=None, description="Prediction confidence score")
    iou: float | None = Field(default=None, description="Intersection over Union with nearest GT")
    model_id: str = Field(default="yolo11s.pt", description="Evaluated model checkpoint")
    model_version: str = Field(default="1.0.0", description="Evaluated model version")
    dataset_id: str = Field(default="safety_v2", description="Dataset identifier")
    dataset_version: str = Field(default="v1.0.0", description="Dataset version tag")
    split: str = Field(default="test", description="Dataset split evaluated ('test', 'val')")
    image_size: list[int] = Field(default_factory=lambda: [1280, 720], description="Image dimensions [W, H]")
    object_size_category: str = Field(default="medium", description="'small' (<32^2), 'medium', 'large' (>96^2)")
    gt_bbox: list[float] | None = Field(default=None, description="Ground truth bounding box [x1, y1, x2, y2]")
    pred_bbox: list[float] | None = Field(default=None, description="Predicted bounding box [x1, y1, x2, y2]")
    nearby_ground_truths: list[dict[str, Any]] = Field(
        default_factory=list, description="Other GT bounding boxes in same image"
    )
    competing_predictions: list[dict[str, Any]] = Field(
        default_factory=list, description="Other predictions emitted in same image"
    )
    similar_sample_ids: list[str] = Field(
        default_factory=list, description="Nearest neighbor image IDs from visual memory"
    )
    embedding_preview: list[float] = Field(
        default_factory=list, description="768D visual embedding sample preview"
    )
    dataset_quality_flags: list[str] = Field(
        default_factory=list, description="Dataset quality notes (e.g. 'crowded_scene', 'low_contrast')"
    )
    review_priority: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Priority score = 0.40*(1-conf) + 0.35*(1-iou) + 0.25*err_weight"
    )
    review_status: str = Field(
        default="UNREVIEWED", description="'UNREVIEWED', 'CONFIRMED_ERROR', 'ANNOTATION_ISSUE', 'VALID_HARD_EXAMPLE', 'SKIPPED'"
    )
    notes: str = Field(default="", description="Reviewer notes")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# Backward-compatibility alias
ErrorPrediction = FailureSampleDetail


class VisualFailureCluster(BaseModel):
    """Unsupervised cluster of visually similar failure samples in embedding space."""

    cluster_id: str = Field(description="Unique cluster ID ('cluster_1', 'cluster_2', ...)")
    label: str = Field(description="Cluster display label ('Cluster 1', 'Cluster 2', ...)")
    sample_count: int = Field(description="Number of failure samples in cluster")
    representative_sample_ids: list[str] = Field(
        default_factory=list, description="Centroid-closest exemplar sample IDs"
    )
    representative_image_paths: list[str] = Field(
        default_factory=list, description="Exemplar image paths for gallery thumbnails"
    )
    primary_error_types: dict[str, int] = Field(
        default_factory=dict, description="Distribution of error types in this cluster"
    )
    avg_confidence: float = Field(default=0.0, description="Average confidence of failures in cluster")
    avg_iou: float = Field(default=0.0, description="Average IoU of failures in cluster")


class ObjectSizePerformance(BaseModel):
    """Detection performance broken down by bounding box pixel area."""

    size_category: str = Field(description="'small' (area < 32^2), 'medium' (32^2 <= area <= 96^2), 'large' (area > 96^2)")
    area_range_px: str = Field(description="Pixel area description")
    gt_count: int = Field(description="Total GT objects in this size category")
    prediction_count: int = Field(description="Total predictions for this size")
    true_positives: int = Field(description="TP count")
    false_positives: int = Field(description="FP count")
    false_negatives: int = Field(description="FN count")
    precision: float = Field(description="Precision score")
    recall: float = Field(description="Recall score")
    f1: float = Field(description="F1 score")
    ap50: float = Field(description="Average Precision at IoU=0.50")


class ResolutionPerformance(BaseModel):
    """Detection performance broken down by image input resolution bands."""

    resolution_range: str = Field(description="'< 480px', '480-720px', '720-1080px', '> 1080px'")
    sample_count: int = Field(description="Total images evaluated in this resolution band")
    true_positives: int = Field(description="TP count")
    false_positives: int = Field(description="FP count")
    false_negatives: int = Field(description="FN count")
    precision: float = Field(description="Precision score")
    recall: float = Field(description="Recall score")
    f1: float = Field(description="F1 score")
    map50: float = Field(description="mAP@50 score")


class PatternAnalysisReport(BaseModel):
    """Multi-dimensional failure pattern correlation report."""

    eval_id: str = Field(description="Evaluation / Benchmark run ID")
    size_performance: list[ObjectSizePerformance] = Field(default_factory=list)
    resolution_performance: list[ResolutionPerformance] = Field(default_factory=list)
    confusion_pairs: list[ConfusionPair] = Field(default_factory=list)
    split_breakdown: dict[str, int] = Field(default_factory=dict)
    summary_findings: list[str] = Field(default_factory=list)


class ConfidenceDistributions(BaseModel):
    """Empirical confidence distributions across prediction categories."""

    tp_confidences: list[float] = Field(default_factory=list)
    fp_confidences: list[float] = Field(default_factory=list)
    fn_confidences: list[float] = Field(default_factory=list)
    tp_histogram: dict[str, int] = Field(default_factory=dict)
    fp_histogram: dict[str, int] = Field(default_factory=dict)


class PRCurveData(BaseModel):
    """Complete Precision-Recall curve coordinates for overall model and per class."""

    overall_pr_curve: list[PRCurvePoint] = Field(default_factory=list)
    class_pr_curves: dict[str, list[PRCurvePoint]] = Field(default_factory=dict)


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
    model_name: str = Field(description="Evaluated model name")
    model_version: str | None = Field(default="1.0.0", description="Model checkpoint version")
    training_run_id: str | None = None
    dataset_id: str = Field(description="Dataset identifier")
    dataset_version: str = Field(default="v1.0.0", description="Dataset version tag")
    preparation_id: str | None = None
    split_used: str = Field(default="test", description="Dataset split evaluated (e.g. test, val)")
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)
    status: EvaluationStatus = EvaluationStatus.CREATED
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    device: str = "cpu"
    software_version: str = "0.1.0"
    environment: dict[str, Any] = Field(default_factory=dict)

    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map50: float = 0.0
    map75: float = 0.0
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
    """Controlled scientific comparison between baseline and candidate models on the same evaluation dataset."""

    comparison_id: str = Field(description="Unique comparison ID ('cmp_...')")
    baseline_benchmark: BenchmarkRun = Field(description="Baseline model benchmark record (M_A)")
    candidate_benchmark: BenchmarkRun = Field(description="Candidate model benchmark record (M_B)")
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
    failure_deltas: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Error category -> {'baseline_count': b, 'candidate_count': c, 'delta': c - b}"
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
