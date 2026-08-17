"""VisionForge Training Pipeline Schemas & Data Models."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TrainingStatus(StrEnum):
    """Lifecycle state of a training run."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TrainingConfig(BaseModel):
    """Strongly typed training configuration."""

    model_name: str = Field(default="yolo11s.pt", description="Base model name or checkpoint key")
    dataset_id: str = Field(description="Target dataset ID")
    preparation_id: str = Field(
        description="Target dataset preparation transaction ID ('prep_...')"
    )
    epochs: int = Field(default=50, ge=1, le=1000, description="Training epochs count")
    batch_size: int = Field(default=16, ge=1, le=256, description="Batch size per step")
    imgsz: int = Field(default=640, ge=32, le=2048, description="Input image size (pixels)")
    learning_rate: float = Field(default=0.01, gt=0.0, lt=1.0, description="Initial learning rate")
    optimizer: str = Field(
        default="auto", description="Optimizer choice ('SGD', 'Adam', 'AdamW', 'auto')"
    )
    device: str = Field(
        default="cpu", description="Execution device ('cpu', 'mps', 'cuda', 'colab_gpu')"
    )
    random_seed: int = Field(default=42, ge=0, description="Random seed for 100% reproducibility")
    checkpoint_frequency: int = Field(
        default=5, ge=1, description="Checkpoint saving frequency (epochs)"
    )
    experiment_name: str = Field(
        default="yolo11_experiment", description="Human readable experiment name"
    )
    output_dir: str | None = Field(default=None, description="Custom artifact output directory")


class MetricSnapshot(BaseModel):
    """Single epoch metric snapshot."""

    epoch: int = Field(description="Epoch number (1-indexed)")
    train_loss: float = Field(default=0.0, description="Total training loss")
    val_loss: float = Field(default=0.0, description="Total validation loss")
    precision: float = Field(default=0.0, description="Object detection Precision")
    recall: float = Field(default=0.0, description="Object detection Recall")
    map50: float = Field(default=0.0, description="Mean Average Precision at IoU=0.50 (mAP@50)")
    map50_95: float = Field(
        default=0.0, description="Mean Average Precision at IoU=0.50:0.95 (mAP@50:95)"
    )


class EvaluationResult(BaseModel):
    """Separate test set evaluation metrics."""

    eval_timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="ISO timestamp"
    )
    test_samples_count: int = Field(description="Total test samples evaluated")
    precision: float = Field(description="Test Precision")
    recall: float = Field(description="Test Recall")
    map50: float = Field(description="Test mAP@50")
    map50_95: float = Field(description="Test mAP@50:95")
    test_loss: float = Field(default=0.0, description="Test loss")


class BoundingBox(BaseModel):
    """Normalized bounding box prediction."""

    class_id: int = Field(description="Category class index")
    class_name: str = Field(description="Category label string")
    confidence: float = Field(description="Prediction confidence score")
    bbox: list[float] = Field(
        description="[x_center, y_center, width, height] normalized to [0, 1]"
    )


class InferencePrediction(BaseModel):
    """Inference smoke test prediction descriptor."""

    image_path: str = Field(description="Evaluated image file path")
    boxes: list[BoundingBox] = Field(default_factory=list, description="Detected bounding boxes")
    inference_ms: float = Field(description="Inference latency in milliseconds")


class SmokeTestResult(BaseModel):
    """Inference smoke test payload."""

    run_id: str = Field(description="Associated training run ID")
    model_name: str = Field(description="Trained model name")
    checkpoint_path: str = Field(description="Checkpoint weight location")
    predictions: list[InferencePrediction] = Field(
        default_factory=list, description="Test image predictions"
    )
    average_latency_ms: float = Field(description="Average latency across test samples")


class TrainingRun(BaseModel):
    """Complete training run execution record."""

    run_id: str = Field(description="Unique run ID ('run_...')")
    experiment_name: str = Field(description="Human readable experiment name")
    dataset_id: str = Field(description="Source dataset ID")
    dataset_version: str = Field(description="Source dataset version")
    preparation_id: str = Field(description="Dataset preparation ID ('prep_...')")
    status: TrainingStatus = Field(default=TrainingStatus.CREATED, description="Lifecycle state")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Creation ISO timestamp"
    )
    completed_at: str | None = Field(default=None, description="Completion ISO timestamp")
    config: TrainingConfig = Field(description="Training configuration")
    metrics_history: list[MetricSnapshot] = Field(
        default_factory=list, description="Per-epoch metric snapshots"
    )
    best_metrics: MetricSnapshot | None = Field(
        default=None, description="Best validation epoch metrics"
    )
    test_evaluation: EvaluationResult | None = Field(
        default=None, description="Separate test set metrics"
    )
    best_checkpoint_path: str | None = Field(default=None, description="Path to best.pt checkpoint")
    last_checkpoint_path: str | None = Field(default=None, description="Path to last.pt checkpoint")
    registered_model_version: str | None = Field(
        default=None, description="Registered model version if registered"
    )
    error_message: str | None = Field(default=None, description="Error details if run failed")
