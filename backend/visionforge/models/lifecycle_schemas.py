"""VisionForge End-to-End Model Lifecycle & Pipeline Schemas.

Covers the entire 9-stage ML experiment-to-deployment production cycle:
1. Dataset Version (Fingerprint, Split audit, Class balance)
2. Training Configuration (Hyperparameters, Augmentations, Optimizer)
3. Training Run (Epoch loss telemetry, GPU monitoring, Checkpoints)
4. Model Artifact (SHA-256 fingerprint, Weights formatting, Storage)
5. Model Evaluation (COCO 101-pt mAP@50, mAP@50:95, Error taxonomy)
6. Latency & Resource Benchmark (P50/P95/P99, FPS, Memory, Warm-up W=5)
7. Model Registry Governance (Versioning, Staging/Production promotion)
8. Model Comparison (Baseline vs Candidate delta, Pareto frontier)
9. Deployment & Active Inference (Vision Lab, Video Tracking, REST endpoints)
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class PipelineStage(StrEnum):
    """The 9 canonical stages of the VisionForge Model Production Pipeline."""

    DATASET_VERSION = "DATASET_VERSION"
    TRAINING_CONFIG = "TRAINING_CONFIG"
    TRAINING_RUN = "TRAINING_RUN"
    MODEL_ARTIFACT = "MODEL_ARTIFACT"
    EVALUATION = "EVALUATION"
    BENCHMARK = "BENCHMARK"
    MODEL_REGISTRY = "MODEL_REGISTRY"
    MODEL_COMPARISON = "MODEL_COMPARISON"
    DEPLOYMENT = "DEPLOYMENT"


class PipelineStatus(StrEnum):
    """Execution status of an end-to-end model lifecycle pipeline."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


class StageExecutionState(StrEnum):
    """Status of an individual pipeline stage."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class StageDetail(BaseModel):
    """Detailed record and metrics for a specific pipeline stage."""

    stage: PipelineStage = Field(description="Stage enum identifier")
    step_number: int = Field(ge=1, le=9, description="Stage index (1 to 9)")
    title: str = Field(description="Stage human display title")
    status: StageExecutionState = Field(
        default=StageExecutionState.PENDING, description="Execution status"
    )
    summary: str = Field(default="", description="High-level execution summary")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Numerical metrics captured during stage"
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict, description="Artifact references (IDs, paths, hashes)"
    )
    logs: list[str] = Field(default_factory=list, description="Stage execution logs")
    started_at: str | None = Field(default=None)
    completed_at: str | None = Field(default=None)


class PipelineLineageNode(BaseModel):
    """Node in the end-to-end provenance and lineage graph."""

    id: str = Field(description="Unique node identifier")
    stage: PipelineStage = Field(description="Associated pipeline stage")
    label: str = Field(description="Display label")
    artifact_type: str = Field(description="Type of entity (dataset, config, run, weights, eval)")
    properties: dict[str, Any] = Field(default_factory=dict)
    parent_node_ids: list[str] = Field(default_factory=list)


class ModelLifecyclePipeline(BaseModel):
    """Complete entity tracking an end-to-end model lifecycle execution."""

    pipeline_id: str = Field(description="Unique pipeline execution ID ('pipe_...')")
    name: str = Field(description="Descriptive pipeline run name")
    dataset_id: str = Field(default="safety_v2", description="Source dataset identifier")
    dataset_version: str = Field(default="v1.0.0", description="Source dataset version tag")
    base_model: str = Field(default="yolo11s.pt", description="Base pretrained model checkpoint")
    target_model_name: str = Field(
        default="yolo11s_safety_production", description="Target model name in registry"
    )
    current_stage: PipelineStage = Field(
        default=PipelineStage.DATASET_VERSION, description="Active stage in pipeline"
    )
    status: PipelineStatus = Field(
        default=PipelineStatus.PENDING, description="Overall pipeline status"
    )
    stages: dict[str, StageDetail] = Field(
        default_factory=dict, description="Dictionary of stage details keyed by stage name"
    )
    is_deployed: bool = Field(default=False, description="Whether final model is deployed to runtime")
    deployment_endpoint: str | None = Field(
        default=None, description="Active endpoint or engine reference"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = Field(default=None)


class CreatePipelineRequest(BaseModel):
    """Payload for creating and initiating a new model lifecycle pipeline."""

    name: str = Field(
        default="Safety Vision Production Lifecycle Run",
        description="Display name for pipeline run",
    )
    dataset_id: str = Field(default="safety_v2", description="Target dataset ID")
    dataset_version: str = Field(default="v1.0.0", description="Dataset version tag")
    base_model: str = Field(default="yolo11s.pt", description="Base model weights")
    target_model_name: str = Field(
        default="yolo11s_safety_v1", description="Registered model name"
    )
    epochs: int = Field(default=50, ge=1, le=500, description="Training epochs")
    batch_size: int = Field(default=16, ge=1, le=128, description="Batch size")
    imgsz: int = Field(default=640, ge=320, le=1280, description="Image input dimension")
    learning_rate: float = Field(default=0.01, ge=1e-5, le=1.0, description="Initial learning rate")
    optimizer: str = Field(default="SGD", description="Optimizer ('SGD', 'AdamW')")
    auto_advance: bool = Field(
        default=True, description="Whether to execute entire pipeline automatically"
    )


class AdvancePipelineRequest(BaseModel):
    """Payload for advancing a paused pipeline to the next stage."""

    target_stage: PipelineStage | None = Field(
        default=None, description="Optional target stage to jump to"
    )


class DeployModelRequest(BaseModel):
    """Payload for deploying the model from a completed pipeline run."""

    environment: str = Field(default="production", description="'staging', 'production'")
    device: str = Field(default="auto", description="'cpu', 'cuda', 'mps', 'auto'")
    warm_up_runs: int = Field(default=5, ge=0, le=20, description="Pre-warm inference iterations")
