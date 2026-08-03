"""VisionForge AI Core Inference Result Schemas."""

from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class ExecutionMetadata(BaseModel):
    """Runtime execution statistics recorded for an inference run."""

    model_name: str = Field(description="Name of the model executing inference")
    model_version: str = Field(description="Version of the executing model")
    device_used: str = Field(default="cpu", description="Compute device target used for inference")
    execution_time_ms: float = Field(description="Inference duration in milliseconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 execution timestamp",
    )


class InferenceError(BaseModel):
    """Structured error descriptor for failed model inference runs."""

    code: str = Field(description="Machine-readable error classification code")
    message: str = Field(description="Human-readable description of error cause")
    details: list[dict[str, Any]] | None = Field(
        default=None, description="Optional diagnostic context or stack details"
    )


class InferenceResult(BaseModel, Generic[DataT]):
    """Standardized inference result envelope returned by all VisionForge vision models."""

    success: bool = Field(default=True, description="Indicates whether inference executed cleanly")
    message: str = Field(
        default="Inference completed successfully",
        description="Status summary message",
    )
    data: DataT | None = Field(default=None, description="Inference output payload data")
    metadata: ExecutionMetadata | None = Field(
        default=None, description="Runtime execution metrics and device metadata"
    )
    error: InferenceError | None = Field(
        default=None, description="Populated with error details if execution fails"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings emitted during inference"
    )
