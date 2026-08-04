"""Vision Engine Internal Metrics Collection."""

import time

from pydantic import BaseModel, Field


class StageMetrics(BaseModel):
    """Timing and diagnostic record for an individual pipeline stage."""

    stage_name: str = Field(description="Name of the pipeline stage")
    duration_ms: float = Field(description="Execution duration in milliseconds")
    status: str = Field(
        default="success",
        description="Stage completion status ('success', 'failed')",
    )


class ExecutionMetrics(BaseModel):
    """Aggregate execution metrics payload for a vision engine task run."""

    total_duration_ms: float = Field(default=0.0, description="Total pipeline execution latency")
    device_used: str = Field(default="cpu", description="Compute device backend used for execution")
    memory_vram_mb: int = Field(default=0, description="VRAM consumption in megabytes")
    memory_ram_mb: int = Field(default=0, description="RAM consumption in megabytes")
    stages: list[StageMetrics] = Field(
        default_factory=list, description="Per-stage timing breakdown"
    )
    warnings: list[str] = Field(default_factory=list, description="Warnings recorded during run")
    errors: list[str] = Field(default_factory=list, description="Errors recorded during run")


class MetricsCollector:
    """Lightweight metrics collector tracking per-stage latency and execution diagnostics."""

    def __init__(self, device_used: str = "cpu") -> None:
        self._device_used = device_used
        self._stages: list[StageMetrics] = []
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._start_time: float = time.perf_counter()

    def record_stage(self, stage_name: str, duration_ms: float, status: str = "success") -> None:
        """Record timing for a single pipeline stage."""
        self._stages.append(
            StageMetrics(stage_name=stage_name, duration_ms=duration_ms, status=status)
        )

    def add_warning(self, warning: str) -> None:
        """Record a non-fatal execution warning message."""
        self._warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Record an execution error message."""
        self._errors.append(error)

    def get_metrics(self) -> ExecutionMetrics:
        """Calculate and return the final ExecutionMetrics payload."""
        total_duration = (time.perf_counter() - self._start_time) * 1000
        return ExecutionMetrics(
            total_duration_ms=round(total_duration, 2),
            device_used=self._device_used,
            stages=self._stages,
            warnings=self._warnings,
            errors=self._errors,
        )
