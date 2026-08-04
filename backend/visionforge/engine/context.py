"""Vision Engine Execution Context Specification."""

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from visionforge.ai.base import BaseVisionModel
from visionforge.ai.types import TaskType
from visionforge.core.config import VisionForgeSettings, get_settings


class ExecutionContext(BaseModel):
    """Context object encapsulating request state, hardware target, settings, and metadata.

    Passed through every pipeline stage without relying on global variables.
    """

    model_config = {"arbitrary_types_allowed": True}

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request tracing identifier",
    )
    task_id: str = Field(
        default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}",
        description="Unique vision task execution identifier",
    )
    task_type: TaskType = Field(description="Computer vision task domain classification")
    settings: VisionForgeSettings = Field(
        default_factory=get_settings,
        description="Application settings instance",
    )
    device: str = Field(default="cpu", description="Compute device target name")
    model_name: str | None = Field(
        default=None,
        description="Optional explicit model name requested for execution",
    )
    model_instance: BaseVisionModel | None = Field(
        default=None,
        description="Resolved model instance bound during execution",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific execution options and parameters",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime execution metadata and telemetry",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 creation timestamp",
    )
