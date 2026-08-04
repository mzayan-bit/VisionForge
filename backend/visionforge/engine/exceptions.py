"""Vision Engine Exception Hierarchy."""

from typing import Any

from visionforge.core.exceptions import VisionForgeException


class EngineException(VisionForgeException):
    """Base exception for all Vision Engine execution errors."""

    def __init__(
        self,
        message: str = "A Vision Engine execution error occurred",
        code: str = "ENGINE_ERROR",
        status_code: int = 500,
        details: list[dict[str, Any]] | None = None,
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status_code,
            details=details,
        )


class TaskValidationError(EngineException):
    """Raised when task validation fails prior to execution."""

    def __init__(
        self,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ):
        super().__init__(
            message=f"Task validation failed: {message}",
            code="TASK_VALIDATION_FAILED",
            status_code=400,
            details=details,
        )


class ModelResolutionError(EngineException):
    """Raised when a requested model cannot be resolved or is incompatible."""

    def __init__(self, message: str):
        super().__init__(
            message=f"Model resolution error: {message}",
            code="MODEL_RESOLUTION_FAILED",
            status_code=404,
        )


class PipelineExecutionError(EngineException):
    """Raised when a stage in the execution pipeline fails."""

    def __init__(self, stage_name: str, reason: str):
        super().__init__(
            message=f"Pipeline stage '{stage_name}' failed: {reason}",
            code="PIPELINE_EXECUTION_FAILED",
            status_code=500,
        )


class TaskCancelledError(EngineException):
    """Raised when a task execution is cancelled."""

    def __init__(self, task_id: str):
        super().__init__(
            message=f"Task '{task_id}' execution was cancelled",
            code="TASK_CANCELLED",
            status_code=409,
        )
