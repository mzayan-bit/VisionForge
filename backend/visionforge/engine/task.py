"""Vision Engine Task Abstractions and Lifecycle States."""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from visionforge.ai.schemas import InferenceResult
from visionforge.ai.types import TaskType
from visionforge.engine.context import ExecutionContext


class TaskState(StrEnum):
    """Execution lifecycle state of a vision task."""

    CREATED = "created"
    QUEUED = "queued"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    EXECUTING = "executing"
    POSTPROCESSING = "postprocessing"
    FORMATTING = "formatting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BaseVisionTask(ABC):
    """Abstract Base Class for all Computer Vision tasks executing within VisionForge.

    Encapsulates task execution lifecycle, state transitions, context, and result handling.
    """

    def __init__(self, context: ExecutionContext):
        self._context = context
        self._state: TaskState = TaskState.CREATED
        self._result: InferenceResult[Any] | None = None
        self._error: Exception | None = None

    @property
    def task_id(self) -> str:
        """Unique task identifier string."""
        return self._context.task_id

    @property
    def task_type(self) -> TaskType:
        """Computer vision task classification."""
        return self._context.task_type

    @property
    def state(self) -> TaskState:
        """Current lifecycle execution state."""
        return self._state

    @property
    def context(self) -> ExecutionContext:
        """Execution context instance."""
        return self._context

    @property
    def result(self) -> InferenceResult[Any] | None:
        """Inference result envelope populated upon completion."""
        return self._result

    @property
    def error(self) -> Exception | None:
        """Recorded error instance if task failed."""
        return self._error

    def update_state(self, new_state: TaskState) -> None:
        """Transition task to a new lifecycle execution state."""
        self._state = new_state

    def set_result(self, result: InferenceResult[Any]) -> None:
        """Set task result and transition state to COMPLETED."""
        self._result = result
        self._state = TaskState.COMPLETED

    def set_error(self, error: Exception) -> None:
        """Set task error and transition state to FAILED."""
        self._error = error
        self._state = TaskState.FAILED

    def cancel(self) -> None:
        """Cancel task execution and transition state to CANCELLED."""
        if self._state not in (TaskState.COMPLETED, TaskState.FAILED):
            self._state = TaskState.CANCELLED

    @abstractmethod
    async def execute(self, payload: Any) -> InferenceResult[Any]:
        """Execute the task pipeline flow for the provided input payload."""
        ...
