"""Vision Engine Task Manager."""

import logging
from functools import lru_cache
from typing import Any

from visionforge.ai.types import TaskType
from visionforge.core.exceptions import VisionForgeException
from visionforge.engine.context import ExecutionContext
from visionforge.engine.task import BaseVisionTask, TaskState

logger = logging.getLogger("visionforge.engine.manager")


class TaskNotFoundError(VisionForgeException):
    """Raised when looking up a task ID that does not exist."""

    def __init__(self, task_id: str):
        super().__init__(
            message=f"Vision task with ID '{task_id}' was not found in TaskManager",
            code="TASK_NOT_FOUND",
            status_code=404,
        )


class GenericVisionTask(BaseVisionTask):
    """Concrete BaseVisionTask implementation for executing engine pipelines."""

    def __init__(self, context: ExecutionContext):
        super().__init__(context)

    async def execute(self, payload: Any) -> Any:
        """Execution is orchestrated through the EnginePipeline."""
        return payload


class TaskManager:
    """Lightweight manager for creating, tracking, and managing vision task lifecycle."""

    def __init__(self) -> None:
        self._tasks: dict[str, BaseVisionTask] = {}

    def create_context(
        self,
        task_type: TaskType,
        model_name: str | None = None,
        device: str = "cpu",
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> ExecutionContext:
        """Create and return a new ExecutionContext instance."""
        ctx_kwargs: dict[str, Any] = {
            "task_type": task_type,
            "device": device,
            "options": options or {},
        }
        if model_name:
            ctx_kwargs["model_name"] = model_name
        if request_id:
            ctx_kwargs["request_id"] = request_id

        return ExecutionContext(**ctx_kwargs)

    def create_task(self, context: ExecutionContext) -> BaseVisionTask:
        """Create and register a vision task instance bound to the execution context."""
        task = GenericVisionTask(context)
        self._tasks[task.task_id] = task
        logger.info(
            "Created task '%s' (type=%s, req_id=%s)",
            task.task_id,
            task.task_type.value,
            context.request_id,
        )
        return task

    def get_task(self, task_id: str) -> BaseVisionTask:
        """Retrieve a registered vision task by task_id.

        Raises:
            TaskNotFoundError: If task_id is not registered.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        return self._tasks[task_id]

    def list_tasks(self, state: TaskState | None = None) -> list[BaseVisionTask]:
        """List registered tasks, optionally filtered by TaskState."""
        tasks = list(self._tasks.values())
        if state is not None:
            return [t for t in tasks if t.state == state]
        return tasks

    def cancel_task(self, task_id: str) -> bool:
        """Cancel execution of a registered vision task by task_id."""
        task = self.get_task(task_id)
        if task.state in (TaskState.COMPLETED, TaskState.FAILED):
            return False

        task.cancel()
        logger.info("Cancelled task '%s'", task_id)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return aggregated summary statistics for managed tasks."""
        all_tasks = list(self._tasks.values())
        active_states = {
            TaskState.QUEUED,
            TaskState.VALIDATING,
            TaskState.PREPROCESSING,
            TaskState.EXECUTING,
            TaskState.POSTPROCESSING,
            TaskState.FORMATTING,
        }
        return {
            "total_tasks": len(all_tasks),
            "active": len([t for t in all_tasks if t.state in active_states]),
            "completed": len([t for t in all_tasks if t.state == TaskState.COMPLETED]),
            "failed": len([t for t in all_tasks if t.state == TaskState.FAILED]),
            "cancelled": len([t for t in all_tasks if t.state == TaskState.CANCELLED]),
        }

    def clear(self) -> None:
        """Clear all tasks from manager memory."""
        self._tasks.clear()


@lru_cache
def get_task_manager() -> TaskManager:
    """Return a cached singleton instance of TaskManager."""
    return TaskManager()
