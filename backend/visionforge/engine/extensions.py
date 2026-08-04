"""Vision Engine Extension Hooks and Plugin Abstractions."""

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from visionforge.ai.types import TaskType
from visionforge.engine.context import ExecutionContext

logger = logging.getLogger("visionforge.engine.extensions")


class ExtensionRegistry:
    """Registry for custom pipeline stages, task factories, and output processors."""

    def __init__(self) -> None:
        self._custom_stages: list[Any] = []
        self._task_factories: dict[TaskType, Callable[..., Any]] = {}
        self._post_processors: dict[TaskType, list[Callable[[ExecutionContext, Any], Any]]] = {}

    def register_stage(self, stage: Any, position: int | None = None) -> None:
        """Register a custom pipeline stage into the execution pipeline."""
        if position is not None and 0 <= position <= len(self._custom_stages):
            self._custom_stages.insert(position, stage)
        else:
            self._custom_stages.append(stage)
        logger.info(
            "Registered custom pipeline stage '%s'",
            getattr(stage, "name", str(stage)),
        )

    def register_task_factory(self, task_type: TaskType, factory_fn: Callable[..., Any]) -> None:
        """Register a custom task factory function for a specific TaskType."""
        self._task_factories[task_type] = factory_fn
        logger.info("Registered custom task factory for TaskType '%s'", task_type.value)

    def register_post_processor(
        self,
        task_type: TaskType,
        processor_fn: Callable[[ExecutionContext, Any], Any],
    ) -> None:
        """Register a post-processing hook for a specific TaskType."""
        if task_type not in self._post_processors:
            self._post_processors[task_type] = []
        self._post_processors[task_type].append(processor_fn)
        logger.info("Registered custom post-processor for TaskType '%s'", task_type.value)

    def get_custom_stages(self) -> list[Any]:
        """Return list of registered custom pipeline stages."""
        return list(self._custom_stages)

    def get_task_factory(self, task_type: TaskType) -> Callable[..., Any] | None:
        """Return task factory function for task_type if registered."""
        return self._task_factories.get(task_type)

    def get_post_processors(
        self, task_type: TaskType
    ) -> list[Callable[[ExecutionContext, Any], Any]]:
        """Return post-processor hook functions for task_type."""
        return list(self._post_processors.get(task_type, []))

    def clear(self) -> None:
        """Clear all registered extensions."""
        self._custom_stages.clear()
        self._task_factories.clear()
        self._post_processors.clear()


@lru_cache
def get_extension_registry() -> ExtensionRegistry:
    """Return a cached singleton instance of ExtensionRegistry."""
    return ExtensionRegistry()
