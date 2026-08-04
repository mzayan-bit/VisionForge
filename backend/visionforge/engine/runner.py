"""Vision Engine Central Execution Orchestrator."""

import logging
from functools import lru_cache
from typing import Any

from visionforge.ai.device import get_device_manager
from visionforge.ai.schemas import InferenceError, InferenceResult
from visionforge.ai.types import TaskType
from visionforge.engine.exceptions import EngineException
from visionforge.engine.extensions import get_extension_registry
from visionforge.engine.manager import TaskManager, get_task_manager
from visionforge.engine.metrics import MetricsCollector
from visionforge.engine.pipeline import EnginePipeline

logger = logging.getLogger("visionforge.engine.runner")


class VisionEngine:
    """Central execution engine operating system layer of VisionForge.

    Orchestrates task creation, context binding, model resolution, pipeline execution,
    metrics recording, and centralized error recovery for all computer vision tasks.
    """

    def __init__(
        self,
        task_manager: TaskManager | None = None,
        pipeline: EnginePipeline | None = None,
    ):
        self._task_manager = task_manager or get_task_manager()
        self._pipeline = pipeline or EnginePipeline()
        self._device_manager = get_device_manager()
        self._extension_registry = get_extension_registry()

    @property
    def task_manager(self) -> TaskManager:
        """Return task manager instance."""
        return self._task_manager

    @property
    def pipeline(self) -> EnginePipeline:
        """Return engine pipeline instance."""
        return self._pipeline

    async def run_task(
        self,
        task_type: TaskType,
        payload: Any,
        model_name: str | None = None,
        device: str | None = None,
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> InferenceResult[Any]:
        """Execute a vision task through the full engine lifecycle.

        Flow: Request -> Validation -> Task Creation -> Model Resolution -> Execution -> Response
        """
        # 1. Device Resolution
        target_device = self._device_manager.resolve_device(device or "auto").value

        # 2. Execution Context Creation
        context = self._task_manager.create_context(
            task_type=task_type,
            model_name=model_name,
            device=target_device,
            options=options or {},
            request_id=request_id,
        )

        # 3. Task Creation & Registration
        task = self._task_manager.create_task(context)
        metrics_collector = MetricsCollector(device_used=target_device)

        try:
            # 4. Pipeline Execution
            result = await self._pipeline.run(context, payload, metrics_collector)

            # 5. Task State Completion Update
            task.set_result(result)
            exec_time = result.metadata.execution_time_ms if result.metadata else 0.0
            logger.info("Task '%s' completed successfully in %.2fms", task.task_id, exec_time)
            return result

        except Exception as exc:
            logger.error("Task '%s' failed: %s", task.task_id, str(exc), exc_info=True)
            task.set_error(exc)
            metrics_collector.add_error(str(exc))

            # Centralized Error Recovery Path
            error_code = getattr(exc, "code", "ENGINE_EXECUTION_ERROR")
            details = getattr(exc, "details", None)

            if isinstance(exc, EngineException):
                err_payload = InferenceError(code=error_code, message=exc.message, details=details)
            else:
                err_payload = InferenceError(
                    code="INTERNAL_ENGINE_ERROR",
                    message=f"An unexpected engine error occurred: {str(exc)}",
                )

            return InferenceResult(
                success=False,
                message=f"Task '{task.task_id}' execution failed: {err_payload.message}",
                data=None,
                metadata=None,
                error=err_payload,
                warnings=metrics_collector.get_metrics().warnings,
            )

    def get_engine_stats(self) -> dict[str, Any]:
        """Return operational engine telemetry and task statistics."""
        return {
            "task_stats": self._task_manager.get_stats(),
            "optimal_device": self._device_manager.get_optimal_device().value,
            "available_devices": [d.value for d in self._device_manager.get_available_devices()],
            "custom_stages_count": len(self._extension_registry.get_custom_stages()),
        }


@lru_cache
def get_vision_engine() -> VisionEngine:
    """Return a cached singleton instance of VisionEngine."""
    return VisionEngine()
