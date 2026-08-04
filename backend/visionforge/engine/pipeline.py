"""Vision Engine Generic Execution Pipeline Architecture."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from visionforge.ai.registry import get_model_registry
from visionforge.ai.schemas import ExecutionMetadata, InferenceResult
from visionforge.engine.context import ExecutionContext
from visionforge.engine.exceptions import (
    ModelResolutionError,
    PipelineExecutionError,
    TaskValidationError,
)
from visionforge.engine.metrics import MetricsCollector
from visionforge.engine.task import TaskState

logger = logging.getLogger("visionforge.engine.pipeline")


class PipelineStage(ABC):
    """Abstract Base Class defining a stage in the Vision Engine pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this pipeline stage."""
        ...

    @property
    def state_on_enter(self) -> TaskState | None:
        """TaskState to transition task to when entering this stage."""
        return None

    @abstractmethod
    async def process(self, context: ExecutionContext, payload: Any) -> Any:
        """Process payload data through this pipeline stage."""
        ...


class ValidationStage(PipelineStage):
    """Validates execution context parameters, task inputs, and option specs."""

    @property
    def name(self) -> str:
        return "validation"

    @property
    def state_on_enter(self) -> TaskState:
        return TaskState.VALIDATING

    async def process(self, context: ExecutionContext, payload: Any) -> Any:
        if payload is None:
            raise TaskValidationError("Input payload cannot be None")
        return payload


class PreProcessingStage(PipelineStage):
    """Pre-processing abstraction stage for input normalization."""

    @property
    def name(self) -> str:
        return "pre_processing"

    @property
    def state_on_enter(self) -> TaskState:
        return TaskState.PREPROCESSING

    async def process(self, context: ExecutionContext, payload: Any) -> Any:
        return payload


class ModelExecutionStage(PipelineStage):
    """Model resolution and execution stage."""

    @property
    def name(self) -> str:
        return "model_execution"

    @property
    def state_on_enter(self) -> TaskState:
        return TaskState.EXECUTING

    async def process(self, context: ExecutionContext, payload: Any) -> Any:
        registry = get_model_registry()

        # 1. Resolve explicit or default model
        model_name = context.model_name
        if model_name:
            if not registry.contains(model_name):
                raise ModelResolutionError(
                    f"Requested model '{model_name}' is not registered in ModelRegistry"
                )
            model = registry.get(model_name)
            context.model_instance = model

            inference_res = await model.predict(payload, **context.options)
            return inference_res.data
        else:
            matching_models = registry.list_models(task=context.task_type)
            if matching_models:
                model = registry.get(matching_models[0].name)
                context.model_instance = model
                context.model_name = model.metadata.name
                inference_res = await model.predict(payload, **context.options)
                return inference_res.data

        return payload


class PostProcessingStage(PipelineStage):
    """Post-processing stage for filtering and formatting output tensors."""

    @property
    def name(self) -> str:
        return "post_processing"

    @property
    def state_on_enter(self) -> TaskState:
        return TaskState.POSTPROCESSING

    async def process(self, context: ExecutionContext, payload: Any) -> Any:
        return payload


class ResultFormattingStage(PipelineStage):
    """Formats pipeline output into standardized InferenceResult envelope."""

    @property
    def name(self) -> str:
        return "result_formatting"

    @property
    def state_on_enter(self) -> TaskState:
        return TaskState.FORMATTING

    async def process(self, context: ExecutionContext, payload: Any) -> Any:
        return payload


class EnginePipeline:
    """Sequential execution pipeline for processing vision tasks through ordered stages."""

    def __init__(self, stages: list[PipelineStage] | None = None):
        self._stages: list[PipelineStage] = stages or [
            ValidationStage(),
            PreProcessingStage(),
            ModelExecutionStage(),
            PostProcessingStage(),
            ResultFormattingStage(),
        ]

    def add_stage(self, stage: PipelineStage, position: int | None = None) -> None:
        """Add a stage to the pipeline."""
        if position is not None and 0 <= position <= len(self._stages):
            self._stages.insert(position, stage)
        else:
            self._stages.append(stage)

    async def run(
        self, context: ExecutionContext, input_payload: Any, metrics_collector: MetricsCollector
    ) -> InferenceResult[Any]:
        """Run payload through all pipeline stages sequentially."""
        current_payload = input_payload
        model_name = context.model_name or "vision-engine-core"
        model_version = (
            context.model_instance.metadata.version if context.model_instance else "1.0.0"
        )

        for stage in self._stages:
            stage_start = time.perf_counter()
            try:
                logger.debug(
                    "Executing pipeline stage '%s' for task '%s'", stage.name, context.task_id
                )
                current_payload = await stage.process(context, current_payload)
                stage_duration_ms = (time.perf_counter() - stage_start) * 1000
                metrics_collector.record_stage(stage.name, stage_duration_ms, status="success")
            except Exception as exc:
                stage_duration_ms = (time.perf_counter() - stage_start) * 1000
                metrics_collector.record_stage(stage.name, stage_duration_ms, status="failed")
                metrics_collector.add_error(str(exc))

                if isinstance(
                    exc, (TaskValidationError, ModelResolutionError, PipelineExecutionError)
                ):
                    raise exc

                raise PipelineExecutionError(stage_name=stage.name, reason=str(exc)) from exc

        exec_metrics = metrics_collector.get_metrics()
        exec_meta = ExecutionMetadata(
            model_name=model_name,
            model_version=model_version,
            device_used=context.device,
            execution_time_ms=exec_metrics.total_duration_ms,
        )

        return InferenceResult(
            success=True,
            message="Task pipeline execution completed successfully",
            data=current_payload,
            metadata=exec_meta,
            warnings=exec_metrics.warnings,
            error=None,
        )
