"""Unit test suite for VisionForge Vision Engine package."""

from typing import Any

import pytest

from visionforge.ai.types import TaskType
from visionforge.engine.context import ExecutionContext
from visionforge.engine.exceptions import TaskValidationError
from visionforge.engine.extensions import ExtensionRegistry
from visionforge.engine.manager import GenericVisionTask, TaskManager, TaskNotFoundError
from visionforge.engine.metrics import MetricsCollector
from visionforge.engine.pipeline import EnginePipeline, PipelineStage
from visionforge.engine.runner import VisionEngine
from visionforge.engine.task import TaskState


def test_execution_context_creation():
    """Verify ExecutionContext instantiation and default field values."""
    ctx = ExecutionContext(
        task_type=TaskType.DETECTION,
        device="mps",
        options={"confidence": 0.5},
    )
    assert ctx.task_type == TaskType.DETECTION
    assert ctx.device == "mps"
    assert ctx.options["confidence"] == 0.5
    assert ctx.request_id is not None
    assert ctx.task_id.startswith("task_")


def test_task_state_transitions():
    """Verify TaskState lifecycle transitions."""
    ctx = ExecutionContext(task_type=TaskType.SEGMENTATION)
    task = GenericVisionTask(ctx)

    assert task.state == TaskState.CREATED
    task.update_state(TaskState.EXECUTING)
    assert task.state == TaskState.EXECUTING

    task.cancel()
    assert task.state == TaskState.CANCELLED


def test_task_manager_lifecycle():
    """Verify TaskManager task creation, tracking, lookup, stats, and cancellation."""
    mgr = TaskManager()
    ctx = mgr.create_context(task_type=TaskType.OCR, device="cpu")

    task = mgr.create_task(ctx)
    assert task.task_id == ctx.task_id
    assert mgr.get_task(task.task_id) is task
    assert mgr.get_stats()["total_tasks"] == 1

    # Task cancellation
    assert mgr.cancel_task(task.task_id) is True
    assert task.state == TaskState.CANCELLED
    assert mgr.get_stats()["cancelled"] == 1

    # Task not found lookup error
    with pytest.raises(TaskNotFoundError):
        mgr.get_task("non-existent-task-id")

    mgr.clear()
    assert mgr.get_stats()["total_tasks"] == 0


def test_metrics_collector():
    """Verify MetricsCollector records stage timings and diagnostics."""
    collector = MetricsCollector(device_used="mps")
    collector.record_stage("validation", 1.5, status="success")
    collector.record_stage("model_execution", 42.3, status="success")
    collector.add_warning("Low confidence threshold")
    collector.add_error("Optional post-processor skipped")

    metrics = collector.get_metrics()
    assert metrics.device_used == "mps"
    assert len(metrics.stages) == 2
    assert metrics.stages[0].stage_name == "validation"
    assert len(metrics.warnings) == 1
    assert len(metrics.errors) == 1
    assert metrics.total_duration_ms >= 0


@pytest.mark.asyncio
async def test_pipeline_stages_and_execution():
    """Verify EnginePipeline stage execution flow."""
    ctx = ExecutionContext(task_type=TaskType.DETECTION)
    metrics_collector = MetricsCollector(device_used="cpu")
    pipeline = EnginePipeline()

    result = await pipeline.run(
        ctx,
        input_payload={"image": "sample.jpg"},
        metrics_collector=metrics_collector,
    )
    assert result.success is True
    assert result.data == {"image": "sample.jpg"}
    assert result.metadata is not None
    assert result.metadata.device_used == "cpu"

    metrics = metrics_collector.get_metrics()
    assert len(metrics.stages) == 5
    stage_names = [s.stage_name for s in metrics.stages]
    assert stage_names == [
        "validation",
        "pre_processing",
        "model_execution",
        "post_processing",
        "result_formatting",
    ]


@pytest.mark.asyncio
async def test_pipeline_validation_failure():
    """Verify validation failure raises TaskValidationError."""
    ctx = ExecutionContext(task_type=TaskType.DETECTION)
    metrics_collector = MetricsCollector(device_used="cpu")
    pipeline = EnginePipeline()

    with pytest.raises(TaskValidationError):
        await pipeline.run(
            ctx,
            input_payload=None,
            metrics_collector=metrics_collector,
        )


def test_extension_registry():
    """Verify plugin extension hooks registration."""
    ext_reg = ExtensionRegistry()

    class CustomStage(PipelineStage):
        @property
        def name(self) -> str:
            return "custom_filter"

        async def process(self, context: ExecutionContext, payload: Any) -> Any:
            return payload

    stage = CustomStage()
    ext_reg.register_stage(stage)
    assert len(ext_reg.get_custom_stages()) == 1
    assert ext_reg.get_custom_stages()[0].name == "custom_filter"

    ext_reg.clear()
    assert len(ext_reg.get_custom_stages()) == 0


@pytest.mark.asyncio
async def test_vision_engine_orchestrator():
    """Verify VisionEngine.run_task orchestrates task execution end-to-end."""
    engine = VisionEngine()
    result = await engine.run_task(
        task_type=TaskType.DEPTH_ESTIMATION,
        payload={"input_frame": "frame_001.png"},
        device="cpu",
    )

    assert result.success is True
    assert result.data == {"input_frame": "frame_001.png"}
    assert engine.get_engine_stats()["task_stats"]["completed"] == 1


@pytest.mark.asyncio
async def test_vision_engine_error_recovery():
    """Verify VisionEngine returns structured error on model resolution failure."""
    engine = VisionEngine()
    result = await engine.run_task(
        task_type=TaskType.DETECTION,
        payload={"image": "test.jpg"},
        model_name="non-existent-model",
        device="cpu",
    )

    assert result.success is False
    assert result.error is not None
    assert "MODEL_RESOLUTION_FAILED" in result.error.code
