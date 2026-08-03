"""Unit test suite for VisionForge AI Core package."""

from typing import Any

import pytest

from visionforge.ai.base import BaseVisionModel
from visionforge.ai.cache import CacheManager
from visionforge.ai.device import DeviceManager, DeviceType
from visionforge.ai.registry import DuplicateModelError, ModelNotFoundError, ModelRegistry
from visionforge.ai.schemas import ExecutionMetadata, InferenceResult
from visionforge.ai.types import (
    InputType,
    MemoryRequirements,
    ModelMetadata,
    ModelStatus,
    OutputType,
    TaskType,
)


class MockTestModel(BaseVisionModel):
    """Mock vision model implementation for contract & lifecycle unit testing."""

    def __init__(self, name: str = "mock-detector-v1", task: TaskType = TaskType.DETECTION):
        super().__init__()
        self._meta = ModelMetadata(
            name=name,
            version="1.0.0",
            author="VisionForge Test Suite",
            task=task,
            license="MIT",
            supported_input_types=[InputType.IMAGE],
            supported_output_types=[OutputType.BOUNDING_BOXES],
            memory_requirements=MemoryRequirements(vram_mb=512, ram_mb=1024, disk_space_mb=250),
            device_support=["cpu", "cuda", "mps"],
            description="Mock vision model for testing AI Core contracts",
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._meta

    async def initialize(self) -> None:
        self._status = ModelStatus.INITIALIZED

    async def load(self, device: str | None = None) -> None:
        self._status = ModelStatus.LOADING
        self._device = device or "cpu"
        self._status = ModelStatus.READY

    async def predict(self, inputs: Any, **kwargs: Any) -> InferenceResult[Any]:
        exec_meta = ExecutionMetadata(
            model_name=self.metadata.name,
            model_version=self.metadata.version,
            device_used=self.device,
            execution_time_ms=1.23,
        )
        return InferenceResult(
            success=True,
            message="Mock prediction completed",
            data={"boxes": [[0, 0, 100, 100]], "scores": [0.99]},
            metadata=exec_meta,
        )

    async def unload(self) -> None:
        self._status = ModelStatus.UNLOADED

    async def cleanup(self) -> None:
        self._status = ModelStatus.UNINITIALIZED


def test_model_metadata_instantiation():
    """Verify ModelMetadata schema defaults and serialization."""
    meta = ModelMetadata(
        name="test-segmenter",
        task=TaskType.SEGMENTATION,
        supported_output_types=[OutputType.SEGMENTATION_MASKS],
    )
    assert meta.name == "test-segmenter"
    assert meta.task == TaskType.SEGMENTATION
    assert meta.status == ModelStatus.UNINITIALIZED
    assert "cpu" in meta.device_support


@pytest.mark.asyncio
async def test_base_vision_model_lifecycle():
    """Verify model lifecycle state transitions."""
    model = MockTestModel(name="test-lifecycle-model")
    assert model.status == ModelStatus.UNINITIALIZED

    await model.initialize()
    assert model.status == ModelStatus.INITIALIZED

    await model.load(device="cpu")
    assert model.status == ModelStatus.READY
    assert model.device == "cpu"

    result = await model.predict(inputs="test_image.png")
    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.model_name == "test-lifecycle-model"

    health = model.health()
    assert health["name"] == "test-lifecycle-model"
    assert health["status"] == ModelStatus.READY.value

    await model.unload()
    assert model.status == ModelStatus.UNLOADED

    await model.cleanup()
    assert model.status == ModelStatus.UNINITIALIZED


def test_model_registry_operations():
    """Verify ModelRegistry register, unregister, duplicate error, and task filtering."""
    registry = ModelRegistry()
    model1 = MockTestModel(name="yolo-detector", task=TaskType.DETECTION)
    model2 = MockTestModel(name="sam-segmenter", task=TaskType.SEGMENTATION)

    # 1. Register models
    registry.register(model1)
    registry.register(model2)
    assert registry.count() == 2
    assert registry.contains("yolo-detector")

    # 2. Duplicate registration error validation
    with pytest.raises(DuplicateModelError):
        registry.register(model1)

    # 3. Lookup model
    retrieved = registry.get("yolo-detector")
    assert retrieved.metadata.name == "yolo-detector"

    with pytest.raises(ModelNotFoundError):
        registry.get("non-existent-model")

    # 4. List and filter by TaskType
    all_models = registry.list_models()
    assert len(all_models) == 2

    detection_models = registry.list_models(task=TaskType.DETECTION)
    assert len(detection_models) == 1
    assert detection_models[0].name == "yolo-detector"

    # 5. Unregister
    removed = registry.unregister("yolo-detector")
    assert removed.metadata.name == "yolo-detector"
    assert registry.count() == 1

    registry.clear()
    assert registry.count() == 0


def test_device_manager():
    """Verify hardware detection and device resolution."""
    dev_mgr = DeviceManager()
    caps = dev_mgr.get_hardware_capabilities()

    assert DeviceType.CPU in caps.available_devices
    assert dev_mgr.get_optimal_device() in caps.available_devices
    assert dev_mgr.resolve_device("auto") == dev_mgr.get_optimal_device()
    assert dev_mgr.resolve_device("cpu") == DeviceType.CPU


def test_cache_manager(tmp_path):
    """Verify CacheManager directory creation and statistics calculation."""
    cache_mgr = CacheManager(cache_dir=str(tmp_path / "model_cache"))

    stats = cache_mgr.get_cache_stats()
    assert stats.total_files == 0

    model_dir = cache_mgr.get_model_cache_dir("test-model", "1.0.0")
    assert model_dir.exists()

    # Create dummy checkpoint file
    dummy_ckpt = model_dir / "weights.bin"
    dummy_ckpt.write_bytes(b"0" * 1024 * 1024)  # 1 MB

    stats_after = cache_mgr.get_cache_stats()
    assert stats_after.total_files == 1
    assert stats_after.total_size_mb >= 1.0

    freed = cache_mgr.clear_cache(model_name="test-model")
    assert freed >= 1024 * 1024
