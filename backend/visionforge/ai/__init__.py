"""VisionForge AI Core Package."""

from visionforge.ai.base import BaseVisionModel
from visionforge.ai.cache import CacheManager, CacheStats, get_cache_manager
from visionforge.ai.device import (
    DeviceManager,
    DeviceType,
    HardwareCapabilities,
    get_device_manager,
)
from visionforge.ai.registry import (
    DuplicateModelError,
    ModelNotFoundError,
    ModelRegistry,
    get_model_registry,
)
from visionforge.ai.schemas import ExecutionMetadata, InferenceError, InferenceResult
from visionforge.ai.types import (
    InputType,
    MemoryRequirements,
    ModelMetadata,
    ModelStatus,
    OutputType,
    TaskType,
)

__all__ = [
    "BaseVisionModel",
    "CacheManager",
    "CacheStats",
    "DeviceManager",
    "DeviceType",
    "DuplicateModelError",
    "ExecutionMetadata",
    "HardwareCapabilities",
    "InferenceError",
    "InferenceResult",
    "InputType",
    "MemoryRequirements",
    "ModelMetadata",
    "ModelNotFoundError",
    "ModelRegistry",
    "ModelStatus",
    "OutputType",
    "TaskType",
    "get_cache_manager",
    "get_device_manager",
    "get_model_registry",
]
