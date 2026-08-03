# VisionForge AI Core Architecture Specification

## 🧠 System Overview & Philosophy

VisionForge AI Core is designed around a fundamental architecture principle: **Zero Model-Specific Sprawl**. 

No AI model implementation, inference code, or checkpoint logic is hardcoded directly into application handlers or routing layers. Instead, AI Core provides a unified, strongly-typed orchestration layer that every vision foundation model adheres to—ensuring long-term maintainability, clean extensibility, and standardized execution contracts.

---

## 🏛️ AI Core Package Layout

```text
backend/visionforge/ai/
├── __init__.py       # Top-level AI Core exports
├── base.py           # BaseVisionModel Abstract Base Class & ModelStatus
├── types.py          # ModelMetadata, TaskType, InputType, OutputType, MemoryRequirements
├── registry.py       # ModelRegistry & get_model_registry singleton
├── schemas.py        # Generic InferenceResult[T], ExecutionMetadata, & InferenceError
├── device.py         # DeviceManager, DeviceType enum, & HardwareCapabilities
└── cache.py          # CacheManager, CacheStats, & disk space management
```

---

## 🔄 Model Lifecycle & Base Class Contract

Every vision model integrated into VisionForge inherits from `visionforge.ai.base.BaseVisionModel`:

```python
from abc import ABC, abstractmethod
from typing import Any
from visionforge.ai.schemas import InferenceResult
from visionforge.ai.types import ModelMetadata, ModelStatus

class BaseVisionModel(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Declarative model metadata specification."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Path resolution and environment verification (no weight loading)."""
        ...

    @abstractmethod
    async def load(self, device: str | None = None) -> None:
        """Load model weights and computational graph into compute device RAM/VRAM."""
        ...

    @abstractmethod
    async def predict(self, inputs: Any, **kwargs: Any) -> InferenceResult[Any]:
        """Execute model inference on provided input modality payload."""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Unload model weights from memory and free GPU VRAM/RAM."""
        ...

    def health(self) -> dict[str, Any]:
        """Return operational health telemetry and status."""
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Perform final resource teardown during application shutdown."""
        ...
```

---

## 🏷️ Model Metadata Specification

Models expose strong metadata definitions via `visionforge.ai.types.ModelMetadata`:

- **Name**: Unique string key (e.g. `'grounding-dino-v1'`).
- **Task**: Classification via `TaskType` enum (`DETECTION`, `SEGMENTATION`, `DEPTH_ESTIMATION`, `OCR`, `VLM`, `RETRIEVAL`, `VIDEO_UNDERSTANDING`, `RECONSTRUCTION_3D`).
- **Supported Modalities**: List of `InputType` (`IMAGE`, `VIDEO`, `TEXT_PROMPT`, `TENSOR`, `MULTIMODAL`) and `OutputType` (`BOUNDING_BOXES`, `SEGMENTATION_MASKS`, `DEPTH_MAP`, `TEXT`, `EMBEDDINGS`, `VIDEO_ANNOTATIONS`, `POINT_CLOUD`).
- **Memory Footprint**: `MemoryRequirements` specifying VRAM, RAM, and disk storage requirements.
- **Hardware Backends**: List of supported execution targets (`"cpu"`, `"cuda"`, `"mps"`).

---

## 📚 Model Registry (`ModelRegistry`)

The `ModelRegistry` acts as the central discovery and lifecycle manager:

- **Automatic Registration**: `registry.register(model_instance)` validates name uniqueness and registers models.
- **Duplicate Prevention**: Attempts to register duplicate model identifiers raise `DuplicateModelError`.
- **Task Filtering**: `registry.list_models(task=TaskType.DETECTION)` filters metadata specs by computer vision task domain.
- **Lookup & Removal**: `registry.get(name)` and `registry.unregister(name)` manage runtime access.

---

## ⚡ Device & Cache Management

### Device Abstraction (`DeviceManager`)
Detects host hardware compute targets without hardcoding dependencies:
- **Supported Backends**: `CUDA` (NVIDIA GPUs), `MPS` (Apple Silicon Metal), `CPU`.
- **Optimal Device Selection**: Automatically resolves `"auto"` to the fastest available compute hardware target on the host system.

### Cache Manager (`CacheManager`)
Manages model weight storage directories and disk consumption:
- **Path Resolution**: Maps models to `~/.cache/visionforge/models/<model_name>/<version>/`.
- **Telemetry**: Computes total cached files and disk consumption in megabytes (`CacheStats`).
- **Cache Purging**: `cache_mgr.clear_cache(model_name)` safely cleans up unused checkpoint directories.

---

## 📦 Standardized Inference Result Envelope

Inference outputs return a generic `InferenceResult[DataT]` envelope:

```json
{
  "success": true,
  "message": "Inference completed successfully",
  "data": {
    "boxes": [[10, 20, 150, 200]],
    "scores": [0.98],
    "labels": ["person"]
  },
  "metadata": {
    "model_name": "yolo-v8-detector",
    "model_version": "1.0.0",
    "device_used": "mps",
    "execution_time_ms": 14.2,
    "timestamp": "2026-08-03T22:50:00+00:00"
  },
  "error": null,
  "warnings": []
}
```

---

## 🛠️ Integration Guide for Future Models

To integrate a new vision model into VisionForge:

1. Create a model module in `visionforge/models/<your_model_name>/`.
2. Inherit from `BaseVisionModel` and define your model's `ModelMetadata`.
3. Implement `initialize()`, `load()`, `predict()`, `unload()`, and `cleanup()`.
4. Register your model instance using `get_model_registry().register(your_model)`.
