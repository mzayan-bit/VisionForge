"""VisionForge Base Vision Model Interface Specification."""

from abc import ABC, abstractmethod
from typing import Any

from visionforge.ai.schemas import InferenceResult
from visionforge.ai.types import ModelMetadata, ModelStatus


class BaseVisionModel(ABC):
    """Abstract Base Class defining the contract for all VisionForge computer vision models.

    Every model integrated into VisionForge must inherit from this class and implement
    the required lifecycle, metadata, and prediction methods.
    """

    def __init__(self) -> None:
        self._status: ModelStatus = ModelStatus.UNINITIALIZED
        self._device: str = "cpu"

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return the declarative metadata specification for this model."""
        ...

    @property
    def status(self) -> ModelStatus:
        """Return the current operational status of this model instance."""
        return self._status

    @property
    def device(self) -> str:
        """Return the currently bound compute device."""
        return self._device

    @abstractmethod
    async def initialize() -> None:
        """Perform initial setup, config parsing, and checkpoint path verification.

        Does NOT load heavy weights into memory.
        """
        ...

    @abstractmethod
    async def load(self, device: str | None = None) -> None:
        """Load model weights and computational graph into target compute memory."""
        ...

    @abstractmethod
    async def predict(self, inputs: Any, **kwargs: Any) -> InferenceResult[Any]:
        """Execute model inference on provided input modality payload."""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Unload model weights from compute device memory and release VRAM/RAM."""
        ...

    def health(self) -> dict[str, Any]:
        """Return operational health metrics and diagnostic state for this model."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "status": self.status.value,
            "device": self.device,
            "memory": self.metadata.memory_requirements.model_dump(),
        }

    @abstractmethod
    async def cleanup() -> None:
        """Perform final resource cleanup during application shutdown."""
        ...
