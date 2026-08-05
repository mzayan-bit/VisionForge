"""VisionForge Model Registry System."""

import logging
from functools import lru_cache

from visionforge.ai.base import BaseVisionModel
from visionforge.ai.types import ModelMetadata, TaskType
from visionforge.core.exceptions import VisionForgeException

logger = logging.getLogger("visionforge.ai.registry")


class DuplicateModelError(VisionForgeException):
    """Raised when attempting to register a model name that already exists in registry."""

    def __init__(self, name: str):
        super().__init__(
            message=f"Model '{name}' is already registered in ModelRegistry",
            code="DUPLICATE_MODEL_REGISTRATION",
            status_code=400,
        )


class ModelNotFoundError(VisionForgeException):
    """Raised when looking up a model name that is not registered."""

    def __init__(self, name: str):
        super().__init__(
            message=f"Model '{name}' was not found in ModelRegistry",
            code="MODEL_NOT_FOUND",
            status_code=404,
        )


class ModelRegistry:
    """Central registry for discovering, registering, and retrieving VisionForge vision models."""

    def __init__(self) -> None:
        self._models: dict[str, BaseVisionModel] = {}

    def register(self, model: BaseVisionModel) -> None:
        """Register a new vision model instance into the registry.

        Raises:
            DuplicateModelError: If a model with the same name is already registered.
        """
        name = model.metadata.name
        if name in self._models:
            logger.warning("Failed duplicate model registration attempt for '%s'", name)
            raise DuplicateModelError(name)

        self._models[name] = model
        logger.info(
            "Registered model '%s' v%s (task=%s)",
            name,
            model.metadata.version,
            model.metadata.task.value,
        )

    def unregister(self, name: str) -> BaseVisionModel:
        """Unregister a vision model by name and return it.

        Raises:
            ModelNotFoundError: If the model name is not registered.
        """
        if name not in self._models:
            raise ModelNotFoundError(name)

        removed_model = self._models.pop(name)
        logger.info("Unregistered model '%s' from ModelRegistry", name)
        return removed_model

    def get(self, name: str) -> BaseVisionModel:
        """Retrieve a registered model instance by name.

        Raises:
            ModelNotFoundError: If the model name is not registered.
        """
        if name not in self._models:
            raise ModelNotFoundError(name)

        return self._models[name]

    def list_models(self, task: TaskType | None = None) -> list[ModelMetadata]:
        """List metadata specifications for models, optionally filtered by TaskType."""
        models_metadata = [model.metadata for model in self._models.values()]
        if task is not None:
            return [meta for meta in models_metadata if meta.task == task]
        return models_metadata

    def contains(self, name: str) -> bool:
        """Check if a model name is registered."""
        return name in self._models

    def count(self) -> int:
        """Return total count of registered models."""
        return len(self._models)

    def clear(self) -> None:
        """Clear all registered models from registry."""
        self._models.clear()
        logger.info("Cleared all models from ModelRegistry")


@lru_cache
def get_model_registry() -> ModelRegistry:
    """Return a cached singleton instance of ModelRegistry."""
    registry = ModelRegistry()
    from visionforge.ai.models.siglip import SigLIPEmbeddingModel

    siglip = SigLIPEmbeddingModel()
    registry.register(siglip)
    return registry
