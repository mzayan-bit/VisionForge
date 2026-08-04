"""VisionForge Model Manager — Storage Layout and Directory Management."""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from visionforge.core.config import get_settings
from visionforge.models.fs import (
    bytes_to_mb,
    directory_file_count,
    directory_size_bytes,
    ensure_directory,
    normalize_path,
    safe_model_name,
)

logger = logging.getLogger("visionforge.models.storage")


class StorageStats(BaseModel):
    """Aggregated storage usage statistics for the model directory tree."""

    root_directory: str = Field(description="Absolute path to storage root")
    exists: bool = Field(description="True if storage root exists on disk")
    total_files: int = Field(default=0, description="Total files across all subdirectories")
    total_size_bytes: int = Field(default=0, description="Total disk consumption in bytes")
    total_size_mb: float = Field(default=0.0, description="Total disk consumption in MB")
    models_count: int = Field(
        default=0, description="Number of model directories present"
    )


class ModelStorage:
    """Manages the on-disk directory layout for installed VisionForge models.

    Directory structure::

        <storage_root>/
        ├── models/           # Installed model checkpoint directories
        │   └── <model_name>/
        │       └── <version>/
        ├── downloads/        # Temporary in-progress download staging area
        ├── metadata/         # Persisted model metadata JSON files
        └── temp/             # Scratch / temporary working directory
    """

    def __init__(self, storage_root: str | None = None) -> None:
        raw = storage_root or get_settings().model_cache_dir
        self._root = normalize_path(raw)

    @property
    def root(self) -> Path:
        """Absolute resolved storage root path."""
        return self._root

    @property
    def models_dir(self) -> Path:
        """Path to installed model checkpoints directory."""
        return self._root / "models"

    @property
    def downloads_dir(self) -> Path:
        """Path to temporary download staging directory."""
        return self._root / "downloads"

    @property
    def metadata_dir(self) -> Path:
        """Path to persisted model metadata directory."""
        return self._root / "metadata"

    @property
    def temp_dir(self) -> Path:
        """Path to scratch / temporary working directory."""
        return self._root / "temp"

    def initialize(self) -> None:
        """Create the full storage directory tree if it does not exist."""
        for d in (self.models_dir, self.downloads_dir, self.metadata_dir, self.temp_dir):
            ensure_directory(d)
        logger.info("Initialized model storage at '%s'", self._root)

    def get_model_dir(self, model_name: str, version: str = "1.0.0") -> Path:
        """Return the directory path for a specific model and version."""
        safe = safe_model_name(model_name)
        return self.models_dir / safe / version

    def get_metadata_path(self, model_name: str) -> Path:
        """Return the metadata JSON file path for a model."""
        safe = safe_model_name(model_name)
        return self.metadata_dir / f"{safe}.json"

    def list_installed_model_dirs(self) -> list[str]:
        """Return names of all model directories present under models/."""
        if not self.models_dir.exists():
            return []
        return sorted(
            d.name for d in self.models_dir.iterdir() if d.is_dir()
        )

    def get_storage_stats(self) -> StorageStats:
        """Calculate and return aggregate storage usage statistics."""
        if not self._root.exists():
            return StorageStats(
                root_directory=str(self._root),
                exists=False,
            )

        total_bytes = directory_size_bytes(self._root)
        total_files = directory_file_count(self._root)
        model_count = len(self.list_installed_model_dirs())

        return StorageStats(
            root_directory=str(self._root),
            exists=True,
            total_files=total_files,
            total_size_bytes=total_bytes,
            total_size_mb=bytes_to_mb(total_bytes),
            models_count=model_count,
        )


@lru_cache
def get_model_storage() -> ModelStorage:
    """Return a cached singleton instance of ModelStorage."""
    return ModelStorage()
