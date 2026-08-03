"""VisionForge Model Cache Management System."""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from visionforge.core.config import get_settings

logger = logging.getLogger("visionforge.ai.cache")


class CacheStats(BaseModel):
    """Statistical summary of disk cache usage."""

    cache_directory: str = Field(description="Expanded absolute path to model cache root directory")
    exists: bool = Field(description="True if cache directory exists on filesystem")
    total_files: int = Field(default=0, description="Total number of cached checkpoint files")
    total_size_bytes: int = Field(default=0, description="Total disk space consumed in bytes")
    total_size_mb: float = Field(default=0.0, description="Total disk space consumed in megabytes")


class CacheManager:
    """Manages disk space, checkpoint file locations, and cleanup for cached AI models."""

    def __init__(self, cache_dir: str | None = None) -> None:
        raw_path = cache_dir or get_settings().model_cache_dir
        self._cache_root = Path(raw_path).expanduser().resolve()

    @property
    def cache_root(self) -> Path:
        """Return expanded absolute Path object to cache root directory."""
        return self._cache_root

    def ensure_cache_dir(self) -> Path:
        """Ensure the cache root directory exists on disk."""
        self._cache_root.mkdir(parents=True, exist_ok=True)
        return self._cache_root

    def get_model_cache_dir(self, model_name: str, version: str = "1.0.0") -> Path:
        """Return the target cache directory Path for a specific model and version."""
        safe_name = model_name.replace("/", "_").replace(":", "_")
        safe_version = version.replace("/", "_")
        model_dir = self._cache_root / safe_name / safe_version
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_cache_stats(self) -> CacheStats:
        """Calculate and return cache usage statistics."""
        if not self._cache_root.exists():
            return CacheStats(
                cache_directory=str(self._cache_root),
                exists=False,
                total_files=0,
                total_size_bytes=0,
                total_size_mb=0.0,
            )

        total_files = 0
        total_size = 0

        for file_path in self._cache_root.rglob("*"):
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size

        size_mb = round(total_size / (1024 * 1024), 2)

        return CacheStats(
            cache_directory=str(self._cache_root),
            exists=True,
            total_files=total_files,
            total_size_bytes=total_size,
            total_size_mb=size_mb,
        )

    def clear_cache(self, model_name: str | None = None) -> int:
        """Delete cached files for a specific model or all models. Returns total bytes freed."""
        if not self._cache_root.exists():
            return 0

        freed_bytes = 0

        if model_name:
            safe_name = model_name.replace("/", "_").replace(":", "_")
            target_dir = self._cache_root / safe_name
            if target_dir.exists():
                for f in target_dir.rglob("*"):
                    if f.is_file():
                        freed_bytes += f.stat().st_size
                        f.unlink()
                logger.info(
                    "Cleared cache for model '%s' (%d bytes freed)",
                    model_name,
                    freed_bytes,
                )
        else:
            for f in self._cache_root.rglob("*"):
                if f.is_file():
                    freed_bytes += f.stat().st_size
                    f.unlink()
            logger.info("Cleared entire VisionForge model cache (%d bytes freed)", freed_bytes)

        return freed_bytes


@lru_cache
def get_cache_manager() -> CacheManager:
    """Return a cached singleton instance of CacheManager."""
    return CacheManager()
