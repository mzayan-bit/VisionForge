"""VisionForge Model Manager — Central Model Management Orchestrator."""

import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from visionforge.core.exceptions import VisionForgeException
from visionforge.models.fs import (
    bytes_to_mb,
    directory_size_bytes,
    ensure_directory,
    safe_delete_directory,
)
from visionforge.models.metadata import InstalledModelMetadata, InstallStatus
from visionforge.models.storage import ModelStorage, get_model_storage
from visionforge.models.validation import (
    ModelValidationError,
    validate_metadata_file,
    validate_model_name,
    validate_version,
)

logger = logging.getLogger("visionforge.models.manager")


class ModelAlreadyExistsError(VisionForgeException):
    """Raised when attempting to install a model that already exists."""

    def __init__(self, name: str, version: str):
        super().__init__(
            message=f"Model '{name}' v{version} is already installed",
            code="MODEL_ALREADY_EXISTS",
            status_code=409,
        )


class ModelNotInstalledError(VisionForgeException):
    """Raised when looking up a model that is not installed."""

    def __init__(self, name: str):
        super().__init__(
            message=f"Model '{name}' is not installed",
            code="MODEL_NOT_INSTALLED",
            status_code=404,
        )


class ModelManager:
    """Central orchestrator for model discovery, installation, validation, and lifecycle.

    Manages the on-disk model storage, persisted metadata, and provides a clean
    interface for API endpoints to interact with the model catalogue.
    """

    def __init__(self, storage: ModelStorage | None = None) -> None:
        self._storage = storage or get_model_storage()

    @property
    def storage(self) -> ModelStorage:
        """Return the underlying ModelStorage instance."""
        return self._storage

    def initialize(self) -> None:
        """Initialize model storage directories on disk."""
        self._storage.initialize()

    # ─── Discovery & Listing ─────────────────────────────────────────

    def list_installed(self) -> list[InstalledModelMetadata]:
        """Return metadata for all installed models by reading persisted metadata files."""
        results: list[InstalledModelMetadata] = []
        meta_dir = self._storage.metadata_dir
        if not meta_dir.exists():
            return results

        for meta_file in sorted(meta_dir.glob("*.json")):
            try:
                meta = validate_metadata_file(meta_file)
                results.append(meta)
            except ModelValidationError:
                logger.warning("Skipping corrupted metadata file: %s", meta_file)

        return results

    def get_model(self, name: str) -> InstalledModelMetadata:
        """Retrieve metadata for a specific installed model.

        Raises:
            ModelNotInstalledError: If the model is not installed.
        """
        meta_path = self._storage.get_metadata_path(name)
        if not meta_path.is_file():
            raise ModelNotInstalledError(name)

        return validate_metadata_file(meta_path)

    def is_installed(self, name: str) -> bool:
        """Check whether a model is installed."""
        return self._storage.get_metadata_path(name).is_file()

    # ─── Installation ────────────────────────────────────────────────

    def prepare_install(
        self, metadata: InstalledModelMetadata
    ) -> InstalledModelMetadata:
        """Validate and prepare a model for installation (pre-download phase).

        Returns the metadata with status updated to INSTALLING and install_path set.

        Raises:
            ModelValidationError: If name or version is invalid.
            ModelAlreadyExistsError: If the model is already installed.
        """
        validate_model_name(metadata.name)
        validate_version(metadata.version)

        if self.is_installed(metadata.name):
            raise ModelAlreadyExistsError(metadata.name, metadata.version)

        model_dir = self._storage.get_model_dir(metadata.name, metadata.version)
        ensure_directory(model_dir)

        metadata.status = InstallStatus.INSTALLING
        metadata.install_path = str(model_dir)
        metadata.installed_at = datetime.now(UTC).isoformat()
        metadata.updated_at = datetime.now(UTC).isoformat()

        self._persist_metadata(metadata)
        logger.info("Prepared installation for model '%s' v%s", metadata.name, metadata.version)
        return metadata

    def finalize_install(self, name: str) -> InstalledModelMetadata:
        """Mark an installed model as INSTALLED and calculate disk usage.

        Called after model files have been placed in the install directory.

        Raises:
            ModelNotInstalledError: If the model metadata does not exist.
        """
        meta = self.get_model(name)
        model_dir = self._storage.get_model_dir(meta.name, meta.version)

        size_bytes = directory_size_bytes(model_dir)
        meta.disk_size_bytes = size_bytes
        meta.disk_size_mb = bytes_to_mb(size_bytes)
        meta.status = InstallStatus.INSTALLED
        meta.updated_at = datetime.now(UTC).isoformat()

        self._persist_metadata(meta)
        logger.info(
            "Finalized installation of '%s' v%s (%.2f MB)",
            meta.name,
            meta.version,
            meta.disk_size_mb,
        )
        return meta

    # ─── Removal ─────────────────────────────────────────────────────

    def remove_model(self, name: str) -> int:
        """Remove an installed model, its directory, and metadata. Returns bytes freed.

        Raises:
            ModelNotInstalledError: If the model is not installed.
        """
        meta = self.get_model(name)
        model_dir = self._storage.get_model_dir(meta.name, meta.version)

        freed = safe_delete_directory(model_dir)

        # Clean up empty parent directory
        parent = model_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

        # Remove metadata file
        meta_path = self._storage.get_metadata_path(name)
        if meta_path.is_file():
            meta_path.unlink()

        logger.info("Removed model '%s' (%.2f MB freed)", name, bytes_to_mb(freed))
        return freed

    # ─── Validation ──────────────────────────────────────────────────

    def validate_model(self, name: str) -> dict[str, Any]:
        """Validate an installed model's metadata and directory integrity.

        Returns a diagnostic report dict.
        """
        report: dict[str, Any] = {"name": name, "valid": True, "warnings": [], "errors": []}

        # 1. Metadata check
        try:
            meta = self.get_model(name)
        except (ModelNotInstalledError, ModelValidationError) as exc:
            report["valid"] = False
            report["errors"].append(str(exc))
            return report

        # 2. Directory existence
        model_dir = self._storage.get_model_dir(meta.name, meta.version)
        if not model_dir.exists():
            report["valid"] = False
            report["errors"].append(f"Model directory missing: {model_dir}")
            return report

        # 3. Disk size recalculation
        actual_size = directory_size_bytes(model_dir)
        if meta.disk_size_bytes > 0 and actual_size != meta.disk_size_bytes:
            report["warnings"].append(
                f"Disk size mismatch: metadata={meta.disk_size_bytes}, actual={actual_size}"
            )

        report["metadata"] = meta.model_dump()
        return report

    # ─── Stats ───────────────────────────────────────────────────────

    def get_manager_status(self) -> dict[str, Any]:
        """Return aggregated model manager status for health endpoints."""
        storage_stats = self._storage.get_storage_stats()
        installed = self.list_installed()
        return {
            "status": "ready",
            "installed_models": len(installed),
            "storage": storage_stats.model_dump(),
        }

    # ─── Internal Helpers ────────────────────────────────────────────

    def _persist_metadata(self, meta: InstalledModelMetadata) -> None:
        """Write model metadata to its JSON file on disk."""
        meta_path = self._storage.get_metadata_path(meta.name)
        ensure_directory(meta_path.parent)
        meta_path.write_text(
            json.dumps(meta.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )


@lru_cache
def get_model_manager() -> ModelManager:
    """Return a cached singleton instance of ModelManager."""
    return ModelManager()
