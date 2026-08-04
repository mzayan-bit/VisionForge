"""VisionForge Model Manager Package."""

from visionforge.models.fs import (
    bytes_to_mb,
    calculate_sha256,
    directory_file_count,
    directory_size_bytes,
    ensure_directory,
    normalize_path,
    safe_delete_directory,
    safe_delete_file,
    safe_model_name,
)
from visionforge.models.manager import (
    ModelAlreadyExistsError,
    ModelManager,
    ModelNotInstalledError,
    get_model_manager,
)
from visionforge.models.metadata import (
    InstalledModelMetadata,
    InstallStatus,
    ModelSource,
)
from visionforge.models.storage import (
    ModelStorage,
    StorageStats,
    get_model_storage,
)
from visionforge.models.validation import (
    ModelValidationError,
    validate_metadata_file,
    validate_model_directory,
    validate_model_name,
    validate_version,
)

__all__ = [
    # Filesystem utilities
    "bytes_to_mb",
    "calculate_sha256",
    "directory_file_count",
    "directory_size_bytes",
    "ensure_directory",
    "normalize_path",
    "safe_delete_directory",
    "safe_delete_file",
    "safe_model_name",
    # Manager
    "ModelAlreadyExistsError",
    "ModelManager",
    "ModelNotInstalledError",
    "get_model_manager",
    # Metadata
    "InstallStatus",
    "InstalledModelMetadata",
    "ModelSource",
    # Storage
    "ModelStorage",
    "StorageStats",
    "get_model_storage",
    # Validation
    "ModelValidationError",
    "validate_metadata_file",
    "validate_model_directory",
    "validate_model_name",
    "validate_version",
]
