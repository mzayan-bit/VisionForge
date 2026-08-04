"""VisionForge Model Manager — Validation Logic."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from visionforge.core.exceptions import VisionForgeException
from visionforge.models.metadata import InstalledModelMetadata

logger = logging.getLogger("visionforge.models.validation")

# Allowed characters: lowercase alphanumeric, hyphens, underscores, dots
_MODEL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")

# Semantic version pattern: major.minor.patch with optional pre-release
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+([a-zA-Z0-9._-]*)?$")


class ModelValidationError(VisionForgeException):
    """Raised when model validation checks fail."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(
            message=f"Model validation failed: {message}",
            code="MODEL_VALIDATION_FAILED",
            status_code=400,
            details=details,
        )


def validate_model_name(name: str) -> None:
    """Validate that a model name is well-formed.

    Raises:
        ModelValidationError: If the name is invalid.
    """
    if not name:
        raise ModelValidationError("Model name cannot be empty")

    if not _MODEL_NAME_PATTERN.match(name):
        raise ModelValidationError(
            f"Invalid model name '{name}'. "
            "Must be 1-128 chars, start with alphanumeric, "
            "and contain only lowercase letters, digits, hyphens, underscores, dots."
        )


def validate_version(version: str) -> None:
    """Validate that a version string follows semantic versioning.

    Raises:
        ModelValidationError: If the version is invalid.
    """
    if not version:
        raise ModelValidationError("Version string cannot be empty")

    if not _VERSION_PATTERN.match(version):
        raise ModelValidationError(
            f"Invalid version '{version}'. Must follow semantic versioning (e.g. '1.0.0')."
        )


def validate_metadata_file(path: Path) -> InstalledModelMetadata:
    """Read and validate a metadata JSON file on disk.

    Raises:
        ModelValidationError: If the file is missing, unreadable, or has invalid schema.
    """
    if not path.is_file():
        raise ModelValidationError(f"Metadata file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelValidationError(f"Corrupted metadata file: {exc}") from exc

    try:
        return InstalledModelMetadata(**data)
    except Exception as exc:
        raise ModelValidationError(f"Invalid metadata schema: {exc}") from exc


def validate_model_directory(model_dir: Path) -> list[str]:
    """Validate that a model directory exists and contains files.

    Returns a list of warning strings (empty if healthy).

    Raises:
        ModelValidationError: If the directory is missing or empty.
    """
    warnings: list[str] = []

    if not model_dir.exists():
        raise ModelValidationError(f"Model directory does not exist: {model_dir}")

    if not model_dir.is_dir():
        raise ModelValidationError(f"Path is not a directory: {model_dir}")

    files = list(model_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())

    if file_count == 0:
        raise ModelValidationError(f"Model directory is empty: {model_dir}")

    return warnings
