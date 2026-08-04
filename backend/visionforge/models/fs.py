"""VisionForge Model Manager — Filesystem Utility Functions."""

import hashlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("visionforge.models.fs")

# 64KB read buffer for hash calculation
_HASH_CHUNK_SIZE = 65536


def safe_model_name(name: str) -> str:
    """Sanitize a model name for safe filesystem directory usage."""
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def ensure_directory(path: Path) -> Path:
    """Create directory and all parents if they do not exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def directory_size_bytes(path: Path) -> int:
    """Calculate total disk usage in bytes for all files under a directory recursively."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def directory_file_count(path: Path) -> int:
    """Count total number of files under a directory recursively."""
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file())


def bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes, rounded to 2 decimal places."""
    return round(size_bytes / (1024 * 1024), 2)


def safe_delete_directory(path: Path) -> int:
    """Safely delete a directory and all contents. Returns bytes freed."""
    if not path.exists():
        return 0

    freed = directory_size_bytes(path)
    shutil.rmtree(path)
    logger.info("Deleted directory '%s' (%.2f MB freed)", path, bytes_to_mb(freed))
    return freed


def safe_delete_file(path: Path) -> int:
    """Safely delete a single file. Returns bytes freed."""
    if not path.is_file():
        return 0

    freed = path.stat().st_size
    path.unlink()
    return freed


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def normalize_path(raw_path: str) -> Path:
    """Expand user home (~) and resolve to an absolute Path."""
    return Path(raw_path).expanduser().resolve()
