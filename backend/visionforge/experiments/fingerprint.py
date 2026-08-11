"""VisionForge Cryptographic Fingerprinting & Artifact Checksum Utilities."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from visionforge.experiments.schemas import ArtifactReference, DatasetFingerprint

logger = logging.getLogger("visionforge.experiments.fingerprint")


def calculate_sha256(file_path: Path | str) -> str:
    """Calculate streaming SHA-256 cryptographic hash of a file."""
    p = Path(file_path).resolve()
    if not p.is_file():
        return "file_not_found"

    hasher = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as exc:
        logger.warning("Failed to hash file '%s': %s", p, exc)
        return "hash_error"


def create_dataset_fingerprint(
    dataset_id: str,
    version: str,
    manifest_data: dict[str, Any],
    preparation_id: str | None = None,
) -> DatasetFingerprint:
    """Generate a stable SHA-256 fingerprint for a dataset version manifest."""
    # 1. Manifest Hash
    raw_json = json.dumps(manifest_data, sort_keys=True, default=str)
    manifest_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()

    num_samples = int(manifest_data.get("num_samples", 0))
    num_classes = len(manifest_data.get("class_names", [])) or int(manifest_data.get("num_classes", 0))

    # 2. Combined Master Fingerprint Hash
    combined_payload = f"{dataset_id}:{version}:{preparation_id}:{num_samples}:{num_classes}:{manifest_hash}"
    fingerprint_hash = hashlib.sha256(combined_payload.encode("utf-8")).hexdigest()

    return DatasetFingerprint(
        dataset_id=dataset_id,
        version=version,
        preparation_id=preparation_id,
        num_samples=num_samples,
        num_classes=num_classes,
        manifest_sha256=manifest_hash,
        fingerprint_hash=fingerprint_hash,
    )


def create_artifact_reference(
    artifact_id: str,
    artifact_type: str,
    name: str,
    file_path: Path | str,
) -> ArtifactReference:
    """Create an ArtifactReference with calculated SHA-256 checksum and size in bytes."""
    p = Path(file_path).resolve()
    exists = p.is_file()
    size_bytes = p.stat().st_size if exists else 0
    checksum = calculate_sha256(p) if exists else None

    return ArtifactReference(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        name=name,
        path=str(p),
        sha256_checksum=checksum,
        size_bytes=size_bytes,
    )
