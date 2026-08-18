"""VisionForge Model Manager — Installed Model Metadata Schema."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from visionforge.ai.types import InputType, MemoryRequirements, OutputType, TaskType


class InstallStatus(StrEnum):
    """Installation lifecycle state of a managed model."""

    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    INSTALLED = "installed"
    VALIDATING = "validating"
    CORRUPTED = "corrupted"
    REMOVING = "removing"


class ModelSource(BaseModel):
    """Origin and download information for a model."""

    provider: str = Field(
        default="local", description="Source provider (e.g. 'huggingface', 'local', 'url')"
    )
    repository: str = Field(default="", description="Source repository identifier or URL")
    download_url: str = Field(default="", description="Direct download URL for model checkpoint")
    sha256: str = Field(
        default="", description="Expected SHA-256 checksum for integrity verification"
    )


class InstalledModelMetadata(BaseModel):
    """Full metadata record for an installed VisionForge model."""

    # Identity
    name: str = Field(description="Unique model identifier name")
    version: str = Field(default="1.0.0", description="Semantic model version")
    author: str = Field(default="VisionForge", description="Model author or maintainer")
    description: str = Field(default="", description="Model description")
    license: str = Field(default="MIT", description="License classification")

    # Classification
    task: TaskType = Field(description="Primary computer vision task type")
    framework: str = Field(
        default="generic", description="ML framework (e.g. 'pytorch', 'onnx', 'tensorflow')"
    )
    supported_input_types: list[InputType] = Field(
        default_factory=lambda: [InputType.IMAGE],
        description="Supported input modalities",
    )
    supported_output_types: list[OutputType] = Field(
        default_factory=list,
        description="Supported output modalities",
    )

    # Hardware
    supported_devices: list[str] = Field(
        default_factory=lambda: ["cpu", "cuda", "mps"],
        description="Supported compute acceleration backends (e.g. 'cpu', 'cuda', 'mps')",
    )
    device_support: list[str] = Field(
        default_factory=lambda: ["cpu", "cuda", "mps"],
        description="Supported compute backends (backward-compatible alias)",
    )
    memory_requirements: MemoryRequirements = Field(
        default_factory=MemoryRequirements,
        description="Resource footprint specification",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_devices(cls, data: Any) -> Any:
        """Ensure supported_devices and device_support are canonicalized from legacy or incoming dicts."""
        if isinstance(data, dict):
            devices = data.get("supported_devices") or data.get("device_support")
            if not devices or not isinstance(devices, list):
                devices = ["cpu", "cuda", "mps"]
            data["supported_devices"] = list(devices)
            data["device_support"] = list(devices)
        return data

    # Source
    source: ModelSource = Field(
        default_factory=ModelSource,
        description="Origin and download provenance",
    )

    # Installation State
    status: InstallStatus = Field(
        default=InstallStatus.AVAILABLE,
        description="Current installation lifecycle state",
    )
    install_path: str = Field(
        default="", description="Absolute filesystem path to installed model directory"
    )
    disk_size_bytes: int = Field(default=0, description="Installed disk consumption in bytes")
    disk_size_mb: float = Field(default=0.0, description="Installed disk consumption in megabytes")

    # Timestamps
    installed_at: str = Field(default="", description="ISO 8601 installation timestamp")
    last_used_at: str = Field(default="", description="ISO 8601 last inference execution timestamp")
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 last metadata update timestamp",
    )
