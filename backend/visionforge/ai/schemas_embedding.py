"""VisionForge AI Core Image Embedding Data Models and Response Schemas."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    """Metadata extracted from the input image."""

    width: int = Field(description="Image width in pixels")
    height: int = Field(description="Image height in pixels")
    format: str = Field(default="RGB", description="Image format (e.g. JPEG, PNG, WEBP)")
    mode: str = Field(default="RGB", description="Color space mode (e.g. RGB, RGBA, L)")
    aspect_ratio: float = Field(description="Aspect ratio (width / height)")
    file_size_bytes: int = Field(default=0, description="Raw file size in bytes if available")


class VectorStats(BaseModel):
    """Statistical summary of the generated embedding vector."""

    min: float = Field(description="Minimum vector component value")
    max: float = Field(description="Maximum vector component value")
    mean: float = Field(description="Arithmetic mean of vector components")
    std: float = Field(description="Standard deviation of vector components")
    non_zero_count: int = Field(description="Count of non-zero elements")


class ImageEmbeddingResult(BaseModel):
    """Reusable, production-grade Image Embedding data object."""

    embedding: list[float] = Field(
        description="L2-normalized dense embedding vector representation"
    )
    dimension: int = Field(default=768, description="Dimensionality of the embedding vector")
    model: str = Field(description="Identifier name of the generating vision model")
    version: str = Field(default="1.0.0", description="Version of the vision model")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 creation timestamp",
    )
    execution_time_ms: float = Field(
        description="Total end-to-end embedding generation execution time in milliseconds"
    )
    loading_time_ms: float = Field(
        default=0.0, description="Model loading time in milliseconds if lazy loaded"
    )
    device_used: str = Field(default="cpu", description="Compute backend used (cpu, cuda, mps)")
    l2_norm: float = Field(default=1.0, description="Calculated L2 norm of the embedding vector")
    image_metadata: ImageMetadata = Field(description="Metadata of the processed image")
    vector_stats: VectorStats = Field(description="Statistical summary of the vector components")
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value metadata for future index & DB compatibility",
    )
