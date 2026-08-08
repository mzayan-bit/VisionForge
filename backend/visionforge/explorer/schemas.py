"""VisionForge Embedding Explorer Data Models and Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProjectionMethod(StrEnum):
    """Dimensionality reduction method options."""

    PCA = "pca"
    TSNE = "tsne"


class ExplorerPoint(BaseModel):
    """Point coordinate descriptor in projected 2D or 3D visual space."""

    id: str = Field(description="Visual memory record ID")
    x: float = Field(description="X coordinate component")
    y: float = Field(description="Y coordinate component")
    z: float | None = Field(default=None, description="Optional Z coordinate for 3D projections")
    image_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Associated image dimensions and parameters"
    )
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    embedding_model: str = Field(
        default="siglip-base-patch16-224", description="Model used for embedding extraction"
    )
    cluster_id: int = Field(default=0, description="K-Means cluster label assignment")
    outlier_score: float = Field(
        default=0.0, description="Normalized anomaly outlier score [0.0..1.0]"
    )
    distance_to_centroid: float = Field(
        default=0.0, description="Euclidean distance to assigned cluster centroid"
    )


class DimensionalityReductionMeta(BaseModel):
    """Execution telemetry for dimensionality reduction projection."""

    method: ProjectionMethod = Field(description="Reduction algorithm applied ('pca' or 'tsne')")
    n_components: int = Field(description="Target dimensions (2 or 3)")
    original_dimension: int = Field(default=768, description="Source embedding vector dimension")
    explained_variance_ratio: list[float] = Field(
        default_factory=list, description="Variance explained per component (PCA only)"
    )
    cumulative_explained_variance: float = Field(
        default=0.0, description="Total cumulative variance ratio (PCA only)"
    )
    perplexity: float | None = Field(
        default=None, description="Perplexity hyperparameter (t-SNE only)"
    )
    random_seed: int = Field(default=42, description="Random state seed for reproducibility")


class ClusteringMeta(BaseModel):
    """K-Means clustering configuration and summary telemetry."""

    method: str = Field(default="kmeans", description="Clustering algorithm")
    n_clusters: int = Field(description="Requested cluster count")
    cluster_sizes: dict[int, int] = Field(
        default_factory=dict, description="Point count per cluster"
    )
    inertia: float = Field(default=0.0, description="K-Means sum of squared distances")


class ExplorerDatasetPayload(BaseModel):
    """Complete explorer dataset payload returned to frontend visualization engine."""

    dataset_id: str = Field(description="Unique configuration projection ID")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO projection generation timestamp",
    )
    points: list[ExplorerPoint] = Field(description="Projected coordinates and metadata points")
    total_points: int = Field(description="Total points projected")
    reduction_meta: DimensionalityReductionMeta = Field(
        description="Projection algorithm telemetry"
    )
    clustering_meta: ClusteringMeta = Field(description="Clustering telemetry")
    execution_time_ms: float = Field(
        description="Total projection and clustering calculation duration in ms"
    )
    cached: bool = Field(
        default=False, description="Whether payload was retrieved from local cache"
    )


class ProjectionRequest(BaseModel):
    """Request payload for generating embedding space projection."""

    method: ProjectionMethod = Field(
        default=ProjectionMethod.PCA, description="Projection algorithm ('pca' or 'tsne')"
    )
    n_components: int = Field(default=2, ge=2, le=3, description="Projection dimensions (2 or 3)")
    perplexity: float = Field(
        default=30.0, ge=2.0, le=100.0, description="t-SNE perplexity parameter"
    )
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    n_clusters: int = Field(default=3, ge=1, le=20, description="Number of K-Means clusters")
    model_filter: str | None = Field(default=None, description="Optional model filter")
