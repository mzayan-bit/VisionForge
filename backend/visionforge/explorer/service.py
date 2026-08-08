"""Embedding Explorer Service Layer & Cache Management."""

import hashlib
import logging
import time
import uuid
from functools import lru_cache
from typing import Any

import numpy as np

from visionforge.explorer.analysis import apply_kmeans, compute_outlier_scores
from visionforge.explorer.reduction import compute_projection
from visionforge.explorer.schemas import (
    ExplorerDatasetPayload,
    ExplorerPoint,
    ProjectionMethod,
    ProjectionRequest,
)
from visionforge.memory.index import VisualMemoryIndex, get_visual_memory_index

logger = logging.getLogger("visionforge.explorer.service")


class ExplorerCache:
    """In-memory local projection cache keyed by dataset state and hyperparameter signature."""

    def __init__(self, max_entries: int = 50):
        self._max_entries = max_entries
        self._cache: dict[str, ExplorerDatasetPayload] = {}

    def get(self, cache_key: str) -> ExplorerDatasetPayload | None:
        """Retrieve cached projection payload if hit."""
        payload = self._cache.get(cache_key)
        if payload:
            logger.info("Explorer projection cache hit for key '%s'", cache_key[:12])
            # Return copy with cached=True
            copy_payload = payload.model_copy()
            copy_payload.cached = True
            return copy_payload
        return None

    def set(self, cache_key: str, payload: ExplorerDatasetPayload) -> None:
        """Cache projection payload."""
        if len(self._cache) >= self._max_entries:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = payload

    def clear(self) -> None:
        """Purge projection cache."""
        self._cache.clear()


class EmbeddingExplorerService:
    """Service layer for embedding dimensionality reduction, clustering, and outlier analysis."""

    def __init__(
        self,
        memory_index: VisualMemoryIndex | None = None,
        cache: ExplorerCache | None = None,
    ):
        self._memory_index = memory_index or get_visual_memory_index()
        self._cache = cache or ExplorerCache()

    def get_explorer_stats(self) -> dict[str, Any]:
        """Return dataset statistics relevant for embedding exploration."""
        mem_stats = self._memory_index.get_stats()
        return {
            "total_records": mem_stats.total_records,
            "vector_dimension": mem_stats.vector_dimension,
            "memory_size_mb": mem_stats.memory_size_mb,
            "supported_projections": ["pca", "tsne"],
            "supported_clustering": ["kmeans"],
        }

    def generate_projection(self, req: ProjectionRequest) -> ExplorerDatasetPayload:
        """Generate 2D/3D projection, K-Means clustering, and outlier scores."""
        start_time = time.perf_counter()

        matrix, record_ids = self._memory_index.get_matrix_and_ids()
        n_samples = len(record_ids)

        if n_samples == 0:
            exec_time = (time.perf_counter() - start_time) * 1000
            dataset_id = f"proj_{uuid.uuid4().hex[:10]}"
            from visionforge.explorer.schemas import (
                ClusteringMeta,
                DimensionalityReductionMeta,
            )

            return ExplorerDatasetPayload(
                dataset_id=dataset_id,
                points=[],
                total_points=0,
                reduction_meta=DimensionalityReductionMeta(
                    method=req.method,
                    n_components=req.n_components,
                    original_dimension=768,
                    perplexity=req.perplexity if req.method == ProjectionMethod.TSNE else None,
                    random_seed=req.random_seed,
                ),
                clustering_meta=ClusteringMeta(n_clusters=0, cluster_sizes={}, inertia=0.0),
                execution_time_ms=round(exec_time, 2),
                cached=False,
            )

        # Generate unique cache key
        dataset_sig = self._compute_dataset_signature(matrix, record_ids)
        cache_key = self._compute_cache_key(dataset_sig, req)

        cached_payload = self._cache.get(cache_key)
        if cached_payload:
            return cached_payload

        # 1. Execute Dimensionality Reduction (PCA or t-SNE)
        coords, reduction_meta = compute_projection(
            matrix=matrix,
            method=req.method,
            n_components=req.n_components,
            perplexity=req.perplexity,
            random_seed=req.random_seed,
        )

        # 2. Execute K-Means Clustering over projected space (or raw matrix)
        cluster_labels, centroids, clustering_meta = apply_kmeans(
            matrix=coords,
            n_clusters=req.n_clusters,
            random_seed=req.random_seed,
        )

        # 3. Compute Outlier Scores & Distance to Centroid
        distances, outlier_scores = compute_outlier_scores(
            matrix=coords,
            labels=cluster_labels,
            centroids=centroids,
        )

        # 4. Construct ExplorerPoint array
        points: list[ExplorerPoint] = []
        for i, rid in enumerate(record_ids):
            rec = self._memory_index.get_record(rid)
            x_val = float(coords[i, 0])
            y_val = float(coords[i, 1])
            z_val = float(coords[i, 2]) if req.n_components == 3 else None

            points.append(
                ExplorerPoint(
                    id=rec.id,
                    x=round(x_val, 4),
                    y=round(y_val, 4),
                    z=round(z_val, 4) if z_val is not None else None,
                    image_metadata=rec.image_metadata,
                    tags=rec.tags,
                    embedding_model="siglip-base-patch16-224",
                    cluster_id=int(cluster_labels[i]),
                    outlier_score=float(outlier_scores[i]),
                    distance_to_centroid=float(distances[i]),
                )
            )

        exec_time_ms = (time.perf_counter() - start_time) * 1000
        dataset_id = f"proj_{uuid.uuid4().hex[:10]}"

        payload = ExplorerDatasetPayload(
            dataset_id=dataset_id,
            points=points,
            total_points=len(points),
            reduction_meta=reduction_meta,
            clustering_meta=clustering_meta,
            execution_time_ms=round(exec_time_ms, 2),
            cached=False,
        )

        # Save to cache
        self._cache.set(cache_key, payload)
        return payload

    def _compute_dataset_signature(self, matrix: np.ndarray, record_ids: list[str]) -> str:
        """Hash dataset matrix shape and record IDs to detect dataset changes."""
        id_str = ",".join(record_ids)
        matrix_sum = float(np.sum(matrix)) if matrix.size > 0 else 0.0
        raw = f"{len(record_ids)}_{matrix.shape}_{matrix_sum:.4f}_{id_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _compute_cache_key(self, dataset_sig: str, req: ProjectionRequest) -> str:
        """Construct deterministic cache key from parameters."""
        p = f"{req.method}_{req.n_components}_{req.perplexity}_{req.random_seed}_{req.n_clusters}"
        raw = f"{dataset_sig}_{p}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@lru_cache
def get_embedding_explorer_service() -> EmbeddingExplorerService:
    """Return a cached singleton instance of EmbeddingExplorerService."""
    return EmbeddingExplorerService()
