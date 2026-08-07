"""VisionForge Visual Search Engine — Fast Similarity Search Orchestrator."""

import logging
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from visionforge.ai.schemas_embedding import ImageEmbeddingResult
from visionforge.ai.types import TaskType
from visionforge.engine.runner import VisionEngine, get_vision_engine
from visionforge.memory.index import VisualMemoryIndex, get_visual_memory_index
from visionforge.search.similarity import (
    SimilarityMetric,
    compute_matrix_cosine_similarity,
    compute_matrix_euclidean_distance,
    validate_embedding_vector,
)

logger = logging.getLogger("visionforge.search.engine")


class SearchResultItem(BaseModel):
    """Descriptor for a single ranked visual memory match."""

    rank: int = Field(description="1-indexed similarity rank order")
    id: str = Field(description="Record ID of the visual memory match")
    similarity_score: float = Field(
        description="Calculated similarity percentage score (0.0 to 1.0)"
    )
    distance: float = Field(description="Distance measurement value")
    image_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata of the matched image"
    )
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    indexed_at: str = Field(description="ISO timestamp when item was indexed")
    embedding_model: str = Field(
        default="siglip-base-patch16-224", description="Model used to encode candidate vector"
    )


class SearchResponsePayload(BaseModel):
    """Result payload returned by Visual Search Engine."""

    search_id: str = Field(description="Unique transaction ID for this search execution")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO query execution timestamp",
    )
    results: list[SearchResultItem] = Field(description="Ranked list of top-K visual matches")
    candidate_count: int = Field(description="Total visual memory items evaluated")
    returned_count: int = Field(description="Total matches satisfying threshold cutoff")
    metric_used: SimilarityMetric = Field(description="Distance metric used for ranking")
    model_used: str = Field(
        default="siglip-base-patch16-224", description="Embedding model used for query"
    )
    embedding_time_ms: float = Field(
        default=0.0, description="Embedding generation duration in milliseconds"
    )
    search_time_ms: float = Field(
        default=0.0, description="Vector similarity search duration in milliseconds"
    )
    total_execution_time_ms: float = Field(
        description="Total end-to-end search duration in milliseconds"
    )
    query_info: dict[str, Any] = Field(
        default_factory=dict, description="Metadata descriptor of the search query input"
    )


class VisualSearchEngine:
    """Production-grade similarity search engine operating over Visual Memory Store."""

    def __init__(
        self,
        memory_index: VisualMemoryIndex | None = None,
        vision_engine: VisionEngine | None = None,
    ):
        self._memory_index = memory_index or get_visual_memory_index()
        self._vision_engine = vision_engine or get_vision_engine()

    async def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute visual search given an input query image file."""
        embed_start = time.perf_counter()

        # 1. Run query image through embedding pipeline
        res = await self._vision_engine.run_task(
            task_type=TaskType.RETRIEVAL,
            payload=image_bytes,
            model_name="siglip-base-patch16-224",
        )

        if not res.success or res.data is None:
            err = res.error.message if res.error else "Embedding extraction failed for query"
            raise RuntimeError(f"Visual search query embedding failed: {err}")

        embedding_res: ImageEmbeddingResult = res.data
        embedding_time_ms = (time.perf_counter() - embed_start) * 1000

        meta = embedding_res.image_metadata
        query_info = {
            "type": "image",
            "resolution": f"{meta.width}x{meta.height}",
            "format": meta.format,
            "file_size_bytes": meta.file_size_bytes,
        }

        return self.search_by_vector(
            query_vector=embedding_res.embedding,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
            query_info=query_info,
            embedding_time_ms=round(embedding_time_ms, 2),
            model_used=embedding_res.model,
        )

    def search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
        query_info: dict[str, Any] | None = None,
        embedding_time_ms: float = 0.0,
        model_used: str = "siglip-base-patch16-224",
    ) -> SearchResponsePayload:
        """Execute high-speed vectorized similarity search over indexed visual memory."""
        start_time = time.perf_counter()
        search_id = f"srch_{uuid.uuid4().hex[:10]}"

        # Validate query vector
        q_arr = validate_embedding_vector(query_vector)

        # Validate Top-K parameters
        clean_top_k = max(1, min(top_k, 100))
        clean_threshold = max(0.0, min(threshold, 0.95))

        matrix, ids = self._memory_index.get_matrix_and_ids()
        candidate_count = len(ids)

        if candidate_count == 0:
            search_time_ms = (time.perf_counter() - start_time) * 1000
            total_time_ms = embedding_time_ms + search_time_ms
            return SearchResponsePayload(
                search_id=search_id,
                results=[],
                candidate_count=0,
                returned_count=0,
                metric_used=metric,
                model_used=model_used,
                embedding_time_ms=embedding_time_ms,
                search_time_ms=round(search_time_ms, 2),
                total_execution_time_ms=round(total_time_ms, 2),
                query_info=query_info or {"type": "vector", "dimension": len(q_arr)},
            )

        # Compute metric matrix scores
        if metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT):
            scores = compute_matrix_cosine_similarity(matrix, q_arr)
            distances = 1.0 - scores
        elif metric == SimilarityMetric.EUCLIDEAN:
            distances, scores = compute_matrix_euclidean_distance(matrix, q_arr)
        else:
            raise ValueError(f"Unsupported similarity metric '{metric}'")

        # Top-K ranking (sort indices by score descending)
        ranked_indices = np.argsort(-scores)

        items: list[SearchResultItem] = []
        rank_counter = 1
        for idx in ranked_indices:
            score_val = float(scores[idx])
            if score_val < clean_threshold:
                continue

            record_id = ids[idx]
            rec = self._memory_index.get_record(record_id)

            items.append(
                SearchResultItem(
                    rank=rank_counter,
                    id=rec.id,
                    similarity_score=round(score_val, 4),
                    distance=round(float(distances[idx]), 4),
                    image_metadata=rec.image_metadata,
                    tags=rec.tags,
                    indexed_at=rec.indexed_at,
                    embedding_model=model_used,
                )
            )
            rank_counter += 1

            if len(items) >= clean_top_k:
                break

        search_time_ms = (time.perf_counter() - start_time) * 1000
        total_time_ms = embedding_time_ms + search_time_ms

        return SearchResponsePayload(
            search_id=search_id,
            results=items,
            candidate_count=candidate_count,
            returned_count=len(items),
            metric_used=metric,
            model_used=model_used,
            embedding_time_ms=embedding_time_ms,
            search_time_ms=round(search_time_ms, 2),
            total_execution_time_ms=round(total_time_ms, 2),
            query_info=query_info or {"type": "vector", "dimension": len(q_arr)},
        )


@lru_cache
def get_visual_search_engine() -> VisualSearchEngine:
    """Return a cached singleton instance of VisualSearchEngine."""
    return VisualSearchEngine()
