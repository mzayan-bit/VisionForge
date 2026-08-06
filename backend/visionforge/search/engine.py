"""VisionForge Visual Search Engine — Fast Similarity Search Orchestrator."""

import logging
import time
from enum import StrEnum
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from visionforge.ai.schemas_embedding import ImageEmbeddingResult
from visionforge.ai.types import TaskType
from visionforge.engine.runner import VisionEngine, get_vision_engine
from visionforge.memory.index import VisualMemoryIndex, get_visual_memory_index

logger = logging.getLogger("visionforge.search.engine")


class SimilarityMetric(StrEnum):
    """Vector distance and similarity measurement metric classification."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class SearchResultItem(BaseModel):
    """Descriptor for a single ranked visual memory match."""

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


class SearchResponsePayload(BaseModel):
    """Result payload returned by Visual Search Engine."""

    results: list[SearchResultItem] = Field(description="Ranked list of top-K visual matches")
    query_execution_time_ms: float = Field(
        description="Total query search duration in milliseconds"
    )
    candidate_count: int = Field(description="Total visual memory items evaluated")
    metric_used: SimilarityMetric = Field(description="Distance metric used for ranking")


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
        return self.search_by_vector(
            query_vector=embedding_res.embedding,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
        )

    def search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute high-speed vectorized similarity search over indexed visual memory."""
        start_time = time.perf_counter()

        matrix, ids = self._memory_index.get_matrix_and_ids()
        candidate_count = len(ids)

        if candidate_count == 0:
            exec_time = (time.perf_counter() - start_time) * 1000
            return SearchResponsePayload(
                results=[],
                query_execution_time_ms=round(exec_time, 2),
                candidate_count=0,
                metric_used=metric,
            )

        q_arr = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm  # Ensure normalized unit vector

        # Compute metric matrix scores
        if metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT):
            # Vectorized dot product dot(Matrix, q) -> shape (N,)
            scores = np.dot(matrix, q_arr)
            distances = 1.0 - scores
        elif metric == SimilarityMetric.EUCLIDEAN:
            # L2 norm distance sqrt(sum((V - q)^2))
            diffs = matrix - q_arr
            distances = np.linalg.norm(diffs, axis=1)
            # Map distance to similarity score in range [0, 1]
            scores = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unsupported similarity metric '{metric}'")

        # Top-K ranking (sort indices by score descending)
        ranked_indices = np.argsort(-scores)

        items: list[SearchResultItem] = []
        for idx in ranked_indices:
            score_val = float(scores[idx])
            if score_val < threshold:
                continue

            record_id = ids[idx]
            rec = self._memory_index.get_record(record_id)

            items.append(
                SearchResultItem(
                    id=rec.id,
                    similarity_score=round(score_val, 4),
                    distance=round(float(distances[idx]), 4),
                    image_metadata=rec.image_metadata,
                    tags=rec.tags,
                    indexed_at=rec.indexed_at,
                )
            )

            if len(items) >= top_k:
                break

        exec_time_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponsePayload(
            results=items,
            query_execution_time_ms=round(exec_time_ms, 2),
            candidate_count=candidate_count,
            metric_used=metric,
        )


@lru_cache
def get_visual_search_engine() -> VisualSearchEngine:
    """Return a cached singleton instance of VisualSearchEngine."""
    return VisualSearchEngine()
