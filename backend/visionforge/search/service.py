"""VisionForge Visual Search Service — Orchestration & Business Logic Layer.

Decouples visual search orchestration from FastAPI REST controllers.
Handles input validation, search execution, history recording, and result formatting.
"""

import logging
from functools import lru_cache

from visionforge.core.exceptions import VisionForgeException
from visionforge.memory.index import VisualMemoryIndex, get_visual_memory_index
from visionforge.search.engine import (
    SearchResponsePayload,
    SimilarityMetric,
    VisualSearchEngine,
    get_visual_search_engine,
)
from visionforge.search.history import (
    SearchHistoryRecord,
    SearchHistoryStore,
    get_search_history_store,
)

logger = logging.getLogger("visionforge.search.service")


class VisualSearchServiceException(VisionForgeException):
    """Base exception for VisualSearchService errors."""

    def __init__(self, message: str, code: str = "SEARCH_SERVICE_ERROR", status_code: int = 400):
        super().__init__(message=message, code=code, status_code=status_code)


class VisualSearchService:
    """Service layer orchestrating image, record, and vector similarity searches."""

    def __init__(
        self,
        search_engine: VisualSearchEngine | None = None,
        memory_index: VisualMemoryIndex | None = None,
        history_store: SearchHistoryStore | None = None,
    ):
        self._search_engine = search_engine or get_visual_search_engine()
        self._memory_index = memory_index or get_visual_memory_index()
        self._history_store = history_store or get_search_history_store()

    async def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute visual search given query image file upload."""
        if not image_bytes:
            raise VisualSearchServiceException("Uploaded query image file is empty")

        response = await self._search_engine.search_by_image(
            image_bytes=image_bytes,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
        )

        self._record_history(response, query_type="image")
        return response

    def search_by_record(
        self,
        record_id: str,
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute visual search using an existing Visual Memory record ID as query."""
        record = self._memory_index.get_record(record_id)

        meta = record.image_metadata
        query_info = {
            "type": "memory_record",
            "query_record_id": record_id,
            "resolution": f"{meta.get('width', 0)}x{meta.get('height', 0)}",
            "tags": record.tags,
        }

        response = self._search_engine.search_by_vector(
            query_vector=record.embedding,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
            query_info=query_info,
            embedding_time_ms=0.0,
            model_used="siglip-base-patch16-224",
        )

        self._record_history(response, query_type="memory_record")
        return response

    def search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute visual search directly using a dense query vector."""
        query_info = {"type": "vector", "dimension": len(query_vector)}

        response = self._search_engine.search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
            query_info=query_info,
            embedding_time_ms=0.0,
            model_used="siglip-base-patch16-224",
        )

        self._record_history(response, query_type="vector")
        return response

    def get_search_history(
        self, limit: int = 50, offset: int = 0
    ) -> list[SearchHistoryRecord]:
        """Return paginated search history records."""
        return self._history_store.get_history(limit=limit, offset=offset)

    def _record_history(self, payload: SearchResponsePayload, query_type: str) -> None:
        """Record search transaction into SearchHistoryStore."""
        rec = SearchHistoryRecord(
            search_id=payload.search_id,
            timestamp=payload.timestamp,
            query_type=query_type,
            query_info=payload.query_info,
            model_used=payload.model_used,
            top_k=len(payload.results),
            threshold=0.0,
            metric_used=payload.metric_used.value,
            candidate_count=payload.candidate_count,
            returned_count=payload.returned_count,
            embedding_time_ms=payload.embedding_time_ms,
            search_time_ms=payload.search_time_ms,
            total_time_ms=payload.total_execution_time_ms,
        )
        self._history_store.record_search(rec)


@lru_cache
def get_visual_search_service() -> VisualSearchService:
    """Return a cached singleton instance of VisualSearchService."""
    return VisualSearchService()
