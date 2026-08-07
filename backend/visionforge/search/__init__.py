"""VisionForge Visual Search Engine Package."""

from visionforge.search.engine import (
    SearchResponsePayload,
    SearchResultItem,
    SimilarityMetric,
    VisualSearchEngine,
    get_visual_search_engine,
)
from visionforge.search.history import (
    SearchHistoryRecord,
    SearchHistoryStore,
    get_search_history_store,
)
from visionforge.search.service import (
    VisualSearchService,
    get_visual_search_service,
)
from visionforge.search.similarity import (
    DimensionMismatchError,
    InvalidEmbeddingError,
    compute_cosine_similarity,
    compute_matrix_cosine_similarity,
    compute_matrix_euclidean_distance,
)

__all__ = [
    "SimilarityMetric",
    "SearchResultItem",
    "SearchResponsePayload",
    "VisualSearchEngine",
    "get_visual_search_engine",
    "VisualSearchService",
    "get_visual_search_service",
    "SearchHistoryRecord",
    "SearchHistoryStore",
    "get_search_history_store",
    "compute_cosine_similarity",
    "compute_matrix_cosine_similarity",
    "compute_matrix_euclidean_distance",
    "DimensionMismatchError",
    "InvalidEmbeddingError",
]
