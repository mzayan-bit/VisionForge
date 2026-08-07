"""VisionForge Similarity Mathematics Abstraction.

Provides model-independent, dimension-agnostic, and numerically safe vector distance
and similarity calculation functions.
"""

import logging
from enum import StrEnum

import numpy as np

from visionforge.core.exceptions import VisionForgeException

logger = logging.getLogger("visionforge.search.similarity")

EPSILON = 1e-12


class SimilarityMetric(StrEnum):
    """Vector distance and similarity measurement metric classification."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class DimensionMismatchError(VisionForgeException):
    """Raised when vector dimensions do not match the index matrix dimension."""

    def __init__(self, query_dim: int, index_dim: int):
        msg = f"Query dimension ({query_dim}D) does not match index dimension ({index_dim}D)"
        super().__init__(
            message=msg,
            code="DIMENSION_MISMATCH",
            status_code=400,
        )


class InvalidEmbeddingError(VisionForgeException):
    """Raised when an embedding vector is empty, corrupted, or contains NaN/Inf values."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid embedding vector: {reason}",
            code="INVALID_EMBEDDING",
            status_code=400,
        )


def validate_embedding_vector(vector: list[float] | np.ndarray) -> np.ndarray:
    """Validate and convert embedding vector to 1D float32 NumPy array.

    Raises:
        InvalidEmbeddingError: If vector is empty, invalid shape, or contains NaN/Inf.
    """
    if vector is None:
        raise InvalidEmbeddingError("Embedding vector cannot be None")

    arr = np.asarray(vector, dtype=np.float32)

    if arr.size == 0:
        raise InvalidEmbeddingError("Embedding vector cannot be empty")

    if arr.ndim != 1:
        arr = arr.flatten()

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise InvalidEmbeddingError("Embedding vector contains NaN or Inf values")

    return arr


def compute_cosine_similarity(u: list[float] | np.ndarray, v: list[float] | np.ndarray) -> float:
    """Compute numerically safe cosine similarity between two vectors.

    Formula: S(u, v) = (u . v) / (||u||_2 * ||v||_2 + EPSILON)
    """
    arr_u = validate_embedding_vector(u)
    arr_v = validate_embedding_vector(v)

    if arr_u.shape != arr_v.shape:
        raise DimensionMismatchError(arr_u.shape[0], arr_v.shape[0])

    norm_u = float(np.linalg.norm(arr_u))
    norm_v = float(np.linalg.norm(arr_v))

    if norm_u < EPSILON or norm_v < EPSILON:
        return 0.0

    dot_product = float(np.dot(arr_u, arr_v))
    similarity = dot_product / ((norm_u * norm_v) + EPSILON)

    # Clip to valid cosine range [-1.0, 1.0] to eliminate floating-point precision error
    return float(np.clip(similarity, -1.0, 1.0))


def compute_matrix_cosine_similarity(
    matrix: np.ndarray, query_vector: list[float] | np.ndarray
) -> np.ndarray:
    """Compute vectorized cosine similarity between matrix (N, D) and query vector (D,).

    Returns 1D array of similarity scores for each candidate in range [-1.0, 1.0].
    """
    q_arr = validate_embedding_vector(query_vector)

    if matrix.size == 0:
        return np.empty((0,), dtype=np.float32)

    if matrix.ndim != 2:
        raise InvalidEmbeddingError(f"Candidate matrix must be 2D array, got {matrix.ndim}D")

    if matrix.shape[1] != q_arr.shape[0]:
        raise DimensionMismatchError(q_arr.shape[0], matrix.shape[1])

    # Normalize query vector
    q_norm = float(np.linalg.norm(q_arr))
    if q_norm > EPSILON:
        q_normalized = q_arr / q_norm
    else:
        q_normalized = q_arr

    # Compute norms of candidate rows
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms < EPSILON, 1.0, row_norms)

    # Matrix-vector dot product
    dots = np.dot(matrix, q_normalized)
    scores = dots / row_norms

    return np.clip(scores, -1.0, 1.0)


def compute_matrix_euclidean_distance(
    matrix: np.ndarray, query_vector: list[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Euclidean distances and mapped similarity scores for candidate matrix.

    Returns tuple (distances, similarity_scores).
    """
    q_arr = validate_embedding_vector(query_vector)

    if matrix.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    if matrix.shape[1] != q_arr.shape[0]:
        raise DimensionMismatchError(q_arr.shape[0], matrix.shape[1])

    diffs = matrix - q_arr
    distances = np.linalg.norm(diffs, axis=1)

    # Map distance d to similarity score in range [0.0, 1.0]: 1.0 / (1.0 + d)
    scores = 1.0 / (1.0 + distances)
    return distances, scores
