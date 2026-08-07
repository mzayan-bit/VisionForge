"""Unit test suite for similarity mathematics and vector calculation abstractions."""

import numpy as np
import pytest

from visionforge.search.similarity import (
    DimensionMismatchError,
    InvalidEmbeddingError,
    compute_cosine_similarity,
    compute_matrix_cosine_similarity,
    compute_matrix_euclidean_distance,
    validate_embedding_vector,
)


def test_validate_embedding_vector_valid():
    """Verify vector validation handles lists and 1D arrays."""
    vec_list = [0.1, 0.2, 0.3]
    arr = validate_embedding_vector(vec_list)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert arr.dtype == np.float32


def test_validate_embedding_vector_invalid():
    """Verify validation raises InvalidEmbeddingError on empty or NaN vectors."""
    with pytest.raises(InvalidEmbeddingError):
        validate_embedding_vector([])

    with pytest.raises(InvalidEmbeddingError):
        validate_embedding_vector(None)

    with pytest.raises(InvalidEmbeddingError):
        validate_embedding_vector([0.1, np.nan, 0.5])

    with pytest.raises(InvalidEmbeddingError):
        validate_embedding_vector([0.1, np.inf, 0.5])


def test_compute_cosine_similarity_identical():
    """Verify identical vectors yield cosine similarity of 1.0."""
    u = [0.5, 0.5, 0.5, 0.5]
    v = [0.5, 0.5, 0.5, 0.5]
    similarity = compute_cosine_similarity(u, v)
    assert abs(similarity - 1.0) < 1e-5


def test_compute_cosine_similarity_orthogonal():
    """Verify orthogonal vectors yield cosine similarity of 0.0."""
    u = [1.0, 0.0, 0.0]
    v = [0.0, 1.0, 0.0]
    similarity = compute_cosine_similarity(u, v)
    assert abs(similarity - 0.0) < 1e-5


def test_compute_cosine_similarity_opposite():
    """Verify opposite vectors yield cosine similarity of -1.0."""
    u = [1.0, 2.0, 3.0]
    v = [-1.0, -2.0, -3.0]
    similarity = compute_cosine_similarity(u, v)
    assert abs(similarity - (-1.0)) < 1e-5


def test_compute_cosine_similarity_zero_vector():
    """Verify zero vector is handled safely without division by zero."""
    u = [0.0, 0.0, 0.0]
    v = [1.0, 2.0, 3.0]
    similarity = compute_cosine_similarity(u, v)
    assert similarity == 0.0


def test_dimension_mismatch_error():
    """Verify dimension mismatch raises DimensionMismatchError."""
    u = [1.0, 2.0]
    v = [1.0, 2.0, 3.0]
    with pytest.raises(DimensionMismatchError):
        compute_cosine_similarity(u, v)

    matrix = np.ones((5, 768), dtype=np.float32)
    query_512 = [0.1] * 512
    with pytest.raises(DimensionMismatchError):
        compute_matrix_cosine_similarity(matrix, query_512)


def test_matrix_cosine_similarity():
    """Verify vectorized matrix similarity calculation."""
    # 3 candidate vectors of dimension 4
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = [1.0, 0.0, 0.0, 0.0]

    scores = compute_matrix_cosine_similarity(matrix, query)
    assert len(scores) == 3
    assert abs(scores[0] - 1.0) < 1e-4
    assert abs(scores[1] - 0.0) < 1e-4
    assert abs(scores[2] - 0.7071) < 1e-3


def test_matrix_euclidean_distance():
    """Verify vectorized matrix Euclidean distance calculation."""
    matrix = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = [1.0, 0.0]

    distances, scores = compute_matrix_euclidean_distance(matrix, query)
    assert distances[0] == 0.0
    assert scores[0] == 1.0
    assert distances[1] == 2.0
    assert abs(scores[1] - (1.0 / 3.0)) < 1e-4
