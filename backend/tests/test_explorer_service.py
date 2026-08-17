"""Unit test suite for Embedding Explorer functionality."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionforge.explorer.analysis import apply_kmeans, compute_outlier_scores
from visionforge.explorer.reduction import InsufficientDataError, compute_projection
from visionforge.explorer.schemas import ProjectionMethod, ProjectionRequest
from visionforge.explorer.service import EmbeddingExplorerService, ExplorerCache
from visionforge.main import app
from visionforge.memory.index import VisualMemoryIndex, VisualMemoryRecord

client = TestClient(app)


def test_compute_projection_pca():
    """Verify PCA dimensionality reduction calculation and metadata."""
    # Synthetic (10, 768) matrix
    np.random.seed(42)
    matrix = np.random.randn(10, 768).astype(np.float32)

    coords, meta = compute_projection(
        matrix=matrix, method=ProjectionMethod.PCA, n_components=2, random_seed=42
    )

    assert coords.shape == (10, 2)
    assert meta.method == ProjectionMethod.PCA
    assert meta.n_components == 2
    assert meta.original_dimension == 768
    assert len(meta.explained_variance_ratio) == 2
    assert meta.cumulative_explained_variance > 0.0


def test_compute_projection_tsne():
    """Verify t-SNE non-linear projection calculation."""
    np.random.seed(42)
    matrix = np.random.randn(12, 768).astype(np.float32)

    coords, meta = compute_projection(
        matrix=matrix,
        method=ProjectionMethod.TSNE,
        n_components=2,
        perplexity=5.0,
        random_seed=42,
    )

    assert coords.shape == (12, 2)
    assert meta.method == ProjectionMethod.TSNE
    assert meta.perplexity == 5.0


def test_compute_projection_small_dataset():
    """Verify small dataset (N=2) generates safe fallback coordinates without error."""
    matrix = np.array([[1.0] * 768, [0.5] * 768], dtype=np.float32)
    coords, meta = compute_projection(matrix=matrix, method=ProjectionMethod.PCA, n_components=2)

    assert coords.shape == (2, 2)
    assert meta.n_components == 2


def test_compute_projection_insufficient_data():
    """Verify empty matrix raises InsufficientDataError."""
    empty_matrix = np.empty((0, 768), dtype=np.float32)
    with pytest.raises(InsufficientDataError):
        compute_projection(empty_matrix, method=ProjectionMethod.PCA)


def test_kmeans_and_outlier_scores():
    """Verify K-Means clustering and centroid outlier distance scoring."""
    coords = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [10.0, 10.0],
            [10.1, 9.9],
            [100.0, 100.0],  # Clear outlier
        ],
        dtype=np.float32,
    )

    labels, centroids, meta = apply_kmeans(coords, n_clusters=2, random_seed=42)
    assert len(labels) == 5
    assert meta.n_clusters == 2

    distances, outlier_scores = compute_outlier_scores(coords, labels, centroids)
    assert len(outlier_scores) == 5
    assert 0.0 <= np.min(outlier_scores) <= 1.0
    assert 0.0 <= np.max(outlier_scores) <= 1.0


def test_explorer_service_and_caching(tmp_path):
    """Verify EmbeddingExplorerService orchestration and projection caching."""
    mem = VisualMemoryIndex(storage_dir=str(tmp_path / "memory"))
    cache = ExplorerCache()
    svc = EmbeddingExplorerService(memory_index=mem, cache=cache)

    # Populate 5 synthetic records
    for i in range(5):
        vec = [float(i)] + [0.0] * 767
        rec = VisualMemoryRecord(id=f"rec_{i}", embedding=vec, tags=["synthetic"])
        mem.add_record(rec)

    req = ProjectionRequest(method=ProjectionMethod.PCA, n_components=2, n_clusters=2)

    # First call (cache miss)
    res1 = svc.generate_projection(req)
    assert res1.total_points == 5
    assert res1.cached is False

    # Second call (cache hit)
    res2 = svc.generate_projection(req)
    assert res2.total_points == 5
    assert res2.cached is True


def test_explorer_api_endpoints():
    """Verify REST API endpoints for explorer statistics and projection."""
    stats_res = client.get("/api/v1/explorer/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["success"] is True

    project_payload = {
        "method": "pca",
        "n_components": 2,
        "perplexity": 30.0,
        "random_seed": 42,
        "n_clusters": 2,
    }
    proj_res = client.post("/api/v1/explorer/project", json=project_payload)
    assert proj_res.status_code == 200
    assert proj_res.json()["success"] is True
