"""Clustering and Outlier Detection Analysis Engine."""

import logging

import numpy as np
from sklearn.cluster import KMeans

from visionforge.explorer.schemas import ClusteringMeta

logger = logging.getLogger("visionforge.explorer.analysis")


def apply_kmeans(
    matrix: np.ndarray, n_clusters: int = 3, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, ClusteringMeta]:
    """Execute K-Means clustering over dataset matrix.

    Returns tuple (cluster_labels, centroids, clustering_meta).
    """
    n_samples = matrix.shape[0]
    if n_samples == 0:
        meta = ClusteringMeta(n_clusters=0, cluster_sizes={}, inertia=0.0)
        empty_lbl = np.empty((0,), dtype=np.int32)
        empty_cen = np.empty((0, matrix.shape[1]), dtype=np.float32)
        return empty_lbl, empty_cen, meta

    # Cap n_clusters to number of available samples
    actual_k = max(1, min(n_clusters, n_samples))

    kmeans = KMeans(n_clusters=actual_k, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(matrix)
    centroids = kmeans.cluster_centers_

    # Calculate cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts, strict=False)}

    meta = ClusteringMeta(
        method="kmeans",
        n_clusters=actual_k,
        cluster_sizes=cluster_sizes,
        inertia=round(float(kmeans.inertia_), 4),
    )

    return labels, centroids, meta


def compute_outlier_scores(
    matrix: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Euclidean distance to assigned cluster centroid and normalized outlier scores.

    Formula:
      d_i = ||x_i - centroid_{c_i}||_2
      outlier_score_i = (d_i - min_d) / (max_d - min_d + 1e-8)

    Returns tuple (distances_to_centroid, normalized_outlier_scores).
    """
    n_samples = matrix.shape[0]
    if n_samples == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    distances = np.zeros((n_samples,), dtype=np.float32)
    for i in range(n_samples):
        cluster_idx = labels[i]
        centroid = centroids[cluster_idx]
        distances[i] = float(np.linalg.norm(matrix[i] - centroid))

    min_d = float(np.min(distances))
    max_d = float(np.max(distances))
    spread = max_d - min_d

    if spread > 1e-8:
        outlier_scores = (distances - min_d) / spread
    else:
        outlier_scores = np.zeros_like(distances)

    return np.round(distances, 4), np.round(outlier_scores, 4)
