"""Dimensionality Reduction Service — PCA & t-SNE Implementation."""

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from visionforge.core.exceptions import VisionForgeException
from visionforge.explorer.schemas import DimensionalityReductionMeta, ProjectionMethod

logger = logging.getLogger("visionforge.explorer.reduction")


class InsufficientDataError(VisionForgeException):
    """Raised when the embedding dataset size is insufficient for projection."""

    def __init__(self, count: int):
        super().__init__(
            message=f"Cannot perform projection on {count} items. Minimum 1 record required.",
            code="INSUFFICIENT_DATA",
            status_code=400,
        )


def compute_projection(
    matrix: np.ndarray,
    method: ProjectionMethod = ProjectionMethod.PCA,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_seed: int = 42,
) -> tuple[np.ndarray, DimensionalityReductionMeta]:
    """Execute PCA or t-SNE projection over an (N, D) float32 matrix.

    Returns tuple of (N, n_components) projected coordinates and reduction metadata.
    """
    n_samples, orig_dim = matrix.shape

    if n_samples == 0:
        raise InsufficientDataError(0)

    # Handle small N gracefully
    if n_samples <= n_components:
        # Generate simple deterministic spread for 1 or 2 points
        coords = np.zeros((n_samples, n_components), dtype=np.float32)
        for i in range(n_samples):
            coords[i, 0] = float(i * 2.0 - (n_samples - 1))
        meta = DimensionalityReductionMeta(
            method=method,
            n_components=n_components,
            original_dimension=orig_dim,
            explained_variance_ratio=[1.0] + [0.0] * (n_components - 1),
            cumulative_explained_variance=1.0,
            perplexity=perplexity if method == ProjectionMethod.TSNE else None,
            random_seed=random_seed,
        )
        return coords, meta

    if method == ProjectionMethod.PCA:
        actual_components = min(n_components, n_samples)
        pca = PCA(n_components=actual_components, random_state=random_seed)
        coords = pca.fit_transform(matrix)

        # Pad component dimensions if actual_components < n_components
        if coords.shape[1] < n_components:
            pad_cols = n_components - coords.shape[1]
            coords = np.pad(coords, ((0, 0), (0, pad_cols)), mode="constant")

        explained_ratios = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        cum_variance = round(float(np.sum(pca.explained_variance_ratio_)), 4)

        meta = DimensionalityReductionMeta(
            method=ProjectionMethod.PCA,
            n_components=n_components,
            original_dimension=orig_dim,
            explained_variance_ratio=explained_ratios,
            cumulative_explained_variance=cum_variance,
            random_seed=random_seed,
        )
        return coords, meta

    elif method == ProjectionMethod.TSNE:
        # Cap perplexity to valid range [2.0, max(2.0, n_samples - 1)]
        safe_perplexity = min(perplexity, max(2.0, float(n_samples - 1)))
        tsne = TSNE(
            n_components=n_components,
            perplexity=safe_perplexity,
            random_state=random_seed,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(matrix)

        meta = DimensionalityReductionMeta(
            method=ProjectionMethod.TSNE,
            n_components=n_components,
            original_dimension=orig_dim,
            perplexity=round(safe_perplexity, 2),
            random_seed=random_seed,
        )
        return coords, meta

    else:
        raise ValueError(f"Unsupported projection method '{method}'")
