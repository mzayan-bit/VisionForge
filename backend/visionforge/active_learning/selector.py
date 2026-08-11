"""VisionForge Active Learning Signal Calculators & Ranking Engine."""

import logging
import math
from typing import Any

import numpy as np

from visionforge.active_learning.schemas import (
    RankedSample,
    SampleSignals,
    SelectionStrategy,
    SignalWeights,
)

logger = logging.getLogger("visionforge.active_learning.selector")


def compute_uncertainty_score(predictions: list[dict[str, Any]]) -> float:
    """Calculate model prediction uncertainty proxy score [0.0, 1.0].

    Uncertainty Proxy Formulation:
    - If predictions exist: U = 1.0 - max(confidence). High score indicates prediction ambiguity.
    - Additional penalty for multiple competing predictions with similar confidence scores.
    """
    if not predictions:
        return 0.50  # Baseline moderate uncertainty for un-detected candidates

    confidences = [float(p.get("confidence", 0.5)) for p in predictions]
    max_conf = max(confidences)

    # Primary confidence margin uncertainty
    margin_uncertainty = max(0.0, min(1.0, 1.0 - max_conf))

    # Competing predictions bonus
    competing_bonus = 0.0
    if len(confidences) > 1:
        sorted_confs = sorted(confidences, reverse=True)
        margin = sorted_confs[0] - sorted_confs[1]
        if margin < 0.15:
            competing_bonus = 0.20 * (1.0 - (margin / 0.15))

    final_score = round(min(1.0, margin_uncertainty + competing_bonus), 4)
    return final_score


def compute_novelty_score(
    candidate_embedding: list[float] | np.ndarray,
    dataset_matrix: np.ndarray | None,
) -> float:
    """Calculate embedding novelty score [0.0, 1.0] based on k-NN distance to existing dataset vectors."""
    if dataset_matrix is None or len(dataset_matrix) == 0:
        return 0.90  # Default high novelty if dataset has no indexed vectors yet

    cand_vec = np.array(candidate_embedding, dtype=np.float32)
    cand_norm = cand_vec / (np.linalg.norm(cand_vec) + 1e-9)

    # Compute cosine distances to dataset matrix
    ds_norms = dataset_matrix / (np.linalg.norm(dataset_matrix, axis=1, keepdims=True) + 1e-9)
    similarities = np.dot(ds_norms, cand_norm)
    min_dist = max(0.0, float(1.0 - np.max(similarities)))

    # Sigmoid normalization to [0.0, 1.0] range
    novelty = 1.0 / (1.0 + math.exp(-6.0 * (min_dist - 0.4)))
    return round(float(novelty), 4)


def farthest_point_diversity_sampling(
    candidate_embeddings: list[np.ndarray],
    k: int,
) -> list[tuple[int, float]]:
    """Greedy k-Center / Farthest-Point Sampling algorithm maximizing pairwise visual coverage distance.

    Returns list of tuples (candidate_index, diversity_score).
    """
    n = len(candidate_embeddings)
    if n == 0:
        return []

    if n <= k:
        return [(i, 1.0) for i in range(n)]

    matrix = np.array(candidate_embeddings, dtype=np.float32)
    norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)

    selected_indices: list[int] = [0]
    min_distances = np.full(n, fill_value=np.inf, dtype=np.float32)

    selected_scores: list[tuple[int, float]] = [(0, 1.0)]

    for _ in range(1, k):
        last_selected = norms[selected_indices[-1]]
        # Calculate cosine distance to last selected point
        dists = 1.0 - np.dot(norms, last_selected)
        min_distances = np.minimum(min_distances, dists)

        # Pick farthest candidate
        next_idx = int(np.argmax(min_distances))
        max_dist = float(min_distances[next_idx])
        selected_indices.append(next_idx)

        diversity_score = round(float(min(1.0, max_dist / 1.5)), 4)
        selected_scores.append((next_idx, diversity_score))

    return selected_scores


def generate_recommendation_reason(
    signals: SampleSignals, strategy: SelectionStrategy
) -> str:
    """Generate human-readable rationale explaining why a sample was recommended."""
    reasons: list[str] = []

    if signals.uncertainty_score > 0.65:
        reasons.append(f"high prediction uncertainty ({signals.uncertainty_score:.2f})")
    elif signals.uncertainty_score > 0.45:
        reasons.append(f"moderate prediction ambiguity ({signals.uncertainty_score:.2f})")

    if signals.novelty_score > 0.65:
        reasons.append(f"visually novel relative to dataset (novelty {signals.novelty_score:.2f})")

    if signals.diversity_score > 0.65:
        reasons.append(f"high visual coverage distance (diversity {signals.diversity_score:.2f})")

    if signals.failure_score > 0.50:
        reasons.append(f"resemble past evaluation failure patterns ({signals.failure_score:.2f})")

    if not reasons:
        reasons.append(f"selected under {strategy} strategy (composite score {signals.composite_score:.2f})")

    capitalized = reasons[0].capitalize()
    if len(reasons) > 1:
        return f"{capitalized} and {reasons[1]}."
    return f"{capitalized}."


def rank_candidate_samples(
    candidate_data: list[dict[str, Any]],
    dataset_matrix: np.ndarray | None,
    strategy: SelectionStrategy,
    weights: SignalWeights,
    top_k: int = 25,
) -> list[RankedSample]:
    """Calculate multi-signal scores and produce weighted ranked sample recommendations."""
    if not candidate_data:
        return []

    # 1. Compute Individual Signals for all candidates
    sample_signals_list: list[SampleSignals] = []
    embeddings_list: list[np.ndarray] = []

    for item in candidate_data:
        img_id = item.get("image_id", "img_unknown")
        img_path = item.get("image_path", "")
        preds = item.get("predictions", [])
        emb = item.get("embedding", [0.0] * 768)

        emb_arr = np.array(emb, dtype=np.float32)
        embeddings_list.append(emb_arr)

        u_score = compute_uncertainty_score(preds)
        n_score = compute_novelty_score(emb_arr, dataset_matrix)
        f_score = float(item.get("failure_score", 0.0))
        q_score = float(item.get("quality_score", 0.5))

        sample_signals_list.append(
            SampleSignals(
                image_id=img_id,
                image_path=img_path,
                uncertainty_score=u_score,
                novelty_score=n_score,
                diversity_score=0.5,  # Placeholder updated in step 2
                failure_score=f_score,
                quality_score=q_score,
            )
        )

    # 2. Diversity Selection via Farthest-Point Sampling
    diversity_results = farthest_point_diversity_sampling(
        embeddings_list, k=min(top_k * 2, len(candidate_data))
    )
    div_map = {idx: score for idx, score in diversity_results}

    for idx, sig in enumerate(sample_signals_list):
        sig.diversity_score = div_map.get(idx, 0.20)

    # 3. Apply Weights based on Selection Strategy
    w_u, w_n, w_d, w_f, w_q = (
        weights.uncertainty,
        weights.novelty,
        weights.diversity,
        weights.failure,
        weights.quality,
    )

    if strategy == SelectionStrategy.UNCERTAINTY:
        w_u, w_n, w_d, w_f, w_q = 0.70, 0.10, 0.10, 0.10, 0.00
    elif strategy == SelectionStrategy.DIVERSITY:
        w_u, w_n, w_d, w_f, w_q = 0.10, 0.20, 0.60, 0.10, 0.00
    elif strategy == SelectionStrategy.NOVELTY:
        w_u, w_n, w_d, w_f, w_q = 0.10, 0.70, 0.10, 0.10, 0.00

    weight_sum = w_u + w_n + w_d + w_f + w_q
    if weight_sum > 0:
        w_u, w_n, w_d, w_f, w_q = (
            w_u / weight_sum,
            w_n / weight_sum,
            w_d / weight_sum,
            w_f / weight_sum,
            w_q / weight_sum,
        )

    # Calculate Composite Score
    for sig in sample_signals_list:
        comp = (
            w_u * sig.uncertainty_score
            + w_n * sig.novelty_score
            + w_d * sig.diversity_score
            + w_f * sig.failure_score
            + w_q * sig.quality_score
        )
        sig.composite_score = round(float(comp), 4)

    # 4. Sort Candidates Descending by Composite Score
    sorted_pairs = sorted(
        enumerate(sample_signals_list), key=lambda x: x[1].composite_score, reverse=True
    )

    # 5. Build Top-K Ranked Samples
    ranked_samples: list[RankedSample] = []
    for rank_idx, (orig_idx, signals) in enumerate(sorted_pairs[:top_k], start=1):
        reason = generate_recommendation_reason(signals, strategy)
        ranked_samples.append(
            RankedSample(
                rank=rank_idx,
                image_id=signals.image_id,
                image_path=signals.image_path,
                composite_score=signals.composite_score,
                signals=signals,
                recommendation_reason=reason,
            )
        )

    return ranked_samples
