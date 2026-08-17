"""Unit and Integration Tests for VisionForge Active Learning & Intelligent Sample Selection System."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionforge.active_learning.schemas import (
    ReviewDecisionRequest,
    ReviewStatus,
    SelectionStrategy,
    SignalWeights,
)
from visionforge.active_learning.selector import (
    compute_novelty_score,
    compute_uncertainty_score,
    farthest_point_diversity_sampling,
    rank_candidate_samples,
)
from visionforge.active_learning.service import (
    ActiveLearningService,
    TestSetProtectionError,
)
from visionforge.main import app

client = TestClient(app)


# ─── Signal Calculators Tests ──────────────────────────────────────


def test_uncertainty_score_calculation():
    """Verify uncertainty proxy scoring formulation for object detection predictions."""
    # High confidence (0.95) -> Low uncertainty
    low_u = compute_uncertainty_score([{"confidence": 0.95}])
    assert low_u < 0.20

    # Low confidence (0.35) -> High uncertainty
    high_u = compute_uncertainty_score([{"confidence": 0.35}])
    assert high_u > 0.60

    # Competing predictions with small confidence margin
    competing_u = compute_uncertainty_score([{"confidence": 0.52}, {"confidence": 0.49}])
    assert competing_u > 0.60


def test_novelty_score_calculation():
    """Verify embedding space distance novelty calculation."""
    cand_emb = [1.0, 0.0, 0.0] + [0.0] * 765
    dataset_matrix = np.array([[0.0, 1.0, 0.0] + [0.0] * 765], dtype=np.float32)

    novelty = compute_novelty_score(cand_emb, dataset_matrix)
    assert 0.0 <= novelty <= 1.0
    assert novelty > 0.70  # Orthogonal vector should have high novelty


def test_farthest_point_diversity_sampling():
    """Verify Greedy k-Center Farthest-Point Sampling algorithm."""
    emb_1 = np.array([1.0, 0.0, 0.0] + [0.0] * 765, dtype=np.float32)
    emb_2 = np.array([0.0, 1.0, 0.0] + [0.0] * 765, dtype=np.float32)
    emb_3 = np.array([0.0, 0.0, 1.0] + [0.0] * 765, dtype=np.float32)
    embeddings = [emb_1, emb_2, emb_3]

    selected = farthest_point_diversity_sampling(embeddings, k=2)
    assert len(selected) == 2
    selected_indices = [idx for idx, _ in selected]
    assert len(set(selected_indices)) == 2


def test_rank_candidate_samples():
    """Verify multi-signal composite candidate ranking and plain-English reasons."""
    candidates = [
        {
            "image_id": "img_001",
            "image_path": "/tmp/img_001.jpg",
            "predictions": [{"confidence": 0.35}],
            "embedding": [0.1] * 768,
        },
        {
            "image_id": "img_002",
            "image_path": "/tmp/img_002.jpg",
            "predictions": [{"confidence": 0.95}],
            "embedding": [0.9] * 768,
        },
    ]

    ranked = rank_candidate_samples(
        candidate_data=candidates,
        dataset_matrix=None,
        strategy=SelectionStrategy.UNCERTAINTY,
        weights=SignalWeights(),
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0].rank == 1
    # Candidate 1 (low confidence) should rank higher under UNCERTAINTY strategy
    assert ranked[0].image_id == "img_001"
    assert ranked[0].recommendation_reason != ""


# ─── Test-Set Protection Tests ────────────────────────────────────


def test_test_set_protection_enforcement():
    """MANDATORY TEST: Verify active learning candidate pool strictly blocks test set evaluation samples."""
    service = ActiveLearningService()

    # 1. Mixed Pool: Test split samples must be stripped out automatically
    mixed_candidates = [
        "/data/train/sample_001.jpg",
        "/data/test/sample_002_test.jpg",
        "/data/val/sample_003.jpg",
    ]
    clean_pool, excluded_count = service.validate_candidate_pool("safety_v2", mixed_candidates)

    assert len(clean_pool) == 2
    assert excluded_count == 1
    assert "/data/test/sample_002_test.jpg" not in clean_pool

    # 2. Pure Test Pool: Must raise TestSetProtectionError
    pure_test_candidates = [
        "/data/test/eval_01.jpg",
        "/data/test/eval_02.jpg",
    ]
    with pytest.raises(TestSetProtectionError) as exc_info:
        service.validate_candidate_pool("safety_v2", pure_test_candidates)

    assert "Test set samples cannot be used for active learning" in str(exc_info.value)


# ─── Service Lifecycle & Review Queue Tests ─────────────────────────


def test_active_learning_service_lifecycle():
    """Test active learning run execution, review queue decisions, and bias analysis."""
    service = ActiveLearningService()

    # 1. Create Run
    run = service.create_run(
        dataset_id="safety_v2",
        model_id="yolo11s.pt",
        strategy=SelectionStrategy.UNCERTAINTY_DIVERSITY,
        top_k=10,
    )

    assert run.run_id.startswith("al_run_")
    assert len(run.selected_samples) == 10
    assert run.strategy == SelectionStrategy.UNCERTAINTY_DIVERSITY

    # 2. Submit Review Decision
    sample_to_review = run.selected_samples[0]
    updated_run = service.submit_review_decision(
        ReviewDecisionRequest(
            run_id=run.run_id,
            image_id=sample_to_review.image_id,
            status=ReviewStatus.ACCEPTED,
            notes="Accepted for safety helmet re-labeling batch",
        )
    )

    assert updated_run.selected_samples[0].review_status == ReviewStatus.ACCEPTED
    assert updated_run.selected_samples[0].notes == "Accepted for safety helmet re-labeling batch"

    # 3. Selection Bias Analysis
    bias_report = service.analyze_selection_bias(run.run_id)
    assert bias_report.run_id == run.run_id
    assert bias_report.total_selected == 10
    assert "median" in bias_report.confidence_distribution

    # 4. Strategy Comparison
    cmp_res = service.compare_strategies(
        dataset_id="safety_v2",
        model_id="yolo11s.pt",
        strategy_a=SelectionStrategy.UNCERTAINTY,
        strategy_b=SelectionStrategy.UNCERTAINTY_DIVERSITY,
        top_k=10,
    )
    assert cmp_res.dataset_id == "safety_v2"
    assert cmp_res.summary_notes != ""


# ─── REST API Endpoint Tests ───────────────────────────────────────


def test_api_create_run_and_review():
    """Test POST /api/v1/active-learning/runs and POST /api/v1/active-learning/review."""
    res = client.post(
        "/api/v1/active-learning/runs",
        json={
            "dataset_id": "safety_v2",
            "model_id": "yolo11s.pt",
            "strategy": "UNCERTAINTY_DIVERSITY",
            "top_k": 5,
        },
    )
    assert res.status_code == 201
    data = res.json()
    run_id = data["run_id"]
    assert len(data["selected_samples"]) == 5

    # Review Decision
    sample_id = data["selected_samples"][0]["image_id"]
    rev_res = client.post(
        "/api/v1/active-learning/review",
        json={
            "run_id": run_id,
            "image_id": sample_id,
            "status": "MARKED_FOR_LABELING",
            "notes": "API test review decision",
        },
    )
    assert rev_res.status_code == 200
    updated = rev_res.json()
    assert updated["selected_samples"][0]["review_status"] == "MARKED_FOR_LABELING"


def test_api_list_runs_and_compare():
    """Test GET /api/v1/active-learning/runs and POST /api/v1/active-learning/compare."""
    # List runs
    list_res = client.get("/api/v1/active-learning/runs")
    assert list_res.status_code == 200
    assert isinstance(list_res.json(), list)

    # Compare strategies
    cmp_res = client.post(
        "/api/v1/active-learning/compare",
        json={
            "dataset_id": "safety_v2",
            "model_id": "yolo11s.pt",
            "candidate_pool_id": "pool_test",
            "strategy_a": "UNCERTAINTY",
            "strategy_b": "DIVERSITY",
            "top_k": 5,
        },
    )
    assert cmp_res.status_code == 200
    assert "summary_notes" in cmp_res.json()
