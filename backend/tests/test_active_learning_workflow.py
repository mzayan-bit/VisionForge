"""Unit and Integration Tests for VisionForge Active Learning & Human-in-the-Loop Workflow."""

import numpy as np
from fastapi.testclient import TestClient

from visionforge.active_learning.schemas import (
    ReviewDecisionType,
    ReviewerAgreementStatus,
    ReviewStatus,
    SelectionStrategy,
    SignalWeights,
)
from visionforge.active_learning.selector import (
    compute_uncertainty_score,
    farthest_point_diversity_sampling,
    rank_candidate_samples,
)
from visionforge.active_learning.service import get_active_learning_service
from visionforge.main import app

client = TestClient(app)


def test_uncertainty_sampling_ranking():
    """Verify uncertainty calculation prioritizes low confidence and ambiguous predictions."""
    # 1. High confidence prediction (0.95) -> Low uncertainty
    high_conf_preds = [{"confidence": 0.95, "class_name": "helmet"}]
    u_low = compute_uncertainty_score(high_conf_preds)
    assert u_low < 0.10

    # 2. Low confidence prediction (0.35) -> High uncertainty
    low_conf_preds = [{"confidence": 0.35, "class_name": "helmet"}]
    u_high = compute_uncertainty_score(low_conf_preds)
    assert u_high > 0.60
    assert u_high > u_low

    # 3. Competing close predictions (0.52 vs 0.48) -> Extra margin penalty
    competing_preds = [
        {"confidence": 0.52, "class_name": "helmet"},
        {"confidence": 0.48, "class_name": "head"},
    ]
    u_comp = compute_uncertainty_score(competing_preds)
    assert u_comp > 0.55


def test_diversity_sampling_farthest_point():
    """Verify farthest-point sampling spreads candidate selection across embedding space."""
    # 4 embeddings: 2 near-identical vectors at [1, 0] and 2 distinct vectors at [0, 1] and [-1, 0]
    emb1 = np.array([1.0, 0.0] + [0.0] * 766, dtype=np.float32)
    emb2 = np.array([0.99, 0.01] + [0.0] * 766, dtype=np.float32)  # near duplicate of emb1
    emb3 = np.array([0.0, 1.0] + [0.0] * 766, dtype=np.float32)
    emb4 = np.array([-1.0, 0.0] + [0.0] * 766, dtype=np.float32)

    embeddings = [emb1, emb2, emb3, emb4]
    selected = farthest_point_diversity_sampling(embeddings, k=3)

    assert len(selected) == 3
    selected_indices = [idx for idx, _ in selected]
    # Should pick emb1 (0), emb3 (2), and emb4 (3), skipping duplicate emb2 (1)
    assert 0 in selected_indices
    assert 2 in selected_indices or 3 in selected_indices


def test_hybrid_sampling_and_exact_budget():
    """Verify hybrid multi-signal ranking strictly respects requested sample budget."""
    candidates = [
        {
            "image_id": f"cand_{i:03d}",
            "image_path": f"/test/cand_{i:03d}.jpg",
            "predictions": [{"confidence": 0.40 if i % 2 == 0 else 0.90, "class_name": "person"}],
            "embedding": [0.1 * (i % 5)] * 768,
            "failure_score": 0.80 if i % 3 == 0 else 0.10,
        }
        for i in range(25)
    ]

    # Test budget = 10
    ranked_10 = rank_candidate_samples(
        candidate_data=candidates,
        dataset_matrix=None,
        strategy=SelectionStrategy.HYBRID,
        weights=SignalWeights(),
        top_k=10,
    )
    assert len(ranked_10) == 10
    assert ranked_10[0].rank == 1
    assert ranked_10[9].rank == 10
    assert ranked_10[0].composite_score >= ranked_10[9].composite_score

    # Test budget = 5
    ranked_5 = rank_candidate_samples(
        candidate_data=candidates,
        dataset_matrix=None,
        strategy=SelectionStrategy.UNCERTAINTY,
        weights=SignalWeights(),
        top_k=5,
    )
    assert len(ranked_5) == 5


def test_candidate_explanation_generator():
    """Verify evidence-based candidate explanations detail exact selection reasons."""
    candidates = [
        {
            "image_id": "cand_rare_01",
            "image_path": "/test/rare.jpg",
            "predictions": [{"confidence": 0.38, "class_name": "gloves"}],
            "embedding": [0.5] * 768,
            "is_rare_class": True,
            "failure_score": 0.75,
        }
    ]

    ranked = rank_candidate_samples(
        candidate_data=candidates,
        dataset_matrix=None,
        strategy=SelectionStrategy.HYBRID,
        weights=SignalWeights(),
        top_k=1,
    )
    sample = ranked[0]
    assert sample.explanation.composite_priority > 0.0
    assert sample.explanation.class_rarity_flag is True
    assert len(sample.explanation.plain_text_reasons) >= 1
    assert any(
        "uncertainty" in r.lower() or "rare" in r.lower()
        for r in sample.explanation.plain_text_reasons
    )


def test_review_decision_submission_and_state_transitions():
    """Verify human review decisions properly update candidate state and queue counts."""
    svc = get_active_learning_service()
    cycle = svc.create_cycle(
        name="Test Review Cycle",
        dataset_id="safety_v2",
        budget=10,
        strategy=SelectionStrategy.HYBRID,
    )

    first_sample_id = cycle.selected_samples[0].image_id

    # 1. Confirm Sample
    dec = svc.record_review_decision(
        cycle_id=cycle.cycle_id,
        sample_id=first_sample_id,
        decision=ReviewDecisionType.CONFIRMED,
        reviewer_id="Reviewer Alice",
        notes="High quality detection",
    )
    assert dec.decision == ReviewDecisionType.CONFIRMED

    # Check updated cycle state
    updated_cycle = svc.get_cycle(cycle.cycle_id)
    target_sample = next(s for s in updated_cycle.selected_samples if s.image_id == first_sample_id)
    assert target_sample.review_status == ReviewStatus.ACCEPTED
    assert target_sample.review_decision == ReviewDecisionType.CONFIRMED
    assert updated_cycle.review_counts["reviewed"] >= 1


def test_reviewer_consensus_and_disagreement():
    """Verify multi-reviewer agreement tracking and conflict detection."""
    svc = get_active_learning_service()
    cycle = svc.create_cycle(name="Consensus Test Cycle", budget=5)
    sample_id = f"cand_unique_{cycle.cycle_id}"

    # Reviewer 1: Confirmed
    svc.record_review_decision(
        cycle_id=cycle.cycle_id,
        sample_id=sample_id,
        decision=ReviewDecisionType.CONFIRMED,
        reviewer_id="Alice",
    )
    consensus_1 = svc.get_sample_consensus(sample_id, cycle_id=cycle.cycle_id)
    assert consensus_1.consensus_status == ReviewerAgreementStatus.UNANIMOUS

    # Reviewer 2: Incorrect Prediction (Conflict!)
    svc.record_review_decision(
        cycle_id=cycle.cycle_id,
        sample_id=sample_id,
        decision=ReviewDecisionType.INCORRECT_PREDICTION,
        reviewer_id="Bob",
    )
    consensus_2 = svc.get_sample_consensus(sample_id, cycle_id=cycle.cycle_id)
    assert consensus_2.consensus_status == ReviewerAgreementStatus.NEEDS_RESOLUTION


def test_commit_cycle_dataset_version():
    """Verify explicit user confirmation commits a new dataset version record."""
    svc = get_active_learning_service()
    cycle = svc.create_cycle(name="Commit Version Test Cycle", budget=5)

    # Approve 2 samples
    svc.record_review_decision(
        cycle_id=cycle.cycle_id,
        sample_id=cycle.selected_samples[0].image_id,
        decision=ReviewDecisionType.CONFIRMED,
    )
    svc.record_review_decision(
        cycle_id=cycle.cycle_id,
        sample_id=cycle.selected_samples[1].image_id,
        decision=ReviewDecisionType.CONFIRMED,
    )

    committed_cycle = svc.commit_cycle_dataset_version(
        cycle_id=cycle.cycle_id,
        new_version_tag="v2.1.0-al-test",
    )

    assert committed_cycle.status == "COMPLETED"
    assert committed_cycle.resulting_dataset_version == "v2.1.0-al-test"
    assert committed_cycle.benchmark_after_map50 is not None


def test_active_learning_cycle_history():
    """Verify longitudinal active learning progression tracking."""
    svc = get_active_learning_service()
    hist = svc.get_cycle_history(dataset_id="safety_v2")
    assert len(hist) >= 2
    assert hist[0].delta_map50 is not None
    assert hist[0].samples_reviewed == 50


def test_active_learning_api_endpoints():
    """Verify REST API endpoints for Active Learning cycles and review workflow."""
    # 1. Create Cycle
    res_create = client.post(
        "/api/v1/active-learning/cycles",
        json={
            "name": "API Test Cycle",
            "dataset_id": "safety_v2",
            "budget": 10,
            "strategy": "HYBRID",
        },
    )
    assert res_create.status_code == 201
    cycle_data = res_create.json()["data"]
    cid = cycle_data["cycle_id"]
    assert len(cycle_data["selected_samples"]) == 10

    # 2. Get Cycle
    res_get = client.get(f"/api/v1/active-learning/cycles/{cid}")
    assert res_get.status_code == 200
    assert res_get.json()["data"]["cycle_id"] == cid

    # 3. Submit Review Decision
    sample_id = cycle_data["selected_samples"][0]["image_id"]
    res_rev = client.post(
        f"/api/v1/active-learning/cycles/{cid}/review",
        json={
            "cycle_id": cid,
            "image_id": sample_id,
            "decision": "CONFIRMED",
            "reviewer_id": "API Reviewer",
            "notes": "Verified via API",
        },
    )
    assert res_rev.status_code == 200
    assert res_rev.json()["data"]["decision"] == "CONFIRMED"

    # 4. Get History
    res_hist = client.get("/api/v1/active-learning/cycles/history?dataset_id=safety_v2")
    assert res_hist.status_code == 200
    assert len(res_hist.json()["data"]) >= 2
