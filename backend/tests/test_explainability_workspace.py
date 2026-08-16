"""Unit and Integration Tests for Model Explainability & Visual Diagnostics Workspace."""

from fastapi.testclient import TestClient

from visionforge.explainability.generator import (
    generate_attribution_map,
)
from visionforge.explainability.schemas import (
    AddResearcherNoteRequest,
    CreateExplanationRequest,
    ExplanationConfig,
    ExplanationMethod,
    ExplanationStatus,
    ReviewExplanationRequest,
    ReviewRating,
)
from visionforge.explainability.service import get_explainability_service
from visionforge.main import app

client = TestClient(app)


def test_attribution_generator_genuine_spatial_computation():
    """Step 3-5: Test spatial attribution matrix and concentration calculation."""
    # 1. Correct prediction test (Energy concentrated inside box)
    target_box = [0.20, 0.20, 0.60, 0.60]
    art_correct, summary_c = generate_attribution_map(
        model_id="yolo11s.pt",
        target_class="helmet",
        target_box=target_box,
        is_correct=True,
    )
    assert art_correct.grid_width == 32
    assert art_correct.grid_height == 32
    assert len(art_correct.heatmap_grid) == 32
    assert art_correct.object_concentration_score > 0.60
    assert "concentrated within the predicted" in summary_c

    # 2. Incorrect prediction / failure test (Energy scattered on background)
    art_wrong, summary_w = generate_attribution_map(
        model_id="yolo11s.pt",
        target_class="vest",
        target_box=target_box,
        is_correct=False,
    )
    assert art_wrong.background_concentration_score > 0.35
    assert "background/context" in summary_w


def test_unsupported_model_or_method_handling():
    """Step 3 & 20: Test unsupported model topology raises informative diagnostic error."""
    svc = get_explainability_service()

    req = CreateExplanationRequest(
        model_id="yolo11s.pt",
        method=ExplanationMethod.PERTURBATION,
        sample_id="img_0099",
        target_class="helmet",
    )
    run = svc.create_explanation(req)
    assert run.status == ExplanationStatus.UNSUPPORTED
    assert run.error_message is not None
    assert "not supported" in run.error_message.lower()


def test_deterministic_caching_behavior():
    """Step 19 & 31: Test cache hit on identical inputs vs cache miss on modified config."""
    svc = get_explainability_service()

    # 1. First execution
    req1 = CreateExplanationRequest(
        model_id="yolo11s.pt",
        model_version="1.0.0",
        sample_id="img_cache_test_01",
        target_class="person",
        method=ExplanationMethod.GRAD_CAM,
        config=ExplanationConfig(method=ExplanationMethod.GRAD_CAM, colormap="jet"),
    )
    run1 = svc.create_explanation(req1)
    assert run1.status == ExplanationStatus.COMPLETED
    assert run1.cache_hit is False

    # 2. Duplicate execution with identical parameters -> CACHE HIT
    run2 = svc.create_explanation(req1)
    assert run2.explanation_id == run1.explanation_id
    assert run2.cache_hit is True

    # 3. Execution with modified method -> CACHE MISS (New Run Created)
    req3 = CreateExplanationRequest(
        model_id="yolo11s.pt",
        model_version="1.0.0",
        sample_id="img_cache_test_01",
        target_class="person",
        method=ExplanationMethod.LAYER_CAM,
        config=ExplanationConfig(method=ExplanationMethod.LAYER_CAM, colormap="jet"),
    )
    run3 = svc.create_explanation(req3)
    assert run3.explanation_id != run1.explanation_id
    assert run3.method == ExplanationMethod.LAYER_CAM
    assert run3.cache_hit is False


def test_human_review_and_researcher_notes():
    """Step 17 & 27: Test logging review ratings and researcher observation notes."""
    svc = get_explainability_service()

    req = CreateExplanationRequest(
        model_id="yolo11s.pt",
        sample_id="img_review_01",
        target_class="helmet",
        method=ExplanationMethod.GRAD_CAM,
    )
    run = svc.create_explanation(req)

    # 1. Review Rating
    reviewed = svc.review_explanation(
        run.explanation_id,
        ReviewExplanationRequest(
            rating=ReviewRating.USEFUL,
            note="Attribution aligns with chin strap and helmet shell.",
        ),
    )
    assert reviewed.review_rating == ReviewRating.USEFUL
    assert len(reviewed.researcher_notes) >= 1

    # 2. Add Researcher Observation Note
    noted = svc.add_researcher_note(
        run.explanation_id,
        AddResearcherNoteRequest(note="Secondary check confirms no background leakage."),
    )
    assert len(noted.researcher_notes) >= 2
    assert "Secondary check" in noted.researcher_notes[-1]


def test_side_by_side_comparison_and_difference_matrix():
    """Step 12 & 26: Test side-by-side comparison and attribution difference map."""
    svc = get_explainability_service()

    run_a = svc.create_explanation(
        CreateExplanationRequest(
            model_id="yolo11s.pt",
            sample_id="img_cmp_a",
            target_class="helmet",
            is_correct_prediction=True,
        )
    )
    run_b = svc.create_explanation(
        CreateExplanationRequest(
            model_id="yolo11s.pt",
            sample_id="img_cmp_b",
            target_class="helmet",
            is_correct_prediction=False,
        )
    )

    cmp_res = svc.compare_explanations(run_a.explanation_id, run_b.explanation_id)
    assert cmp_res.attribution_difference_score > 0.0
    assert len(cmp_res.attribution_difference_grid) == 32
    assert len(cmp_res.diagnostic_notes) > 0


def test_explainability_rest_api_endpoints():
    """Step 29: Test all explainability REST API routes."""
    # 1. Create explanation
    res_create = client.post(
        "/api/v1/explainability/explanations",
        json={
            "model_id": "yolo11s.pt",
            "sample_id": "img_api_test_01",
            "target_class": "vest",
            "method": "GRAD_CAM",
        },
    )
    assert res_create.status_code == 201
    run_data = res_create.json()["data"]
    exp_id = run_data["explanation_id"]
    assert exp_id.startswith("exp_")

    # 2. List explanations
    res_list = client.get("/api/v1/explainability/explanations")
    assert res_list.status_code == 200
    runs = res_list.json()["data"]
    assert len(runs) > 0

    # 3. Get single explanation
    res_get = client.get(f"/api/v1/explainability/explanations/{exp_id}")
    assert res_get.status_code == 200
    assert res_get.json()["data"]["explanation_id"] == exp_id

    # 4. Get status
    res_stat = client.get(f"/api/v1/explainability/explanations/{exp_id}/status")
    assert res_stat.status_code == 200
    assert res_stat.json()["data"]["status"] == "COMPLETED"

    # 5. Get artifact
    res_art = client.get(f"/api/v1/explainability/explanations/{exp_id}/artifact")
    assert res_art.status_code == 200
    assert "heatmap_grid" in res_art.json()["data"]

    # 6. Post review
    res_rev = client.post(
        f"/api/v1/explainability/explanations/{exp_id}/review",
        json={"rating": "NEEDS_INVESTIGATION", "note": "Unusual edge gradient"},
    )
    assert res_rev.status_code == 200
    assert res_rev.json()["data"]["review_rating"] == "NEEDS_INVESTIGATION"

    # 7. Post researcher note
    res_note = client.post(
        f"/api/v1/explainability/explanations/{exp_id}/notes",
        json={"note": "API test observation note"},
    )
    assert res_note.status_code == 200
    assert "API test observation note" in res_note.json()["data"]["researcher_notes"]
