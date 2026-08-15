"""Unit and Integration Tests for VisionForge Data-Centric Computer Vision Workspace."""

import pytest
from fastapi.testclient import TestClient

from visionforge.datasets.analyzer import DatasetQualityAnalyzer
from visionforge.datasets.intelligence_schemas import (
    AnnotationQualityFlag,
    CurationDecision,
    ImageQualityFlag,
)
from visionforge.datasets.intelligence_service import get_dataset_intelligence_service
from visionforge.main import app

client = TestClient(app)


def test_dataset_profiling_and_class_distribution():
    """Verify dataset profiling generates authentic class distribution and telemetry."""
    svc = get_dataset_intelligence_service()
    profile = svc.get_or_compute_profile(dataset_id="safety_v2", dataset_version="v2.0.0")

    assert profile.dataset_id == "safety_v2"
    assert profile.total_samples > 4000
    assert profile.total_annotations > 8000
    assert len(profile.class_distribution) >= 4
    assert any(c.class_name == "person" for c in profile.class_distribution)
    assert any(c.is_rare_class for c in profile.class_distribution if c.class_name == "gloves")
    assert profile.image_statistics.mean_width > 0
    assert profile.annotation_statistics.mean_boxes_per_image > 0
    assert len(profile.class_cooccurrence) > 0


def test_image_quality_flags():
    """Verify image quality analyzer flags corrupted, small, and extreme aspect ratio images."""
    # 1. Corrupted Image
    iss_corrupt = DatasetQualityAnalyzer.inspect_image_quality(
        sample_id="img_c",
        image_path="/test/corrupt.jpg",
        width=0,
        height=0,
        is_corrupted=True,
    )
    assert len(iss_corrupt) == 1
    assert iss_corrupt[0].flag == ImageQualityFlag.CORRUPTED.value
    assert iss_corrupt[0].severity == "CRITICAL"

    # 2. Very Small Image (40x40px)
    iss_small = DatasetQualityAnalyzer.inspect_image_quality(
        sample_id="img_s",
        image_path="/test/small.jpg",
        width=40,
        height=40,
    )
    assert any(i.flag == ImageQualityFlag.VERY_SMALL.value for i in iss_small)

    # 3. Extreme Aspect Ratio (800x100px = 8:1)
    iss_aspect = DatasetQualityAnalyzer.inspect_image_quality(
        sample_id="img_a",
        image_path="/test/banner.jpg",
        width=800,
        height=100,
    )
    assert any(i.flag == ImageQualityFlag.EXTREME_ASPECT_RATIO.value for i in iss_aspect)


def test_annotation_quality_flags():
    """Verify bounding box validator flags zero-area, out-of-bounds, tiny, and duplicate boxes."""
    # 1. Zero Area Box
    annos_zero = [{"class_name": "helmet", "bbox": [100.0, 100.0, 100.0, 150.0]}]  # x1=x2 -> w=0
    iss_zero = DatasetQualityAnalyzer.inspect_annotation_quality(
        sample_id="img_z",
        image_path="/test/z.jpg",
        image_width=1280,
        image_height=720,
        annotations=annos_zero,
    )
    assert any(i.flag == AnnotationQualityFlag.ZERO_AREA_BOX.value for i in iss_zero)

    # 2. Out of Bounds Box
    annos_oob = [{"class_name": "person", "bbox": [1100.0, 200.0, 1400.0, 600.0]}]  # x2=1400 > 1280
    iss_oob = DatasetQualityAnalyzer.inspect_annotation_quality(
        sample_id="img_o",
        image_path="/test/o.jpg",
        image_width=1280,
        image_height=720,
        annotations=annos_oob,
    )
    assert any(i.flag == AnnotationQualityFlag.OUT_OF_BOUNDS_COORDINATES.value for i in iss_oob)

    # 3. Tiny Box
    annos_tiny = [{"class_name": "bolt", "bbox": [10.0, 10.0, 12.0, 12.0]}]  # Area = 4 / (1280*720) = 0.000004
    iss_tiny = DatasetQualityAnalyzer.inspect_annotation_quality(
        sample_id="img_t",
        image_path="/test/t.jpg",
        image_width=1280,
        image_height=720,
        annotations=annos_tiny,
    )
    assert any(i.flag == AnnotationQualityFlag.TINY_BOX.value for i in iss_tiny)

    # 4. Duplicate Box (IoU >= 0.95)
    annos_dup = [
        {"class_name": "vest", "bbox": [100.0, 100.0, 200.0, 200.0]},
        {"class_name": "vest", "bbox": [101.0, 100.0, 201.0, 200.0]},
    ]
    iss_dup = DatasetQualityAnalyzer.inspect_annotation_quality(
        sample_id="img_d",
        image_path="/test/d.jpg",
        image_width=1280,
        image_height=720,
        annotations=annos_dup,
    )
    assert any(i.flag == AnnotationQualityFlag.DUPLICATE_BOX.value for i in iss_dup)


def test_class_cooccurrence_computation():
    """Verify pairwise class co-occurrence matrix calculation."""
    annos = {
        "img1": [{"class_name": "person"}, {"class_name": "helmet"}],
        "img2": [{"class_name": "person"}, {"class_name": "helmet"}, {"class_name": "vest"}],
        "img3": [{"class_name": "person"}],
    }
    coocs = DatasetQualityAnalyzer.compute_class_cooccurrence(annos, ["person", "helmet", "vest"])

    person_helmet = next((c for c in coocs if (c.class_a == "person" and c.class_b == "helmet") or (c.class_a == "helmet" and c.class_b == "person")), None)
    assert person_helmet is not None
    assert person_helmet.cooccurrence_count == 2
    assert person_helmet.cooccurrence_rate == pytest.approx(2 / 3.0, rel=1e-2)


def test_cross_split_leakage_detection():
    """Verify exact hash and visual representation leakage detection across partitions."""
    samples_by_split = {
        "train": [
            {"id": "train_01", "file_path": "/train/01.jpg", "content_hash": "hash_identical_01"},
            {"id": "train_02", "file_path": "/train/02.jpg", "content_hash": "hash_train_02"},
        ],
        "test": [
            {"id": "test_01", "file_path": "/test/01.jpg", "content_hash": "hash_identical_01"},  # EXACT DUPLICATE
            {"id": "test_02", "file_path": "/test/02.jpg", "content_hash": "hash_test_02"},
        ],
    }

    leaks = DatasetQualityAnalyzer.detect_cross_split_leakage(samples_by_split)
    assert len(leaks) >= 1
    assert leaks[0].sample_a_id == "train_01"
    assert leaks[0].sample_b_id == "test_01"
    assert leaks[0].match_type == "EXACT_HASH"
    assert leaks[0].similarity_score == 1.0


def test_hard_sample_prioritization():
    """Verify composite hard sample difficulty ranking."""
    samples = [
        {"id": "s_easy", "file_path": "/img/easy.jpg", "split": "train", "confidence": 0.95, "annotations": [{"class_name": "person"}]},
        {"id": "s_hard", "file_path": "/img/hard.jpg", "split": "train", "confidence": 0.40, "annotations": [{"class_name": "person"}, {"class_name": "helmet"}, {"class_name": "vest"}, {"class_name": "gloves"}]},
    ]
    eval_fails = [
        {"image_id": "s_hard", "predicted_class": "person"},
        {"image_id": "s_hard", "predicted_class": "head"},
    ]

    ranked = DatasetQualityAnalyzer.prioritize_hard_samples(samples, eval_fails)
    assert len(ranked) == 2
    assert ranked[0].sample_id == "s_hard"
    assert ranked[0].prioritization_score > ranked[1].prioritization_score


def test_dataset_diff_computation():
    """Verify dataset diff between baseline version v1.0.0 and curated v2.0.0."""
    svc = get_dataset_intelligence_service()
    diff = svc.compute_dataset_diff("safety_v2", "v1.0.0", "v2.0.0")

    assert diff.dataset_id == "safety_v2"
    assert diff.version_a == "v1.0.0"
    assert diff.version_b == "v2.0.0"
    assert diff.annotations_count_delta > 0
    assert "Version" in diff.summary


def test_curation_decision_submission():
    """Verify recording curation decisions into Human Review Queue."""
    svc = get_dataset_intelligence_service()
    decision = CurationDecision(
        review_id="rev_test_01",
        sample_id="sample_test_01",
        issue_id="iss_test_01",
        decision="NEEDS_CORRECTION",
        category="annotation_review",
        notes="Tighten bounding box around helmet.",
        reviewer="Test Reviewer",
    )

    svc.record_curation_decision(decision)
    decisions = svc.list_curation_decisions(sample_id="sample_test_01")
    assert len(decisions) >= 1
    assert decisions[-1].decision == "NEEDS_CORRECTION"


def test_datasets_intelligence_api_endpoints():
    """Verify REST API endpoints for Dataset Intelligence workspace."""
    # 1. Profile
    res_prof = client.get("/api/v1/datasets/intelligence/profile?dataset_id=safety_v2&version=v2.0.0")
    assert res_prof.status_code == 200
    assert res_prof.json()["data"]["dataset_id"] == "safety_v2"

    # 2. Health Scorecard
    res_health = client.get("/api/v1/datasets/intelligence/health?dataset_id=safety_v2")
    assert res_health.status_code == 200
    assert "overall_integrity" in res_health.json()["data"]

    # 3. Quality Issues
    res_iss = client.get("/api/v1/datasets/intelligence/issues?dataset_id=safety_v2")
    assert res_iss.status_code == 200
    assert isinstance(res_iss.json()["data"], list)

    # 4. Leakage
    res_leak = client.get("/api/v1/datasets/intelligence/leakage?dataset_id=safety_v2")
    assert res_leak.status_code == 200
    assert isinstance(res_leak.json()["data"], list)

    # 5. Hard Samples
    res_hard = client.get("/api/v1/datasets/intelligence/hard-samples?dataset_id=safety_v2")
    assert res_hard.status_code == 200
    assert isinstance(res_hard.json()["data"], list)

    # 6. Diff
    res_diff = client.get("/api/v1/datasets/intelligence/diff?dataset_id=safety_v2&version_a=v1.0.0&version_b=v2.0.0")
    assert res_diff.status_code == 200
    assert res_diff.json()["data"]["annotations_count_delta"] > 0

    # 7. Report
    res_rep = client.get("/api/v1/datasets/intelligence/report?dataset_id=safety_v2&version=v2.0.0")
    assert res_rep.status_code == 200
    assert "# VisionForge Dataset Intelligence Report" in res_rep.text
