"""Comprehensive Unit and Integration Tests for Model Evaluation & Error Analysis Workspace."""

from fastapi.testclient import TestClient

from visionforge.evaluation.analyzer import ErrorAnalyzer
from visionforge.evaluation.schemas import (
    ErrorCategory,
    EvaluationConfig,
)
from visionforge.evaluation.service import get_evaluation_service
from visionforge.main import app

client = TestClient(app)


def test_synthetic_evaluation_error_categorization():
    """Step 34: Test synthetic deterministic dataset with all 5 error categories."""
    config = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.25)
    analyzer = ErrorAnalyzer(config)

    # 1. Correct Match (TP)
    gt_tp = [{"class_id": 0, "class_name": "helmet", "bbox": [10.0, 10.0, 50.0, 50.0]}]
    pred_tp = [{"class_id": 0, "class_name": "helmet", "confidence": 0.92, "bbox": [11.0, 10.0, 49.0, 50.0]}]
    errs_tp = analyzer.analyze_image("img_tp", "img_tp.jpg", gt_tp, pred_tp)
    assert len(errs_tp) == 0

    # 2. False Positive (Pred with no GT)
    gt_fp = []
    pred_fp = [{"class_id": 1, "class_name": "vest", "confidence": 0.84, "bbox": [100.0, 100.0, 200.0, 300.0]}]
    errs_fp = analyzer.analyze_image("img_fp", "img_fp.jpg", gt_fp, pred_fp)
    assert len(errs_fp) == 1
    assert errs_fp[0].error_type == ErrorCategory.FALSE_POSITIVE

    # 3. False Negative (Missed GT with no Pred)
    gt_fn = [{"class_id": 2, "class_name": "person", "bbox": [50.0, 50.0, 150.0, 350.0]}]
    pred_fn = []
    errs_fn = analyzer.analyze_image("img_fn", "img_fn.jpg", gt_fn, pred_fn)
    assert len(errs_fn) == 1
    assert errs_fn[0].error_type == ErrorCategory.FALSE_NEGATIVE

    # 4. Wrong Class / Misclassification (Matched spatial box with wrong class)
    gt_wc = [{"class_id": 0, "class_name": "helmet", "bbox": [10.0, 10.0, 50.0, 50.0]}]
    pred_wc = [{"class_id": 3, "class_name": "gloves", "confidence": 0.77, "bbox": [10.0, 10.0, 50.0, 50.0]}]
    errs_wc = analyzer.analyze_image("img_wc", "img_wc.jpg", gt_wc, pred_wc)
    assert len(errs_wc) >= 1
    assert any(e.error_type == ErrorCategory.MISCLASSIFICATION for e in errs_wc)

    # 5. Poor Localization (Sub-threshold IoU 0.1 <= IoU < 0.5)
    gt_loc = [{"class_id": 0, "class_name": "helmet", "bbox": [0.0, 0.0, 100.0, 100.0]}]
    pred_loc = [{"class_id": 0, "class_name": "helmet", "confidence": 0.80, "bbox": [50.0, 0.0, 150.0, 100.0]}]
    errs_loc = analyzer.analyze_image("img_loc", "img_loc.jpg", gt_loc, pred_loc)
    assert any(e.error_type == ErrorCategory.POOR_LOCALIZATION for e in errs_loc)


def test_confidence_threshold_sweep_operating_points():
    """Step 6: Verify metrics across threshold operating points [0.20..0.80]."""
    svc = get_evaluation_service()
    runs = svc.list_evaluations()
    assert len(runs) > 0
    eval_id = runs[0].eval_id

    pts = svc.get_threshold_analysis(eval_id)
    assert len(pts) >= 7
    for pt in pts:
        assert 0.0 <= pt.confidence_threshold <= 1.0
        assert 0.0 <= pt.precision <= 1.0
        assert 0.0 <= pt.recall <= 1.0
        assert 0.0 <= pt.f1 <= 1.0


def test_confusion_pairs_aggregation():
    """Step 13: Verify measured classification confusion pairs aggregation."""
    svc = get_evaluation_service()
    runs = svc.list_evaluations()
    eval_id = runs[0].eval_id

    conf_data = svc.get_confusion_data(eval_id)
    assert len(conf_data.class_names) > 0
    assert len(conf_data.matrix) > 0
    assert len(conf_data.confusion_pairs) > 0

    first_pair = conf_data.confusion_pairs[0]
    assert first_pair.ground_truth_class != ""
    assert first_pair.predicted_class != ""
    assert first_pair.count > 0


def test_failure_gallery_filtering_and_prioritization():
    """Step 15 & 23: Verify failure gallery filters and transparent prioritization."""
    svc = get_evaluation_service()
    runs = svc.list_evaluations()
    eval_id = runs[0].eval_id

    # All failures sorted by priority
    failures = svc.get_failure_gallery(eval_id, sort_by="priority", limit=20)
    assert len(failures) > 0

    # Ensure descending priority score
    for i in range(len(failures) - 1):
        assert failures[i].review_priority >= failures[i + 1].review_priority

    # Filter by error category
    fn_failures = svc.get_failure_gallery(eval_id, error_type=ErrorCategory.FALSE_NEGATIVE)
    for f in fn_failures:
        assert f.error_type == ErrorCategory.FALSE_NEGATIVE


def test_visual_failure_clusters():
    """Step 17: Verify visual failure clustering into Cluster 1, Cluster 2, Cluster 3."""
    svc = get_evaluation_service()
    runs = svc.list_evaluations()
    eval_id = runs[0].eval_id

    clusters = svc.get_failure_clusters(eval_id)
    assert len(clusters) == 3
    assert clusters[0].label == "Cluster 1"
    assert clusters[1].label == "Cluster 2"
    assert clusters[2].label == "Cluster 3"
    for cl in clusters:
        assert cl.sample_count > 0
        assert len(cl.representative_sample_ids) > 0


def test_object_size_and_resolution_breakdowns():
    """Steps 19-20: Verify object size and image resolution performance breakdowns."""
    svc = get_evaluation_service()
    runs = svc.list_evaluations()
    eval_id = runs[0].eval_id

    report = svc.get_pattern_analysis(eval_id)
    assert len(report.size_performance) == 3
    size_categories = [s.size_category for s in report.size_performance]
    assert "small" in size_categories
    assert "medium" in size_categories
    assert "large" in size_categories

    assert len(report.resolution_performance) >= 3
    assert len(report.summary_findings) > 0


def test_active_learning_queue_integration():
    """Step 22: Verify sending failure sample directly to Active Learning."""
    svc = get_evaluation_service()
    runs = svc.list_evaluations()
    eval_id = runs[0].eval_id

    failures = svc.get_failure_gallery(eval_id, limit=5)
    sample_to_queue = failures[0].sample_id

    res = svc.send_failure_to_active_learning(eval_id, sample_to_queue)
    assert res["status"] == "QUEUED"
    assert res["sample_id"] == sample_to_queue

    # Check that review status updated
    detail = svc.get_failure_detail(eval_id, sample_to_queue)
    assert detail is not None
    assert detail.review_status == "SENT_TO_ACTIVE_LEARNING"


def test_same_dataset_model_comparison_and_regressions():
    """Steps 24-27 & 35: Verify model comparison on same dataset and regression detection."""
    svc = get_evaluation_service()

    # Create Baseline
    base = svc.create_benchmark_run(
        name="Baseline YOLOv11s",
        model_name="yolo11s_base.pt",
        dataset_id="safety_v2",
        dataset_version="v1.0.0",
        split_used="test",
        is_baseline=True,
    )

    # Create Candidate on SAME dataset version and split
    cand = svc.create_benchmark_run(
        name="Candidate YOLOv11s Finetuned",
        model_name="yolo11s_cand.pt",
        dataset_id="safety_v2",
        dataset_version="v1.0.0",
        split_used="test",
        is_baseline=False,
    )

    cmp_res = svc.compare_benchmarks(base.benchmark_id, cand.benchmark_id)
    assert cmp_res.is_directly_comparable is True
    assert "map50" in cmp_res.metric_deltas
    assert "precision" in cmp_res.metric_deltas
    assert "recall" in cmp_res.metric_deltas
    assert len(cmp_res.failure_deltas) > 0


def test_rest_api_evaluation_workspace_endpoints():
    """Step 32: Verify REST API endpoints for Model Evaluation workspace."""
    # 1. List runs
    res = client.get("/api/v1/evaluation/runs")
    assert res.status_code == 200
    runs = res.json()["data"]
    assert len(runs) > 0
    eval_id = runs[0]["eval_id"]

    # 2. Get thresholds
    res_thresh = client.get(f"/api/v1/evaluation/runs/{eval_id}/thresholds")
    assert res_thresh.status_code == 200
    assert len(res_thresh.json()["data"]) >= 7

    # 3. Get confusion
    res_conf = client.get(f"/api/v1/evaluation/runs/{eval_id}/confusion")
    assert res_conf.status_code == 200
    assert "matrix" in res_conf.json()["data"]

    # 4. Get PR curves
    res_pr = client.get(f"/api/v1/evaluation/runs/{eval_id}/pr-curves")
    assert res_pr.status_code == 200
    assert len(res_pr.json()["data"]["overall_pr_curve"]) > 0

    # 5. Get failures
    res_fail = client.get(f"/api/v1/evaluation/runs/{eval_id}/failures?limit=10")
    assert res_fail.status_code == 200
    failures = res_fail.json()["data"]
    assert len(failures) > 0
    sample_id = failures[0]["sample_id"]

    # 6. Get failure detail
    res_det = client.get(f"/api/v1/evaluation/runs/{eval_id}/failures/{sample_id}")
    assert res_det.status_code == 200
    assert res_det.json()["data"]["sample_id"] == sample_id

    # 7. Get failure clusters
    res_clust = client.get(f"/api/v1/evaluation/runs/{eval_id}/failure-clusters")
    assert res_clust.status_code == 200
    assert len(res_clust.json()["data"]) == 3

    # 8. Get pattern analysis
    res_pat = client.get(f"/api/v1/evaluation/runs/{eval_id}/pattern-analysis")
    assert res_pat.status_code == 200
    assert len(res_pat.json()["data"]["size_performance"]) == 3

    # 9. Send failure to active learning
    res_al = client.post(f"/api/v1/evaluation/runs/{eval_id}/failures/{sample_id}/active-learning")
    assert res_al.status_code == 200
    assert res_al.json()["data"]["status"] == "QUEUED"
