"""Unit and Integration Tests for VisionForge Research Benchmark & Evaluation Lab."""

import pytest
from fastapi.testclient import TestClient

from visionforge.evaluation.analyzer import ErrorAnalyzer
from visionforge.evaluation.metrics import (
    calculate_iou_boxes,
    compute_ap_from_pr,
    evaluate_detections,
)
from visionforge.evaluation.runtime import ModelRuntimeBenchmarker
from visionforge.evaluation.schemas import (
    ErrorCategory,
    EvaluationConfig,
    RegressionStatus,
)
from visionforge.evaluation.service import get_evaluation_service
from visionforge.main import app

client = TestClient(app)


def test_iou_calculation_and_formats():
    """Verify IoU calculation on known bounding boxes in both xyxy and xywh formats."""
    # 1. Identical boxes (IoU = 1.0)
    box1 = [10.0, 10.0, 50.0, 50.0]
    assert calculate_iou_boxes(box1, box1, format="xyxy") == pytest.approx(1.0)

    # 2. Disjoint boxes (IoU = 0.0)
    box2 = [100.0, 100.0, 150.0, 150.0]
    assert calculate_iou_boxes(box1, box2, format="xyxy") == pytest.approx(0.0)

    # 3. 50% Overlap box
    # Box A: [0, 0, 10, 10] -> Area 100
    # Box B: [5, 0, 15, 10] -> Area 100
    # Inter: [5, 0, 10, 10] -> Area 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3 ~ 0.3333
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [5.0, 0.0, 15.0, 10.0]
    assert calculate_iou_boxes(box_a, box_b, format="xyxy") == pytest.approx(1.0 / 3.0)

    # 4. xywh format test
    box_c = [5.0, 5.0, 10.0, 10.0]  # [0, 0, 10, 10]
    box_d = [10.0, 5.0, 10.0, 10.0]  # [5, 0, 15, 10]
    assert calculate_iou_boxes(box_c, box_d, format="xywh") == pytest.approx(1.0 / 3.0)


def test_ap_interpolation_mathematics():
    """Verify 101-point COCO-style PR interpolation of Average Precision."""
    # Perfect detection (Precision = 1.0 for all Recall) -> AP = 1.0
    recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
    precisions = [1.0, 1.0, 1.0, 1.0, 1.0]
    ap = compute_ap_from_pr(recalls, precisions, num_points=101)
    assert ap == pytest.approx(1.0)

    # Monotonically decaying precision
    recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
    precisions = [1.0, 0.8, 0.6, 0.4, 0.2]
    ap_decay = compute_ap_from_pr(recalls, precisions, num_points=101)
    assert 0.5 < ap_decay < 0.9


def test_detection_metrics_evaluation():
    """Verify multi-class detection evaluation engine calculates mAP, per-class metrics, and confusion matrix."""
    class_names = ["helmet", "person"]

    gts = {
        "img_01": [
            {"class_id": 0, "class_name": "helmet", "bbox": [10.0, 10.0, 50.0, 50.0]},
            {"class_id": 1, "class_name": "person", "bbox": [10.0, 10.0, 100.0, 200.0]},
        ]
    }
    preds = {
        "img_01": [
            {"class_id": 0, "class_name": "helmet", "confidence": 0.95, "bbox": [11.0, 10.0, 49.0, 50.0]},
            {"class_id": 1, "class_name": "person", "confidence": 0.90, "bbox": [12.0, 9.0, 99.0, 201.0]},
        ]
    }

    metrics, per_class, threshold_pts, confusion = evaluate_detections(
        ground_truths_by_image=gts,
        predictions_by_image=preds,
        class_names=class_names,
        iou_threshold=0.5,
        confidence_threshold=0.25,
    )

    assert metrics.map50 > 0.90
    assert metrics.map50_95 > 0.80
    assert len(per_class) == 2
    assert per_class[0].class_name == "helmet"
    assert per_class[0].support == 1
    assert per_class[0].true_positives == 1
    assert len(threshold_pts) > 5
    assert len(confusion.class_names) == 3  # helmet, person, background


def test_error_analyzer_taxonomy():
    """Verify diagnostic error analyzer categorizes detection failures correctly."""
    cfg = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.25)
    analyzer = ErrorAnalyzer(cfg)

    # 1. False Positive (Pred on empty ground truth)
    errs_fp = analyzer.analyze_image(
        image_id="img_fp",
        image_path="/test/img_fp.jpg",
        ground_truths=[],
        predictions=[{"class_id": 0, "class_name": "helmet", "confidence": 0.85, "bbox": [10.0, 10.0, 50.0, 50.0]}],
    )
    assert len(errs_fp) >= 1
    assert errs_fp[0].error_type in (ErrorCategory.FALSE_POSITIVE, ErrorCategory.BACKGROUND_DETECTION)

    # 2. False Negative (Unmatched GT)
    errs_fn = analyzer.analyze_image(
        image_id="img_fn",
        image_path="/test/img_fn.jpg",
        ground_truths=[{"class_id": 0, "class_name": "helmet", "bbox": [10.0, 10.0, 50.0, 50.0]}],
        predictions=[],
    )
    assert len(errs_fn) == 1
    assert errs_fn[0].error_type == ErrorCategory.FALSE_NEGATIVE

    # 3. Misclassification (High IoU with wrong class label)
    errs_misc = analyzer.analyze_image(
        image_id="img_misc",
        image_path="/test/img_misc.jpg",
        ground_truths=[{"class_id": 0, "class_name": "helmet", "bbox": [10.0, 10.0, 50.0, 50.0]}],
        predictions=[{"class_id": 1, "class_name": "person", "confidence": 0.88, "bbox": [10.0, 10.0, 50.0, 50.0]}],
    )
    assert len(errs_misc) >= 1
    assert any(e.error_type == ErrorCategory.MISCLASSIFICATION for e in errs_misc)

    # 4. Poor Localization (Sub-threshold IoU 0.1 <= IoU < 0.5)
    errs_loc = analyzer.analyze_image(
        image_id="img_loc",
        image_path="/test/img_loc.jpg",
        ground_truths=[{"class_id": 0, "class_name": "helmet", "bbox": [0.0, 0.0, 100.0, 100.0]}],
        predictions=[{"class_id": 0, "class_name": "helmet", "confidence": 0.80, "bbox": [50.0, 0.0, 150.0, 100.0]}],
    )
    assert len(errs_loc) >= 1
    assert any(e.error_type == ErrorCategory.POOR_LOCALIZATION for e in errs_loc)


def test_runtime_benchmarker():
    """Verify runtime benchmarker excludes warm-up iterations and computes percentiles and throughput."""
    benchmarker = ModelRuntimeBenchmarker(warmup_iterations=3, evaluated_iterations=15, device="cpu")
    runtime = benchmarker.benchmark_model(model_parameters_m=11.1, model_size_mb=22.5)

    assert runtime.warmup_iterations == 3
    assert runtime.evaluated_iterations == 15
    assert runtime.total_latency_ms_mean > 0.0
    assert runtime.total_latency_ms_p95 >= runtime.total_latency_ms_mean
    assert runtime.throughput_fps > 0.0
    assert runtime.model_parameters_m == 11.1


def test_fair_comparison_and_regression_detection():
    """Verify fair model comparison enforces scientific control and calculates deltas and regressions."""
    svc = get_evaluation_service()

    # 1. Create Baseline Run
    base_run = svc.create_benchmark_run(
        name="Test Baseline Model",
        model_name="yolo11s_base",
        model_version="1.0.0",
        dataset_id="test_safety_dataset",
        dataset_version="v1.0.0",
        dataset_fingerprint="sha256_test_fingerprint_01",
        split_used="test",
        is_baseline=True,
    )

    # 2. Create Candidate Run (Matching Dataset & Split)
    cand_run = svc.create_benchmark_run(
        name="Test Candidate Model",
        model_name="rtdetr_cand",
        model_version="2.0.0",
        dataset_id="test_safety_dataset",
        dataset_version="v1.0.0",
        dataset_fingerprint="sha256_test_fingerprint_01",
        split_used="test",
        is_baseline=False,
        baseline_benchmark_id=base_run.benchmark_id,
    )

    # Compare
    cmp_res = svc.compare_benchmarks(baseline_id=base_run.benchmark_id, candidate_id=cand_run.benchmark_id)
    assert cmp_res.is_directly_comparable is True
    assert "map50" in cmp_res.metric_deltas
    assert "throughput_fps" in cmp_res.metric_deltas
    assert cmp_res.regression_status in (
        RegressionStatus.IMPROVED,
        RegressionStatus.NEUTRAL,
        RegressionStatus.REGRESSION,
    )

    # 3. Create Incompatible Run (Different Split)
    incomp_run = svc.create_benchmark_run(
        name="Incompatible Split Model",
        model_name="yolo11s_val",
        model_version="1.0.0",
        dataset_id="test_safety_dataset",
        dataset_version="v1.0.0",
        dataset_fingerprint="sha256_test_fingerprint_01",
        split_used="val",  # Different split!
    )

    cmp_incomp = svc.compare_benchmarks(baseline_id=base_run.benchmark_id, candidate_id=incomp_run.benchmark_id)
    assert cmp_incomp.is_directly_comparable is False
    assert len(cmp_incomp.incompatibility_reasons) > 0
    assert cmp_incomp.regression_status == RegressionStatus.INCOMPARABLE


def test_benchmarks_api_endpoints():
    """Verify REST API endpoints for research benchmarks."""
    # 1. List benchmarks
    res_list = client.get("/api/v1/benchmarks/runs")
    assert res_list.status_code == 200
    runs = res_list.json()["data"]
    assert len(runs) >= 2
    bench_id = runs[0]["benchmark_id"]

    # 2. Get benchmark details
    res_get = client.get(f"/api/v1/benchmarks/runs/{bench_id}")
    assert res_get.status_code == 200
    assert res_get.json()["data"]["benchmark_id"] == bench_id

    # 3. Get Markdown report
    res_rep = client.get(f"/api/v1/benchmarks/runs/{bench_id}/report")
    assert res_rep.status_code == 200
    assert "# VisionForge Research Benchmark Report" in res_rep.text

    # 4. Get Failures
    res_fail = client.get(f"/api/v1/benchmarks/runs/{bench_id}/failures?limit=10")
    assert res_fail.status_code == 200
    assert isinstance(res_fail.json()["data"], list)

    # 5. History
    res_hist = client.get("/api/v1/benchmarks/history")
    assert res_hist.status_code == 200
    assert len(res_hist.json()["data"]) >= 2
