"""Tests for Model Evaluation and Error Analysis."""

import pytest

from visionforge.evaluation.analyzer import ErrorAnalyzer, calculate_iou
from visionforge.evaluation.schemas import (
    ErrorCategory,
    EvaluationConfig,
    EvaluationRun,
    EvaluationStatus,
)
from visionforge.evaluation.service import EvaluationService


def test_calculate_iou_perfect_match():
    # bbox format: [x_center, y_center, width, height]
    box_a = [0.5, 0.5, 0.2, 0.2]
    box_b = [0.5, 0.5, 0.2, 0.2]
    iou = calculate_iou(box_a, box_b)
    assert iou == pytest.approx(1.0)


def test_calculate_iou_no_overlap():
    box_a = [0.2, 0.2, 0.1, 0.1]
    box_b = [0.8, 0.8, 0.1, 0.1]
    iou = calculate_iou(box_a, box_b)
    assert iou == pytest.approx(0.0)


def test_calculate_iou_partial_overlap():
    box_a = [0.5, 0.5, 0.4, 0.4]
    box_b = [0.6, 0.5, 0.4, 0.4]
    iou = calculate_iou(box_a, box_b)
    assert iou > 0.0 and iou < 1.0


def test_error_analyzer_true_positive():
    config = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.25)
    analyzer = ErrorAnalyzer(config)

    gt = [{"class_id": 1, "class_name": "cat", "bbox": [0.5, 0.5, 0.4, 0.4]}]
    pred = [{"class_id": 1, "class_name": "cat", "confidence": 0.9, "bbox": [0.5, 0.5, 0.4, 0.4]}]

    errors = analyzer.analyze_image("img1", "img1.jpg", gt, pred)
    # TP doesn't generate an error prediction record unless asked, currently we only return errors
    assert len(errors) == 0


def test_error_analyzer_false_positive():
    config = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.25)
    analyzer = ErrorAnalyzer(config)

    gt = []
    pred = [{"class_id": 1, "class_name": "cat", "confidence": 0.9, "bbox": [0.5, 0.5, 0.4, 0.4]}]

    errors = analyzer.analyze_image("img1", "img1.jpg", gt, pred)

    assert len(errors) == 1
    assert errors[0].error_type == ErrorCategory.FALSE_POSITIVE


def test_error_analyzer_false_negative():
    config = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.25)
    analyzer = ErrorAnalyzer(config)

    gt = [{"class_id": 1, "class_name": "cat", "bbox": [0.5, 0.5, 0.4, 0.4]}]
    pred = []

    errors = analyzer.analyze_image("img1", "img1.jpg", gt, pred)

    assert len(errors) == 1
    assert errors[0].error_type == ErrorCategory.FALSE_NEGATIVE


def test_error_analyzer_misclassification():
    config = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.25)
    analyzer = ErrorAnalyzer(config)

    gt = [{"class_id": 1, "class_name": "cat", "bbox": [0.5, 0.5, 0.4, 0.4]}]
    pred = [{"class_id": 2, "class_name": "dog", "confidence": 0.9, "bbox": [0.5, 0.5, 0.4, 0.4]}]

    errors = analyzer.analyze_image("img1", "img1.jpg", gt, pred)

    assert len(errors) == 1
    assert errors[0].error_type == ErrorCategory.MISCLASSIFICATION


def test_error_analyzer_poor_localization():
    config = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.25)
    analyzer = ErrorAnalyzer(config)

    gt = [{"class_id": 1, "class_name": "cat", "bbox": [0.5, 0.5, 0.4, 0.4]}]
    # Shifted slightly, IoU > 0.1 but < 0.5
    pred = [{"class_id": 1, "class_name": "cat", "confidence": 0.9, "bbox": [0.65, 0.65, 0.4, 0.4]}]

    errors = analyzer.analyze_image("img1", "img1.jpg", gt, pred)

    # We should get a POOR_LOCALIZATION and a FALSE_NEGATIVE (because the GT was not "matched" effectively for recall)
    types = [e.error_type for e in errors]
    assert ErrorCategory.POOR_LOCALIZATION in types
    assert ErrorCategory.FALSE_NEGATIVE in types


def test_error_analyzer_low_confidence():
    config = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.5)
    analyzer = ErrorAnalyzer(config)

    gt = []
    pred = [{"class_id": 1, "class_name": "cat", "confidence": 0.3, "bbox": [0.5, 0.5, 0.4, 0.4]}]

    errors = analyzer.analyze_image("img1", "img1.jpg", gt, pred)

    assert len(errors) == 1
    assert errors[0].error_type == ErrorCategory.LOW_CONFIDENCE


def test_benchmark_fairness_validation():
    # Setup mock runs
    run1 = EvaluationRun(
        eval_id="eval_1",
        model_name="yolo11s.pt",
        dataset_id="safety_v2",
        dataset_version="v2",
        split_used="test",
        status=EvaluationStatus.COMPLETED,
        config=EvaluationConfig(),
    )
    run2 = EvaluationRun(
        eval_id="eval_2",
        model_name="rtdetr-l.pt",
        dataset_id="safety_v2",
        dataset_version="v1",  # DIFFERENT VERSION
        split_used="test",
        status=EvaluationStatus.COMPLETED,
        config=EvaluationConfig(),
    )

    svc = EvaluationService()
    # Manually inject to service memory since we mock
    svc.get_evaluation = lambda eid: run1 if eid == "eval_1" else run2

    with pytest.raises(ValueError, match="Fair comparison violated"):
        svc.create_benchmark(["eval_1", "eval_2"])
