"""Diagnostic Error Analyzer & Failure Classifier for VisionForge.

Provides deep error taxonomy and evidence extraction:
1. Missed Object (False Negative)
2. False Positive (Background Detection)
3. Wrong Class (Misclassification on matched spatial detection)
4. Poor Localization (0.10 <= IoU < matching threshold)
5. Duplicate Detection (Redundant detection on already matched ground truth)
6. Low Confidence (Filtered predictions below confidence threshold)
"""

import logging
import uuid
from typing import Any

from visionforge.evaluation.metrics import calculate_iou_boxes
from visionforge.evaluation.schemas import (
    ErrorCategory,
    EvaluationConfig,
    FailureSampleDetail,
)

logger = logging.getLogger("visionforge.evaluation.analyzer")

# Backward-compatibility alias
calculate_iou = calculate_iou_boxes


def _compute_bbox_area(bbox: list[float] | None) -> float:
    """Calculate pixel or relative area from bbox [x1, y1, x2, y2] or [xc, yc, w, h]."""
    if not bbox or len(bbox) < 4:
        return 0.0
    # Check if xywh format
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        return bbox[2] * bbox[3]
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return w * h


def _categorize_object_size(bbox: list[float] | None) -> str:
    """Classify bounding box into small (<32^2), medium (32^2-96^2), or large (>96^2)."""
    area = _compute_bbox_area(bbox)
    if area <= 1.0:
        area = area * 640.0 * 640.0
    if area < 32.0 * 32.0:
        return "small"
    if area <= 96.0 * 96.0:
        return "medium"
    return "large"


def _calculate_review_priority(
    confidence: float | None,
    iou: float | None,
    error_type: ErrorCategory,
) -> float:
    """Compute transparent review priority score in [0.0, 1.0]."""
    conf = confidence if confidence is not None else 0.5
    iou_val = iou if iou is not None else 0.0

    weight_map = {
        ErrorCategory.MISCLASSIFICATION: 1.0,
        ErrorCategory.WRONG_CLASS: 1.0,
        ErrorCategory.FALSE_NEGATIVE: 0.85,
        ErrorCategory.SMALL_OBJECT_FAILURE: 0.80,
        ErrorCategory.POOR_LOCALIZATION: 0.70,
        ErrorCategory.FALSE_POSITIVE: 0.65,
        ErrorCategory.BACKGROUND_DETECTION: 0.65,
        ErrorCategory.DUPLICATE_DETECTION: 0.50,
        ErrorCategory.LOW_CONFIDENCE: 0.30,
    }
    err_weight = weight_map.get(error_type, 0.5)
    score = (0.40 * (1.0 - conf)) + (0.35 * (1.0 - iou_val)) + (0.25 * err_weight)
    return round(max(0.0, min(1.0, score)), 4)


class ErrorAnalyzer:
    """Engine to perform granular diagnostic error taxonomy analysis on predictions."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def analyze_image(
        self,
        image_id: str,
        image_path: str,
        ground_truths: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        eval_id: str = "eval_default",
        model_id: str = "yolo11s.pt",
        model_version: str = "1.0.0",
        dataset_id: str = "safety_v2",
        dataset_version: str = "v1.0.0",
        split: str = "test",
    ) -> list[FailureSampleDetail]:
        """Analyze a single image and return detailed categorized error predictions."""
        errors: list[FailureSampleDetail] = []

        valid_preds = [
            p for p in predictions if p.get("confidence", 1.0) >= self.config.confidence_threshold
        ]
        valid_preds.sort(key=lambda x: x.get("confidence", 1.0), reverse=True)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        # 1. Match high-confidence predictions to ground truths
        for p_idx, pred in enumerate(valid_preds):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                iou = calculate_iou_boxes(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.config.iou_threshold and best_gt_idx >= 0:
                gt = ground_truths[best_gt_idx]
                if best_gt_idx in matched_gt:
                    # Duplicate redundant detection
                    size_cat = _categorize_object_size(pred.get("bbox"))
                    p_score = _calculate_review_priority(
                        pred.get("confidence"), best_iou, ErrorCategory.DUPLICATE_DETECTION
                    )
                    errors.append(
                        FailureSampleDetail(
                            sample_id=f"fail_{uuid.uuid4().hex[:8]}",
                            eval_id=eval_id,
                            image_id=image_id,
                            image_path=image_path,
                            ground_truth_class=gt.get("class_name"),
                            predicted_class=pred.get("class_name"),
                            confidence=pred.get("confidence"),
                            iou=round(best_iou, 4),
                            error_type=ErrorCategory.DUPLICATE_DETECTION,
                            model_id=model_id,
                            model_version=model_version,
                            dataset_id=dataset_id,
                            dataset_version=dataset_version,
                            split=split,
                            object_size_category=size_cat,
                            gt_bbox=gt.get("bbox"),
                            pred_bbox=pred.get("bbox"),
                            nearby_ground_truths=ground_truths,
                            competing_predictions=valid_preds,
                            review_priority=p_score,
                        )
                    )
                    matched_pred.add(p_idx)
                elif pred.get("class_id") == gt.get("class_id"):
                    # True Positive Match
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(p_idx)
                else:
                    # Misclassification / Wrong Class
                    size_cat = _categorize_object_size(pred.get("bbox"))
                    p_score = _calculate_review_priority(
                        pred.get("confidence"), best_iou, ErrorCategory.MISCLASSIFICATION
                    )
                    errors.append(
                        FailureSampleDetail(
                            sample_id=f"fail_{uuid.uuid4().hex[:8]}",
                            eval_id=eval_id,
                            image_id=image_id,
                            image_path=image_path,
                            ground_truth_class=gt.get("class_name"),
                            predicted_class=pred.get("class_name"),
                            confidence=pred.get("confidence"),
                            iou=round(best_iou, 4),
                            error_type=ErrorCategory.MISCLASSIFICATION,
                            model_id=model_id,
                            model_version=model_version,
                            dataset_id=dataset_id,
                            dataset_version=dataset_version,
                            split=split,
                            object_size_category=size_cat,
                            gt_bbox=gt.get("bbox"),
                            pred_bbox=pred.get("bbox"),
                            nearby_ground_truths=ground_truths,
                            competing_predictions=valid_preds,
                            review_priority=p_score,
                        )
                    )
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(p_idx)

            elif 0.10 <= best_iou < self.config.iou_threshold and best_gt_idx >= 0:
                # Poor localization error
                gt = ground_truths[best_gt_idx]
                size_cat = _categorize_object_size(pred.get("bbox"))
                p_score = _calculate_review_priority(
                    pred.get("confidence"), best_iou, ErrorCategory.POOR_LOCALIZATION
                )
                errors.append(
                    FailureSampleDetail(
                        sample_id=f"fail_{uuid.uuid4().hex[:8]}",
                        eval_id=eval_id,
                        image_id=image_id,
                        image_path=image_path,
                        ground_truth_class=gt.get("class_name"),
                        predicted_class=pred.get("class_name"),
                        confidence=pred.get("confidence"),
                        iou=round(best_iou, 4),
                        error_type=ErrorCategory.POOR_LOCALIZATION,
                        model_id=model_id,
                        model_version=model_version,
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        split=split,
                        object_size_category=size_cat,
                        gt_bbox=gt.get("bbox"),
                        pred_bbox=pred.get("bbox"),
                        nearby_ground_truths=ground_truths,
                        competing_predictions=valid_preds,
                        review_priority=p_score,
                    )
                )
                matched_pred.add(p_idx)

        # 2. Unmatched Predictions are False Positives
        for p_idx, pred in enumerate(valid_preds):
            if p_idx not in matched_pred:
                size_cat = _categorize_object_size(pred.get("bbox"))
                p_score = _calculate_review_priority(
                    pred.get("confidence"), 0.0, ErrorCategory.FALSE_POSITIVE
                )
                errors.append(
                    FailureSampleDetail(
                        sample_id=f"fail_{uuid.uuid4().hex[:8]}",
                        eval_id=eval_id,
                        image_id=image_id,
                        image_path=image_path,
                        predicted_class=pred.get("class_name"),
                        confidence=pred.get("confidence"),
                        iou=0.0,
                        error_type=ErrorCategory.FALSE_POSITIVE,
                        model_id=model_id,
                        model_version=model_version,
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        split=split,
                        object_size_category=size_cat,
                        pred_bbox=pred.get("bbox"),
                        nearby_ground_truths=ground_truths,
                        competing_predictions=valid_preds,
                        review_priority=p_score,
                    )
                )

        # 3. Unmatched Ground Truths are False Negatives (Missed Objects)
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx not in matched_gt:
                size_cat = _categorize_object_size(gt.get("bbox"))
                p_score = _calculate_review_priority(0.0, 0.0, ErrorCategory.FALSE_NEGATIVE)
                errors.append(
                    FailureSampleDetail(
                        sample_id=f"fail_{uuid.uuid4().hex[:8]}",
                        eval_id=eval_id,
                        image_id=image_id,
                        image_path=image_path,
                        ground_truth_class=gt.get("class_name"),
                        confidence=0.0,
                        iou=0.0,
                        error_type=ErrorCategory.FALSE_NEGATIVE,
                        model_id=model_id,
                        model_version=model_version,
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        split=split,
                        object_size_category=size_cat,
                        gt_bbox=gt.get("bbox"),
                        nearby_ground_truths=ground_truths,
                        competing_predictions=valid_preds,
                        review_priority=p_score,
                    )
                )

        # 4. Low Confidence Detections (filtered out below threshold)
        low_conf_preds = [
            p for p in predictions if p.get("confidence", 1.0) < self.config.confidence_threshold
        ]
        for pred in low_conf_preds:
            size_cat = _categorize_object_size(pred.get("bbox"))
            errors.append(
                FailureSampleDetail(
                    sample_id=f"fail_{uuid.uuid4().hex[:8]}",
                    eval_id=eval_id,
                    image_id=image_id,
                    image_path=image_path,
                    predicted_class=pred.get("class_name"),
                    confidence=pred.get("confidence"),
                    iou=0.0,
                    error_type=ErrorCategory.LOW_CONFIDENCE,
                    model_id=model_id,
                    model_version=model_version,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    split=split,
                    object_size_category=size_cat,
                    pred_bbox=pred.get("bbox"),
                    nearby_ground_truths=ground_truths,
                    competing_predictions=valid_preds,
                    review_priority=0.2,
                )
            )

        return errors
