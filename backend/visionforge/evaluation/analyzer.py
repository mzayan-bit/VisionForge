"""Error Analyzer for VisionForge Evaluation."""

import logging
from typing import Any

from visionforge.evaluation.schemas import (
    ErrorCategory,
    ErrorPrediction,
    EvaluationConfig,
)

logger = logging.getLogger("visionforge.evaluation.analyzer")


def calculate_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x_center, y_center, width, height] format.
    """
    # Convert to [x1, y1, x2, y2]
    x_a1 = box_a[0] - box_a[2] / 2
    y_a1 = box_a[1] - box_a[3] / 2
    x_a2 = box_a[0] + box_a[2] / 2
    y_a2 = box_a[1] + box_a[3] / 2

    x_b1 = box_b[0] - box_b[2] / 2
    y_b1 = box_b[1] - box_b[3] / 2
    x_b2 = box_b[0] + box_b[2] / 2
    y_b2 = box_b[1] + box_b[3] / 2

    # Intersection area
    inter_x1 = max(x_a1, x_b1)
    inter_y1 = max(y_a1, y_b1)
    inter_x2 = min(x_a2, x_b2)
    inter_y2 = min(y_a2, y_b2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    # Union area
    box_a_area = (x_a2 - x_a1) * (y_a2 - y_a1)
    box_b_area = (x_b2 - x_b1) * (y_b2 - y_b1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area + 1e-9)
    return iou


class ErrorAnalyzer:
    """Engine to perform diagnostic error analysis on predictions."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def analyze_image(
        self,
        image_id: str,
        image_path: str,
        ground_truths: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> list[ErrorPrediction]:
        """Analyze a single image and return detailed error predictions."""
        errors: list[ErrorPrediction] = []

        # Filter predictions by confidence
        valid_preds = [p for p in predictions if p["confidence"] >= self.config.confidence_threshold]

        # 1. Match predictions to ground truths using IoU
        matched_gt = set()
        matched_pred = set()

        # Sort predictions by confidence descending
        valid_preds.sort(key=lambda x: x["confidence"], reverse=True)

        for p_idx, pred in enumerate(valid_preds):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.config.iou_threshold:
                gt = ground_truths[best_gt_idx]
                if pred["class_id"] == gt["class_id"]:
                    # True Positive
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(p_idx)
                else:
                    # Misclassification
                    errors.append(ErrorPrediction(
                        image_id=image_id,
                        image_path=image_path,
                        ground_truth_class=gt["class_name"],
                        predicted_class=pred["class_name"],
                        confidence=pred["confidence"],
                        iou=best_iou,
                        error_type=ErrorCategory.MISCLASSIFICATION,
                        gt_bbox=gt["bbox"],
                        pred_bbox=pred["bbox"],
                    ))
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(p_idx)
            elif best_iou > 0.1: # Threshold for poor localization
                gt = ground_truths[best_gt_idx]
                errors.append(ErrorPrediction(
                    image_id=image_id,
                    image_path=image_path,
                    ground_truth_class=gt.get("class_name"),
                    predicted_class=pred["class_name"],
                    confidence=pred["confidence"],
                    iou=best_iou,
                    error_type=ErrorCategory.POOR_LOCALIZATION,
                    gt_bbox=gt["bbox"],
                    pred_bbox=pred["bbox"],
                ))
                # Do not mark as matched GT so it might still be a false negative
                matched_pred.add(p_idx)

        # 2. Any unmatched predictions are False Positives or Background Detections
        for p_idx, pred in enumerate(valid_preds):
            if p_idx not in matched_pred:
                errors.append(ErrorPrediction(
                    image_id=image_id,
                    image_path=image_path,
                    predicted_class=pred["class_name"],
                    confidence=pred["confidence"],
                    iou=0.0,
                    error_type=ErrorCategory.FALSE_POSITIVE,
                    pred_bbox=pred["bbox"],
                ))

        # 3. Any unmatched ground truths are False Negatives (Missed Objects)
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx not in matched_gt:
                errors.append(ErrorPrediction(
                    image_id=image_id,
                    image_path=image_path,
                    ground_truth_class=gt["class_name"],
                    confidence=0.0,
                    iou=0.0,
                    error_type=ErrorCategory.FALSE_NEGATIVE,
                    gt_bbox=gt["bbox"],
                ))

        # 4. Low Confidence errors (predictions that were filtered out but might have matched a GT)
        low_conf_preds = [p for p in predictions if p["confidence"] < self.config.confidence_threshold]
        for pred in low_conf_preds:
             errors.append(ErrorPrediction(
                 image_id=image_id,
                 image_path=image_path,
                 predicted_class=pred["class_name"],
                 confidence=pred["confidence"],
                 iou=None,
                 error_type=ErrorCategory.LOW_CONFIDENCE,
                 pred_bbox=pred["bbox"],
             ))

        return errors
