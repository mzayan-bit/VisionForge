"""Diagnostic Error Analyzer & Failure Classifier for VisionForge."""

import logging
from typing import Any

from visionforge.evaluation.metrics import calculate_iou_boxes
from visionforge.evaluation.schemas import (
    ErrorCategory,
    ErrorPrediction,
    EvaluationConfig,
)

logger = logging.getLogger("visionforge.evaluation.analyzer")

# Backward-compatibility alias
calculate_iou = calculate_iou_boxes


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
    ) -> list[ErrorPrediction]:
        """Analyze a single image and return detailed categorized error predictions."""
        errors: list[ErrorPrediction] = []

        valid_preds = [p for p in predictions if p.get("confidence", 1.0) >= self.config.confidence_threshold]
        valid_preds.sort(key=lambda x: x.get("confidence", 1.0), reverse=True)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        gt_match_counts: dict[int, int] = {}

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
                    # Duplicate redundant detection on already matched ground truth
                    errors.append(
                        ErrorPrediction(
                            image_id=image_id,
                            image_path=image_path,
                            ground_truth_class=gt.get("class_name"),
                            predicted_class=pred.get("class_name"),
                            confidence=pred.get("confidence"),
                            iou=best_iou,
                            error_type=ErrorCategory.DUPLICATE_DETECTION,
                            gt_bbox=gt.get("bbox"),
                            pred_bbox=pred.get("bbox"),
                            sample_link=f"/datasets?image={image_id}",
                        )
                    )
                    matched_pred.add(p_idx)
                elif pred.get("class_id") == gt.get("class_id"):
                    # True Positive Match
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(p_idx)
                    gt_match_counts[best_gt_idx] = gt_match_counts.get(best_gt_idx, 0) + 1
                else:
                    # Misclassification (high IoU match with wrong class label)
                    errors.append(
                        ErrorPrediction(
                            image_id=image_id,
                            image_path=image_path,
                            ground_truth_class=gt.get("class_name"),
                            predicted_class=pred.get("class_name"),
                            confidence=pred.get("confidence"),
                            iou=best_iou,
                            error_type=ErrorCategory.MISCLASSIFICATION,
                            gt_bbox=gt.get("bbox"),
                            pred_bbox=pred.get("bbox"),
                            sample_link=f"/datasets?image={image_id}",
                        )
                    )
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(p_idx)

            elif 0.1 <= best_iou < self.config.iou_threshold and best_gt_idx >= 0:
                # Poor localization error (overlapping object but sub-threshold IoU)
                gt = ground_truths[best_gt_idx]
                errors.append(
                    ErrorPrediction(
                        image_id=image_id,
                        image_path=image_path,
                        ground_truth_class=gt.get("class_name"),
                        predicted_class=pred.get("class_name"),
                        confidence=pred.get("confidence"),
                        iou=best_iou,
                        error_type=ErrorCategory.POOR_LOCALIZATION,
                        gt_bbox=gt.get("bbox"),
                        pred_bbox=pred.get("bbox"),
                        sample_link=f"/datasets?image={image_id}",
                    )
                )
                matched_pred.add(p_idx)

        # 2. Unmatched Predictions are False Positives
        for p_idx, pred in enumerate(valid_preds):
            if p_idx not in matched_pred:
                errors.append(
                    ErrorPrediction(
                        image_id=image_id,
                        image_path=image_path,
                        predicted_class=pred.get("class_name"),
                        confidence=pred.get("confidence"),
                        iou=0.0,
                        error_type=ErrorCategory.FALSE_POSITIVE,
                        pred_bbox=pred.get("bbox"),
                        sample_link=f"/datasets?image={image_id}",
                    )
                )

        # 3. Unmatched Ground Truths are False Negatives (Missed Objects)
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx not in matched_gt:
                errors.append(
                    ErrorPrediction(
                        image_id=image_id,
                        image_path=image_path,
                        ground_truth_class=gt.get("class_name"),
                        confidence=0.0,
                        iou=0.0,
                        error_type=ErrorCategory.FALSE_NEGATIVE,
                        gt_bbox=gt.get("bbox"),
                        sample_link=f"/datasets?image={image_id}",
                    )
                )

        # 4. Low Confidence Detections (filtered out below threshold)
        low_conf_preds = [
            p for p in predictions if p.get("confidence", 1.0) < self.config.confidence_threshold
        ]
        for pred in low_conf_preds:
            errors.append(
                ErrorPrediction(
                    image_id=image_id,
                    image_path=image_path,
                    predicted_class=pred.get("class_name"),
                    confidence=pred.get("confidence"),
                    iou=None,
                    error_type=ErrorCategory.LOW_CONFIDENCE,
                    pred_bbox=pred.get("bbox"),
                    sample_link=f"/datasets?image={image_id}",
                )
            )

        return errors
