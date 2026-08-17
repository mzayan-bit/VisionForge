"""Standard Object Detection Evaluation Metrics Engine.

Implements rigorous, mathematically verified evaluation metrics:
- Intersection over Union (IoU)
- Precision, Recall, F1
- Average Precision (AP@50, AP@75, AP@50:95) via COCO 101-point PR interpolation
- Per-class metric decomposition & PR curve points
- Multi-threshold operating point curves (0.10..0.90)
- Multi-class Confusion Matrix with Background false positive / false negative tracking
"""

from typing import Any

import numpy as np

from visionforge.evaluation.schemas import (
    ConfusionMatrixData,
    DetectionMetrics,
    PerClassMetrics,
    PRCurvePoint,
    ThresholdPoint,
)


def calculate_iou_boxes(box_a: list[float], box_b: list[float], format: str = "auto") -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.

    Supported formats:
    - 'xyxy': [x1, y1, x2, y2]
    - 'xywh': [x_center, y_center, width, height]
    - 'auto': automatically detects if boxes are in center-width-height format
    """
    is_xywh = format == "xywh" or (
        format == "auto"
        and (
            box_a[2] < box_a[0]
            or box_b[2] < box_b[0]
            or box_a[3] < box_a[1]
            or box_b[3] < box_b[1]
            or (
                max(box_a) <= 1.0
                and max(box_b) <= 1.0
                and min(box_a[2], box_a[3]) < min(box_a[0], box_a[1])
            )
        )
    )

    if is_xywh:
        x_a1 = box_a[0] - box_a[2] / 2.0
        y_a1 = box_a[1] - box_a[3] / 2.0
        x_a2 = box_a[0] + box_a[2] / 2.0
        y_a2 = box_a[1] + box_a[3] / 2.0

        x_b1 = box_b[0] - box_b[2] / 2.0
        y_b1 = box_b[1] - box_b[3] / 2.0
        x_b2 = box_b[0] + box_b[2] / 2.0
        y_b2 = box_b[1] + box_b[3] / 2.0
    else:  # xyxy
        x_a1, y_a1, x_a2, y_a2 = box_a[0], box_a[1], box_a[2], box_a[3]
        x_b1, y_b1, x_b2, y_b2 = box_b[0], box_b[1], box_b[2], box_b[3]

    inter_x1 = max(x_a1, x_b1)
    inter_y1 = max(y_a1, y_b1)
    inter_x2 = min(x_a2, x_b2)
    inter_y2 = min(y_a2, y_b2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, x_a2 - x_a1) * max(0.0, y_a2 - y_a1)
    area_b = max(0.0, x_b2 - x_b1) * max(0.0, y_b2 - y_b1)

    union_area = area_a + area_b - inter_area
    if union_area <= 1e-9:
        return 0.0

    return float(inter_area / union_area)


def compute_ap_from_pr(
    recalls: list[float], precisions: list[float], num_points: int = 101
) -> float:
    """Calculate Average Precision (AP) via COCO 101-point or VOC 11-point interpolation."""
    if not recalls or not precisions or len(recalls) != len(precisions):
        return 0.0

    # Ensure precision is monotonically decreasing from right to left
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Interpolate at standard points (0.00, 0.01, ..., 1.00)
    recall_thresholds = np.linspace(0.0, 1.0, num_points)
    interpolated_precisions = []

    mrec_arr = np.array(mrec)
    mpre_arr = np.array(mpre)

    for r_thresh in recall_thresholds:
        # Find precision where recall >= r_thresh
        inds = np.where(mrec_arr >= r_thresh)[0]
        if len(inds) > 0:
            interpolated_precisions.append(float(mpre_arr[inds[0]]))
        else:
            interpolated_precisions.append(0.0)

    return float(np.mean(interpolated_precisions))


def evaluate_detections(
    ground_truths_by_image: dict[str, list[dict[str, Any]]],
    predictions_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
) -> tuple[DetectionMetrics, list[PerClassMetrics], list[ThresholdPoint], ConfusionMatrixData]:
    """Execute complete multi-class detection evaluation over all dataset samples.

    ground_truths item schema: {'class_id': int, 'class_name': str, 'bbox': list[float]}
    predictions item schema: {'class_id': int, 'class_name': str, 'confidence': float, 'bbox': list[float]}
    """
    num_classes = len(class_names)
    iou_eval_thresholds = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]

    # Collect total ground truth counts per class
    gt_counts = {cid: 0 for cid in range(num_classes)}
    for gts in ground_truths_by_image.values():
        for gt in gts:
            cid = gt.get("class_id", 0)
            if cid in gt_counts:
                gt_counts[cid] += 1

    per_class_results: list[PerClassMetrics] = []
    ap50_list = []
    ap75_list = []
    map50_95_list = []
    prec_list = []
    rec_list = []
    f1_list = []

    all_tp_ious: list[float] = []

    # 1. Per-Class Precision, Recall, and AP calculation across IoU thresholds
    for class_id in range(num_classes):
        cname = class_names[class_id]
        n_gt = gt_counts[class_id]

        # Gather all predictions for this class across images
        class_preds = []
        for img_id, preds in predictions_by_image.items():
            for p in preds:
                if p.get("class_id") == class_id:
                    class_preds.append((img_id, p["confidence"], p["bbox"]))

        # Sort predictions descending by confidence
        class_preds.sort(key=lambda x: x[1], reverse=True)
        total_pred_count = len(class_preds)

        if n_gt == 0 and total_pred_count == 0:
            per_class_results.append(
                PerClassMetrics(
                    class_id=class_id,
                    class_name=cname,
                    precision=1.0,
                    recall=1.0,
                    f1=1.0,
                    map50=1.0,
                    map75=1.0,
                    map50_95=1.0,
                    support=0,
                    predictions_count=0,
                )
            )
            continue

        # Evaluate AP at each IoU threshold (0.50..0.95)
        ap_per_iou = []
        c_tp_at_default = 0
        c_fp_at_default = 0
        c_pr_curve_points: list[PRCurvePoint] = []

        for iou_th in iou_eval_thresholds:
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            matched_gts: dict[str, set[int]] = {img_id: set() for img_id in ground_truths_by_image}

            for p_idx, (img_id, conf, pred_bbox) in enumerate(class_preds):
                gts = ground_truths_by_image.get(img_id, [])
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gts):
                    if gt.get("class_id") != class_id:
                        continue
                    if gt_idx in matched_gts[img_id]:
                        continue

                    iou = calculate_iou_boxes(pred_bbox, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_th and best_gt_idx >= 0:
                    tp[p_idx] = 1.0
                    matched_gts[img_id].add(best_gt_idx)
                    if abs(iou_th - iou_threshold) < 1e-4 and conf >= confidence_threshold:
                        all_tp_ious.append(best_iou)
                else:
                    fp[p_idx] = 1.0

            # Cumulative TP / FP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)

            recalls = (cum_tp / n_gt) if n_gt > 0 else np.zeros_like(cum_tp)
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

            ap = compute_ap_from_pr(list(recalls), list(precisions))
            ap_per_iou.append(ap)

            # Record default IoU curve points and default TP/FP counts
            if abs(iou_th - iou_threshold) < 1e-4:
                # Filter by confidence threshold for default operating point
                valid_indices = [
                    i for i, cp in enumerate(class_preds) if cp[1] >= confidence_threshold
                ]
                c_tp_at_default = int(np.sum(tp[valid_indices])) if valid_indices else 0
                c_fp_at_default = int(np.sum(fp[valid_indices])) if valid_indices else 0

                # Sample up to 20 representative PR curve coordinates for visualization
                if len(recalls) > 0:
                    step = max(1, len(recalls) // 20)
                    for idx in range(0, len(recalls), step):
                        c_pr_curve_points.append(
                            PRCurvePoint(
                                recall=round(float(recalls[idx]), 4),
                                precision=round(float(precisions[idx]), 4),
                            )
                        )

        ap50 = ap_per_iou[0] if len(ap_per_iou) > 0 else 0.0
        # AP@75 is index 5 in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, ...]
        ap75 = ap_per_iou[5] if len(ap_per_iou) > 5 else ap50
        map50_95 = float(np.mean(ap_per_iou)) if ap_per_iou else 0.0

        c_fn_at_default = max(0, n_gt - c_tp_at_default)
        c_prec = c_tp_at_default / (c_tp_at_default + c_fp_at_default + 1e-9)
        c_rec = c_tp_at_default / (n_gt + 1e-9)
        c_f1 = 2 * (c_prec * c_rec) / (c_prec + c_rec + 1e-9)

        ap50_list.append(ap50)
        ap75_list.append(ap75)
        map50_95_list.append(map50_95)
        prec_list.append(c_prec)
        rec_list.append(c_rec)
        f1_list.append(c_f1)

        per_class_results.append(
            PerClassMetrics(
                class_id=class_id,
                class_name=cname,
                precision=round(float(c_prec), 4),
                recall=round(float(c_rec), 4),
                f1=round(float(c_f1), 4),
                map50=round(float(ap50), 4),
                map75=round(float(ap75), 4),
                map50_95=round(float(map50_95), 4),
                support=n_gt,
                predictions_count=total_pred_count,
                true_positives=c_tp_at_default,
                false_positives=c_fp_at_default,
                false_negatives=c_fn_at_default,
                pr_curve_points=c_pr_curve_points,
            )
        )

    # 2. Multi-threshold Operating Point Analysis (0.10..0.90)
    threshold_points: list[ThresholdPoint] = []
    for conf_th in np.arange(0.10, 0.95, 0.10):
        tot_tp = 0
        tot_fp = 0
        tot_gt = sum(gt_counts.values())

        for img_id, preds in predictions_by_image.items():
            gts = ground_truths_by_image.get(img_id, [])
            valid_preds = [p for p in preds if p["confidence"] >= conf_th]
            matched_gts_set: set[int] = set()

            for pred in sorted(valid_preds, key=lambda x: x["confidence"], reverse=True):
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in matched_gts_set or pred["class_id"] != gt["class_id"]:
                        continue
                    iou = calculate_iou_boxes(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tot_tp += 1
                    matched_gts_set.add(best_gt_idx)
                else:
                    tot_fp += 1

        tot_fn = max(0, tot_gt - tot_tp)
        th_p = tot_tp / (tot_tp + tot_fp + 1e-9)
        th_r = tot_tp / (tot_gt + 1e-9)
        th_f1 = 2 * (th_p * th_r) / (th_p + th_r + 1e-9)

        threshold_points.append(
            ThresholdPoint(
                confidence_threshold=round(float(conf_th), 2),
                precision=round(float(th_p), 4),
                recall=round(float(th_r), 4),
                f1=round(float(th_f1), 4),
                true_positives=tot_tp,
                false_positives=tot_fp,
                false_negatives=tot_fn,
            )
        )

    # 3. Multi-class Confusion Matrix Generation
    # Matrix dimensions: (num_classes + 1) x (num_classes + 1) where index num_classes is 'background'
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    total_eval_boxes = 0

    for img_id, preds in predictions_by_image.items():
        gts = ground_truths_by_image.get(img_id, [])
        valid_preds = [p for p in preds if p["confidence"] >= confidence_threshold]
        matched_gt_indices: set[int] = set()
        matched_pred_indices: set[int] = set()

        for p_idx, pred in enumerate(valid_preds):
            p_cid = min(pred["class_id"], num_classes - 1)
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt_indices:
                    continue
                iou = calculate_iou_boxes(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_cid = min(gts[best_gt_idx]["class_id"], num_classes - 1)
                conf_matrix[gt_cid, p_cid] += 1
                matched_gt_indices.add(best_gt_idx)
                matched_pred_indices.add(p_idx)
                total_eval_boxes += 1
            else:
                # Background False Positive (GT = background, Pred = p_cid)
                conf_matrix[num_classes, p_cid] += 1
                total_eval_boxes += 1

        # Unmatched GTs are False Negatives (GT = gt_cid, Pred = background)
        for gt_idx, gt in enumerate(gts):
            if gt_idx not in matched_gt_indices:
                gt_cid = min(gt["class_id"], num_classes - 1)
                conf_matrix[gt_cid, num_classes] += 1
                total_eval_boxes += 1

    confusion_data = ConfusionMatrixData(
        class_names=list(class_names) + ["background"],
        matrix=[[int(val) for val in row] for row in conf_matrix],
        total_samples=total_eval_boxes,
    )

    # 4. Overall Aggregate Detection Metrics
    agg_prec = float(np.mean(prec_list)) if prec_list else 0.0
    agg_rec = float(np.mean(rec_list)) if rec_list else 0.0
    agg_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    agg_map50 = float(np.mean(ap50_list)) if ap50_list else 0.0
    agg_map75 = float(np.mean(ap75_list)) if ap75_list else 0.0
    agg_map50_95 = float(np.mean(map50_95_list)) if map50_95_list else 0.0
    mean_iou = float(np.mean(all_tp_ious)) if all_tp_ious else 0.0
    total_preds_all = sum(len(p) for p in predictions_by_image.values())

    metrics = DetectionMetrics(
        precision=round(agg_prec, 4),
        recall=round(agg_rec, 4),
        f1=round(agg_f1, 4),
        mean_iou=round(mean_iou, 4),
        map50=round(agg_map50, 4),
        map75=round(agg_map75, 4),
        map50_95=round(agg_map50_95, 4),
        support_gt_count=sum(gt_counts.values()),
        total_predictions=total_preds_all,
    )

    return metrics, per_class_results, threshold_points, confusion_data
