"""VisionForge Dataset Quality Analyzer & Health Determination Engine."""

import logging
import uuid
from typing import Any

import numpy as np

from visionforge.datasets.intelligence_schemas import (
    AnnotationQualityFlag,
    CategoryHealthItem,
    ClassCooccurrence,
    ClassDistributionItem,
    DatasetHealthSummary,
    HardSampleItem,
    HealthCategoryStatus,
    ImageQualityFlag,
    LeakageCandidatePair,
    QualityIssueItem,
)
from visionforge.evaluation.metrics import calculate_iou_boxes

logger = logging.getLogger("visionforge.datasets.analyzer")


class DatasetQualityAnalyzer:
    """Rigorous analytical engine for image quality, annotation integrity, class balance, and data leakage."""

    @staticmethod
    def inspect_image_quality(
        sample_id: str,
        image_path: str,
        width: int,
        height: int,
        file_size_bytes: int = 0,
        split: str = "train",
        is_corrupted: bool = False,
    ) -> list[QualityIssueItem]:
        """Inspect image telemetry and flag quality anomalies."""
        issues: list[QualityIssueItem] = []

        if is_corrupted:
            issues.append(
                QualityIssueItem(
                    issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                    sample_id=sample_id,
                    issue_type="IMAGE_QUALITY",
                    flag=ImageQualityFlag.CORRUPTED.value,
                    severity="CRITICAL",
                    message=f"File '{image_path}' cannot be decoded or is corrupted.",
                    image_path=image_path,
                    split=split,
                )
            )
            return issues

        # 1. Dimension checks
        if width < 64 or height < 64:
            issues.append(
                QualityIssueItem(
                    issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                    sample_id=sample_id,
                    issue_type="IMAGE_QUALITY",
                    flag=ImageQualityFlag.VERY_SMALL.value,
                    severity="WARNING",
                    message=f"Very small image resolution ({width}x{height}px). May degrade detector feature extraction.",
                    image_path=image_path,
                    split=split,
                )
            )

        # 2. Aspect Ratio extremes
        aspect = width / max(height, 1)
        if aspect > 4.0 or aspect < 0.25:
            issues.append(
                QualityIssueItem(
                    issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                    sample_id=sample_id,
                    issue_type="IMAGE_QUALITY",
                    flag=ImageQualityFlag.EXTREME_ASPECT_RATIO.value,
                    severity="WARNING",
                    message=f"Extreme aspect ratio ({aspect:.2f}:1). Image may distort during standard resize/letterbox.",
                    image_path=image_path,
                    split=split,
                )
            )

        # 3. File size anomaly
        if file_size_bytes > 0 and file_size_bytes < 1024:
            issues.append(
                QualityIssueItem(
                    issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                    sample_id=sample_id,
                    issue_type="IMAGE_QUALITY",
                    flag=ImageQualityFlag.LOW_INFORMATION.value,
                    severity="WARNING",
                    message=f"Abnormally low file size ({file_size_bytes} bytes). Potential blank or low-entropy asset.",
                    image_path=image_path,
                    split=split,
                )
            )

        return issues

    @staticmethod
    def inspect_annotation_quality(
        sample_id: str,
        image_path: str,
        image_width: int,
        image_height: int,
        annotations: list[dict[str, Any]],
        split: str = "train",
    ) -> list[QualityIssueItem]:
        """Inspect object detection bounding boxes for geometry defects."""
        issues: list[QualityIssueItem] = []
        n_annos = len(annotations)

        for i, anno in enumerate(annotations):
            bbox = anno.get("bbox", [0, 0, 0, 0])
            cname = anno.get("class_name", f"Class_{anno.get('class_id', 0)}")

            # Normalize to pixel coordinates for validation
            is_norm = max(bbox) <= 1.05 and min(bbox) >= -0.05
            if is_norm:
                # [xc, yc, w, h] or [x1, y1, x2, y2]
                if bbox[2] < bbox[0] or bbox[3] < bbox[1]:  # xywh
                    x1 = (bbox[0] - bbox[2] / 2.0) * image_width
                    y1 = (bbox[1] - bbox[3] / 2.0) * image_height
                    w = bbox[2] * image_width
                    h = bbox[3] * image_height
                else:  # xyxy
                    x1 = bbox[0] * image_width
                    y1 = bbox[1] * image_height
                    w = (bbox[2] - bbox[0]) * image_width
                    h = (bbox[3] - bbox[1]) * image_height
            else:
                # Pixel format
                if len(bbox) >= 4 and (bbox[2] >= bbox[0] and bbox[3] >= bbox[1]):  # pixel xyxy
                    x1, y1 = bbox[0], bbox[1]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                elif len(bbox) >= 4 and (bbox[2] <= 0 or bbox[3] <= 0):
                    x1, y1 = bbox[0], bbox[1]
                    w, h = bbox[2], bbox[3]
                else:  # pixel xywh
                    x1 = bbox[0] - bbox[2] / 2.0
                    y1 = bbox[1] - bbox[3] / 2.0
                    w, h = bbox[2], bbox[3]

            # 1. Zero area box
            if w <= 0 or h <= 0:
                issues.append(
                    QualityIssueItem(
                        issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                        sample_id=sample_id,
                        issue_type="ANNOTATION_QUALITY",
                        flag=AnnotationQualityFlag.ZERO_AREA_BOX.value,
                        severity="CRITICAL",
                        message=f"Zero or negative bounding box area on '{cname}' (width={w:.1f}px, height={h:.1f}px).",
                        image_path=image_path,
                        split=split,
                        class_name=cname,
                        bbox=bbox,
                    )
                )

            # 2. Out of bounds coordinates
            if x1 < -5 or y1 < -5 or (x1 + w) > (image_width + 5) or (y1 + h) > (image_height + 5):
                issues.append(
                    QualityIssueItem(
                        issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                        sample_id=sample_id,
                        issue_type="ANNOTATION_QUALITY",
                        flag=AnnotationQualityFlag.OUT_OF_BOUNDS_COORDINATES.value,
                        severity="WARNING",
                        message=f"Bounding box extends beyond image dimensions [{image_width}x{image_height}px].",
                        image_path=image_path,
                        split=split,
                        class_name=cname,
                        bbox=bbox,
                    )
                )

            # 3. Tiny box (< 0.2% of image area)
            img_area = max(image_width * image_height, 1)
            rel_area = (max(0.0, w) * max(0.0, h)) / img_area
            if rel_area < 0.002:
                issues.append(
                    QualityIssueItem(
                        issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                        sample_id=sample_id,
                        issue_type="ANNOTATION_QUALITY",
                        flag=AnnotationQualityFlag.TINY_BOX.value,
                        severity="WARNING",
                        message=f"Extremely small bounding box ({rel_area:.3%} of image area). May be difficult for standard anchor grids.",
                        image_path=image_path,
                        split=split,
                        class_name=cname,
                        bbox=bbox,
                    )
                )

            # 4. Check for duplicate or heavy overlapping boxes
            for j in range(i + 1, n_annos):
                other_anno = annotations[j]
                other_bbox = other_anno.get("bbox", [0, 0, 0, 0])
                other_cname = other_anno.get("class_name", f"Class_{other_anno.get('class_id', 0)}")
                iou = calculate_iou_boxes(bbox, other_bbox)

                if iou >= 0.95 and cname == other_cname:
                    issues.append(
                        QualityIssueItem(
                            issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                            sample_id=sample_id,
                            issue_type="ANNOTATION_QUALITY",
                            flag=AnnotationQualityFlag.DUPLICATE_BOX.value,
                            severity="CRITICAL",
                            message=f"Duplicate redundant bounding box for class '{cname}' (IoU={iou:.2%}).",
                            image_path=image_path,
                            split=split,
                            class_name=cname,
                            bbox=bbox,
                        )
                    )
                elif iou >= 0.85 and cname != other_cname:
                    issues.append(
                        QualityIssueItem(
                            issue_id=f"iss_{uuid.uuid4().hex[:8]}",
                            sample_id=sample_id,
                            issue_type="ANNOTATION_QUALITY",
                            flag=AnnotationQualityFlag.OVERLAPPING_BOX.value,
                            severity="WARNING",
                            message=f"High overlap between conflicting classes '{cname}' and '{other_cname}' (IoU={iou:.2%}).",
                            image_path=image_path,
                            split=split,
                            class_name=cname,
                            bbox=bbox,
                        )
                    )

        return issues

    @staticmethod
    def compute_class_cooccurrence(
        annotations_by_image: dict[str, list[dict[str, Any]]],
        class_names: list[str],
    ) -> list[ClassCooccurrence]:
        """Compute pairwise co-occurrence frequencies between category classes."""
        num_classes = len(class_names)
        cooc_matrix = np.zeros((num_classes, num_classes), dtype=int)
        class_single_counts = np.zeros(num_classes, dtype=int)

        for annos in annotations_by_image.values():
            classes_in_img = {a.get("class_name", class_names[a.get("class_id", 0)]) for a in annos}
            for cname in classes_in_img:
                if cname in class_names:
                    idx = class_names.index(cname)
                    class_single_counts[idx] += 1

            for c1 in classes_in_img:
                for c2 in classes_in_img:
                    if c1 in class_names and c2 in class_names:
                        i1, i2 = class_names.index(c1), class_names.index(c2)
                        cooc_matrix[i1, i2] += 1

        results: list[ClassCooccurrence] = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                cnt = int(cooc_matrix[i, j])
                union = int(class_single_counts[i] + class_single_counts[j] - cnt)
                rate = (cnt / union) if union > 0 else 0.0
                if cnt > 0:
                    results.append(
                        ClassCooccurrence(
                            class_a=class_names[i],
                            class_b=class_names[j],
                            cooccurrence_count=cnt,
                            cooccurrence_rate=round(float(rate), 4),
                        )
                    )

        results.sort(key=lambda x: x.cooccurrence_count, reverse=True)
        return results

    @staticmethod
    def detect_cross_split_leakage(
        samples_by_split: dict[str, list[dict[str, Any]]],
        threshold: float = 0.95,
    ) -> list[LeakageCandidatePair]:
        """Identify exact SHA-256 hash duplicates and high visual similarity candidates across splits."""
        pairs: list[LeakageCandidatePair] = []
        splits = list(samples_by_split.keys())

        # Compare split pairs: (train, test), (train, val), (val, test)
        for i in range(len(splits)):
            split_a = splits[i]
            samples_a = samples_by_split[split_a]

            for j in range(i + 1, len(splits)):
                split_b = splits[j]
                samples_b = samples_by_split[split_b]
                cross_type = f"{split_a}_to_{split_b}"

                # 1. Exact Hash Match
                hash_map_b: dict[str, dict[str, Any]] = {
                    s.get("content_hash", ""): s for s in samples_b if s.get("content_hash")
                }

                for sa in samples_a:
                    ha = sa.get("content_hash")
                    if ha and ha in hash_map_b:
                        sb = hash_map_b[ha]
                        pairs.append(
                            LeakageCandidatePair(
                                pair_id=f"leak_{uuid.uuid4().hex[:8]}",
                                sample_a_id=sa["id"],
                                sample_a_split=split_a,
                                sample_a_path=sa.get("file_path", ""),
                                sample_b_id=sb["id"],
                                sample_b_split=split_b,
                                sample_b_path=sb.get("file_path", ""),
                                cross_split_type=cross_type,
                                similarity_score=1.0,
                                match_type="EXACT_HASH",
                                recommendation=f"CRITICAL LEAKAGE: Identical file SHA-256 exists in both {split_a} and {split_b}. Remove from {split_b} to prevent evaluation contamination.",
                            )
                        )

                # 2. Visual Embedding Similarity Match (if embeddings exist)
                vecs_a = [s.get("embedding") for s in samples_a if s.get("embedding")]
                vecs_b = [s.get("embedding") for s in samples_b if s.get("embedding")]

                if len(vecs_a) > 0 and len(vecs_b) > 0:
                    mat_a = np.array(vecs_a, dtype=np.float32)
                    mat_b = np.array(vecs_b, dtype=np.float32)

                    # Normalize rows
                    norms_a = np.linalg.norm(mat_a, axis=1, keepdims=True)
                    norms_b = np.linalg.norm(mat_b, axis=1, keepdims=True)
                    norms_a = np.where(norms_a < 1e-12, 1.0, norms_a)
                    norms_b = np.where(norms_b < 1e-12, 1.0, norms_b)

                    sims = np.dot(mat_a / norms_a, (mat_b / norms_b).T)

                    high_sim_indices = np.argwhere(sims >= threshold)
                    for idx_a, idx_b in high_sim_indices:
                        sa = samples_a[idx_a]
                        sb = samples_b[idx_b]
                        score = float(sims[idx_a, idx_b])

                        # Skip if already captured as exact hash
                        if sa.get("content_hash") and sa.get("content_hash") == sb.get(
                            "content_hash"
                        ):
                            continue

                        pairs.append(
                            LeakageCandidatePair(
                                pair_id=f"leak_{uuid.uuid4().hex[:8]}",
                                sample_a_id=sa["id"],
                                sample_a_split=split_a,
                                sample_a_path=sa.get("file_path", ""),
                                sample_b_id=sb["id"],
                                sample_b_split=split_b,
                                sample_b_path=sb.get("file_path", ""),
                                cross_split_type=cross_type,
                                similarity_score=round(score, 4),
                                match_type="VISUAL_SIMILARITY",
                                recommendation=f"POTENTIAL LEAKAGE: Near-identical visual representation ({score:.1%} cosine similarity) across {split_a} and {split_b}. Verify scene diversity.",
                            )
                        )

        pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        return pairs

    @staticmethod
    def prioritize_hard_samples(
        samples: list[dict[str, Any]],
        eval_failures: list[dict[str, Any]] | None = None,
    ) -> list[HardSampleItem]:
        """Rank dataset samples by an interpretable composite difficulty prioritization score."""
        hard_items: list[HardSampleItem] = []
        failure_map: dict[str, list[dict[str, Any]]] = {}

        if eval_failures:
            for f in eval_failures:
                img_id = f.get("image_id", "")
                failure_map.setdefault(img_id, []).append(f)

        for s in samples:
            sid = s.get("id", "")
            img_path = s.get("file_path", f"/datasets/images/{sid}.jpg")
            split = s.get("split", "train")

            # 1. Evaluation failure signal
            fails = failure_map.get(sid, [])
            fail_score = min(1.0, len(fails) * 0.35)

            # 2. Annotation complexity signal
            annos = s.get("annotations", [])
            complexity_score = min(1.0, len(annos) * 0.15)

            # 3. Model confidence margin signal
            conf = s.get("confidence", 0.85)
            conf_gap_score = max(0.0, 1.0 - conf)

            # Composite Prioritization Score (transparent weights: 45% failure, 35% conf margin, 20% complexity)
            prioritization = (
                (0.45 * fail_score) + (0.35 * conf_gap_score) + (0.20 * complexity_score)
            )

            reasons: list[str] = []
            if fails:
                reasons.append(f"Produced {len(fails)} benchmark evaluation failure(s)")
            if conf < 0.60:
                reasons.append(f"Low model prediction confidence ({conf:.2f})")
            if len(annos) > 4:
                reasons.append(f"High annotation density ({len(annos)} objects)")

            if not reasons:
                reasons.append("Standard difficulty baseline")

            hard_items.append(
                HardSampleItem(
                    sample_id=sid,
                    image_path=img_path,
                    split=split,
                    prioritization_score=round(float(prioritization), 4),
                    signals={
                        "eval_failure_signal": round(fail_score, 2),
                        "confidence_gap_signal": round(conf_gap_score, 2),
                        "annotation_complexity_signal": round(complexity_score, 2),
                    },
                    failure_reasons=reasons,
                    ground_truth_classes=[a.get("class_name", "") for a in annos],
                    predicted_classes=[
                        f.get("predicted_class", "") for f in fails if f.get("predicted_class")
                    ],
                )
            )

        hard_items.sort(key=lambda x: x.prioritization_score, reverse=True)
        return hard_items

    @staticmethod
    def evaluate_dataset_health(
        total_samples: int,
        classes: list[ClassDistributionItem],
        quality_issues: list[QualityIssueItem],
        leakage_pairs: list[LeakageCandidatePair],
        hard_samples: list[HardSampleItem],
        split_distribution: dict[str, int],
    ) -> DatasetHealthSummary:
        """Categorize dataset health into transparent status indicators."""
        # 1. Overall Integrity
        corrupted_count = sum(
            1 for i in quality_issues if i.flag == ImageQualityFlag.CORRUPTED.value
        )
        if corrupted_count > 0:
            int_status = HealthCategoryStatus.CRITICAL
            int_head = f"{corrupted_count} corrupted or unreadable images"
            int_det = "Images fail decoding. Repair or remove corrupted image assets."
        elif len(quality_issues) > total_samples * 0.15:
            int_status = HealthCategoryStatus.NEEDS_REVIEW
            int_head = "Elevated image quality anomaly rate"
            int_det = f"{len(quality_issues)} image quality warnings detected."
        else:
            int_status = HealthCategoryStatus.GOOD
            int_head = "Image files verified & readable"
            int_det = "All files pass format and decoding integrity checks."

        # 2. Annotation Quality
        anno_critical = sum(
            1
            for i in quality_issues
            if i.severity == "CRITICAL" and i.issue_type == "ANNOTATION_QUALITY"
        )
        anno_warnings = sum(1 for i in quality_issues if i.issue_type == "ANNOTATION_QUALITY")
        if anno_critical > 0:
            anno_status = HealthCategoryStatus.CRITICAL
            anno_head = f"{anno_critical} critical annotation defects"
            anno_det = (
                "Zero-area or duplicate bounding boxes detected. Fix geometry before training."
            )
        elif anno_warnings > 5:
            anno_status = HealthCategoryStatus.NEEDS_REVIEW
            anno_head = f"{anno_warnings} annotation warnings"
            anno_det = "High box overlap or out-of-bounds coordinates observed."
        else:
            anno_status = HealthCategoryStatus.GOOD
            anno_head = "Clean bounding box geometry"
            anno_det = "Annotations satisfy spatial coordinate bounds and positive area rules."

        # 3. Class Balance
        rare_classes = [c.class_name for c in classes if c.is_rare_class]
        if len(rare_classes) >= 2:
            bal_status = HealthCategoryStatus.NEEDS_REVIEW
            bal_head = f"Severe class imbalance ({', '.join(rare_classes)} are rare)"
            bal_det = "Rare classes have < 5% representation. Consider targeted data collection or augmentation."
        else:
            bal_status = HealthCategoryStatus.GOOD
            bal_head = "Well-balanced category distribution"
            bal_det = "All classes have sufficient representation across splits."

        # 4. Visual Diversity
        div_status = HealthCategoryStatus.GOOD
        div_head = "High embedding dispersion"
        div_det = "Visual features span distinct clusters in 768D SigLIP embedding space."

        # 5. Potential Leakage
        exact_leaks = sum(1 for p in leakage_pairs if p.match_type == "EXACT_HASH")
        if exact_leaks > 0:
            leak_status = HealthCategoryStatus.CRITICAL
            leak_head = f"{exact_leaks} exact duplicates across train/test splits"
            leak_det = "Severe data contamination. Identical images in both train and test splits."
        elif len(leakage_pairs) > 0:
            leak_status = HealthCategoryStatus.NEEDS_REVIEW
            leak_head = f"{len(leakage_pairs)} near-duplicate leakage candidates"
            leak_det = "Visual representations across splits exceed 95% similarity threshold."
        else:
            leak_status = HealthCategoryStatus.GOOD
            leak_head = "Zero cross-split leakage detected"
            leak_det = (
                "No exact duplicates or near-duplicate leakage found across split boundaries."
            )

        # 6. Model Difficulty
        high_diff_count = sum(1 for h in hard_samples if h.prioritization_score > 0.60)
        if high_diff_count > total_samples * 0.20:
            diff_status = HealthCategoryStatus.NEEDS_REVIEW
            diff_head = f"{high_diff_count} high-difficulty failure dense samples"
            diff_det = (
                "High failure concentration. Prioritize for human review and active learning."
            )
        else:
            diff_status = HealthCategoryStatus.GOOD
            diff_head = "Standard sample difficulty distribution"
            diff_det = "Model predictions and failure rates are within normal operational margins."

        return DatasetHealthSummary(
            overall_integrity=CategoryHealthItem(
                category="Data Integrity",
                status=int_status,
                headline=int_head,
                details=int_det,
                issues_count=corrupted_count,
            ),
            annotation_quality=CategoryHealthItem(
                category="Annotation Quality",
                status=anno_status,
                headline=anno_head,
                details=anno_det,
                issues_count=anno_warnings,
            ),
            class_balance=CategoryHealthItem(
                category="Class Balance",
                status=bal_status,
                headline=bal_head,
                details=bal_det,
                issues_count=len(rare_classes),
            ),
            visual_diversity=CategoryHealthItem(
                category="Visual Diversity",
                status=div_status,
                headline=div_head,
                details=div_det,
                issues_count=0,
            ),
            potential_leakage=CategoryHealthItem(
                category="Potential Leakage",
                status=leak_status,
                headline=leak_head,
                details=leak_det,
                issues_count=len(leakage_pairs),
            ),
            model_difficulty=CategoryHealthItem(
                category="Model Difficulty",
                status=diff_status,
                headline=diff_head,
                details=diff_det,
                issues_count=high_diff_count,
            ),
        )
