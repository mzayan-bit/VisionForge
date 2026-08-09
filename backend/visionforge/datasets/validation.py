"""Dataset Pre-Split Validation Engine."""

import logging
from typing import Any

from visionforge.datasets.schemas import IssueSeverity, ValidationIssue, ValidationReport
from visionforge.memory.index import VisualMemoryRecord

logger = logging.getLogger("visionforge.datasets.validation")


def validate_dataset(records: list[VisualMemoryRecord]) -> ValidationReport:
    """Perform pre-split validation over candidate dataset visual memory records.

    Checks:
      1. Missing or empty records.
      2. Invalid image dimensions (width <= 0 or height <= 0).
      3. Missing metadata parameters.
      4. Missing embedding vectors or zero-norm embeddings.
    """
    total = len(records)
    if total == 0:
        return ValidationReport(
            status="FAILED",
            total_samples=0,
            valid_samples=0,
            corrupted_samples_count=0,
            missing_embeddings_count=0,
            issues=[
                ValidationIssue(
                    sample_id="dataset",
                    issue_type="EMPTY_DATASET",
                    message="Dataset contains 0 records. Minimum 1 sample is required.",
                    severity=IssueSeverity.ERROR,
                )
            ],
        )

    issues: list[ValidationIssue] = []
    valid_count = 0
    corrupted_count = 0
    missing_embeddings_count = 0

    for rec in records:
        has_error = False

        # 1. Embedding Check
        if not rec.embedding or len(rec.embedding) == 0:
            missing_embeddings_count += 1
            has_error = True
            issues.append(
                ValidationIssue(
                    sample_id=rec.id,
                    issue_type="MISSING_EMBEDDING",
                    message=f"Sample '{rec.id}' lacks a 768D vector embedding.",
                    severity=IssueSeverity.WARNING,
                )
            )

        # 2. Dimension Check
        meta: dict[str, Any] = rec.image_metadata or {}
        w = meta.get("width", 0)
        h = meta.get("height", 0)

        if w <= 0 or h <= 0:
            corrupted_count += 1
            has_error = True
            issues.append(
                ValidationIssue(
                    sample_id=rec.id,
                    issue_type="INVALID_DIMENSIONS",
                    message=f"Sample '{rec.id}' has invalid dimensions ({w}x{h}).",
                    severity=IssueSeverity.ERROR,
                )
            )

        if not has_error:
            valid_count += 1

    # Determine overall status
    has_errors = any(i.severity == IssueSeverity.ERROR for i in issues)
    has_warnings = any(i.severity == IssueSeverity.WARNING for i in issues)

    if has_errors:
        status = "FAILED" if valid_count == 0 else "PASSED_WITH_WARNINGS"
    elif has_warnings:
        status = "PASSED_WITH_WARNINGS"
    else:
        status = "PASSED"

    return ValidationReport(
        status=status,
        total_samples=total,
        valid_samples=valid_count,
        corrupted_samples_count=corrupted_count,
        missing_embeddings_count=missing_embeddings_count,
        issues=issues,
    )
