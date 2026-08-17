"""VisionForge Data-Centric Computer Vision Workspace Schemas.

Comprehensive data models for:
- Dataset Profiles & Deep Statistics
- Dataset Health Determination across 6 categories
- Image & Bounding Box Quality Taxonomy
- Exact & Near-Duplicate Detection and Cross-Split Leakage
- Hard Sample Prioritization Scoring
- Human Review Queue Curation Decisions
- Dataset Versioning Snapshots and Granular Dataset Diffing
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class HealthCategoryStatus(StrEnum):
    """Health determination status for a dataset dimension."""

    GOOD = "GOOD"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    CRITICAL = "CRITICAL"


class ImageQualityFlag(StrEnum):
    """Diagnostic flags for image asset defects."""

    VERY_SMALL = "VERY_SMALL"
    EXTREME_ASPECT_RATIO = "EXTREME_ASPECT_RATIO"
    BLANK_IMAGE = "BLANK_IMAGE"
    VERY_DARK = "VERY_DARK"
    VERY_BRIGHT = "VERY_BRIGHT"
    LOW_INFORMATION = "LOW_INFORMATION"
    CORRUPTED = "CORRUPTED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"


class AnnotationQualityFlag(StrEnum):
    """Diagnostic flags for bounding box annotation defects."""

    ZERO_AREA_BOX = "ZERO_AREA_BOX"
    OUT_OF_BOUNDS_COORDINATES = "OUT_OF_BOUNDS_COORDINATES"
    EXTREME_BBOX_ASPECT_RATIO = "EXTREME_BBOX_ASPECT_RATIO"
    TINY_BOX = "TINY_BOX"
    OVERLAPPING_BOX = "OVERLAPPING_BOX"
    DUPLICATE_BOX = "DUPLICATE_BOX"


class CategoryHealthItem(BaseModel):
    """Structured health assessment for a specific dataset dimension."""

    category: str = Field(description="Health dimension category name")
    status: HealthCategoryStatus = Field(description="Health status (GOOD, NEEDS_REVIEW, CRITICAL)")
    headline: str = Field(description="Concise summary headline")
    details: str = Field(description="Transparent diagnostic explanation")
    issues_count: int = Field(default=0, description="Total flagged issues in this category")


class DatasetHealthSummary(BaseModel):
    """Transparent multi-dimensional dataset health scorecard."""

    overall_integrity: CategoryHealthItem = Field(
        description="File readability and format compliance"
    )
    annotation_quality: CategoryHealthItem = Field(
        description="Bounding box and label geometry integrity"
    )
    class_balance: CategoryHealthItem = Field(
        description="Category distribution parity and rare classes"
    )
    visual_diversity: CategoryHealthItem = Field(
        description="Embedding space dispersion and coverage"
    )
    potential_leakage: CategoryHealthItem = Field(
        description="Cross-split duplication and contamination risk"
    )
    model_difficulty: CategoryHealthItem = Field(
        description="Hard sample count and failure density"
    )


class ClassDistributionItem(BaseModel):
    """Detailed category class representation statistics."""

    class_id: int = Field(description="Zero-indexed class ID")
    class_name: str = Field(description="Human-readable class name")
    sample_count: int = Field(description="Number of images containing this class")
    sample_percentage: float = Field(
        description="Percentage of dataset images containing this class"
    )
    annotation_count: int = Field(description="Total bounding box annotations of this class")
    avg_annotations_per_image: float = Field(
        description="Average instances per image where present"
    )
    is_rare_class: bool = Field(
        default=False, description="Flagged if under-represented (< 5% of total)"
    )
    is_dominant_class: bool = Field(
        default=False, description="Flagged if dominant (> 40% of total)"
    )
    split_counts: dict[str, int] = Field(
        default_factory=dict, description="Annotation counts per split partition"
    )


class ImageStatistics(BaseModel):
    """Comprehensive image resolution and file telemetry."""

    min_width: int = Field(description="Minimum image width in pixels")
    max_width: int = Field(description="Maximum image width in pixels")
    mean_width: float = Field(description="Mean image width in pixels")
    min_height: int = Field(description="Minimum image height in pixels")
    max_height: int = Field(description="Maximum image height in pixels")
    mean_height: float = Field(description="Mean image height in pixels")
    mean_aspect_ratio: float = Field(description="Mean width / height aspect ratio")
    format_distribution: dict[str, int] = Field(
        default_factory=dict, description="Image file extensions breakdown"
    )
    resolution_bins: dict[str, int] = Field(
        default_factory=dict, description="Counts by resolution tier"
    )
    total_size_bytes: int = Field(default=0, description="Total disk footprint of dataset images")


class AnnotationStatistics(BaseModel):
    """Aggregate bounding box geometry statistics."""

    total_boxes: int = Field(description="Total object bounding boxes")
    mean_boxes_per_image: float = Field(description="Average objects per image")
    max_boxes_per_image: int = Field(description="Maximum objects in a single image")
    mean_box_relative_area: float = Field(
        description="Mean relative box area (box_area / image_area)"
    )
    size_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Box size tiers: 'tiny' (<0.5%), 'small' (0.5-2%), 'medium' (2-15%), 'large' (>15%)",
    )


class ClassCooccurrence(BaseModel):
    """Frequency of two classes appearing together in the same image."""

    class_a: str = Field(description="First class name")
    class_b: str = Field(description="Second class name")
    cooccurrence_count: int = Field(description="Number of images containing both classes")
    cooccurrence_rate: float = Field(description="Co-occurrence percentage relative to union")


class QualityIssueItem(BaseModel):
    """Individual flagged diagnostic defect in an image or annotation."""

    issue_id: str = Field(description="Unique issue identifier")
    sample_id: str = Field(description="Associated image sample identifier")
    issue_type: str = Field(
        description="'IMAGE_QUALITY', 'ANNOTATION_QUALITY', 'DUPLICATE', 'LEAKAGE', 'OUTLIER'"
    )
    flag: str = Field(description="Diagnostic flag name")
    severity: str = Field(default="WARNING", description="'WARNING' or 'CRITICAL'")
    message: str = Field(description="Transparent explanation of the issue")
    image_path: str = Field(description="Path to image asset")
    split: str = Field(default="train", description="Dataset split containing this sample")
    class_name: str | None = Field(default=None, description="Class name if annotation defect")
    bbox: list[float] | None = Field(default=None, description="Bounding box if annotation defect")
    review_status: str = Field(
        default="UNREVIEWED", description="'UNREVIEWED', 'ACCEPTED', 'REJECTED', 'CORRECTED'"
    )
    detected_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class LeakageCandidatePair(BaseModel):
    """Detected cross-split near-duplicate or exact duplicate pair."""

    pair_id: str = Field(description="Unique leakage pair ID")
    sample_a_id: str = Field(description="First sample identifier")
    sample_a_split: str = Field(description="First sample split (e.g. 'train')")
    sample_a_path: str = Field(description="First sample path")
    sample_b_id: str = Field(description="Second sample identifier")
    sample_b_split: str = Field(description="Second sample split (e.g. 'test')")
    sample_b_path: str = Field(description="Second sample path")
    cross_split_type: str = Field(description="'train_to_test', 'train_to_val', 'val_to_test'")
    similarity_score: float = Field(
        description="Cosine similarity score or 1.0 for exact hash match"
    )
    match_type: str = Field(description="'EXACT_HASH' or 'VISUAL_SIMILARITY'")
    recommendation: str = Field(description="Actionable curation guidance for researcher")


class HardSampleItem(BaseModel):
    """Sample prioritized for review due to model difficulty or ambiguity."""

    sample_id: str = Field(description="Sample identifier")
    image_path: str = Field(description="Path to image asset")
    split: str = Field(description="Dataset split partition")
    prioritization_score: float = Field(
        description="Interpretable composite difficulty score [0.0, 1.0]"
    )
    signals: dict[str, float] = Field(
        default_factory=dict,
        description="Component signal scores (eval_failure, confidence_gap, isolation)",
    )
    failure_reasons: list[str] = Field(
        default_factory=list, description="Descriptive list of failure causes"
    )
    ground_truth_classes: list[str] = Field(
        default_factory=list, description="Ground truth classes present"
    )
    predicted_classes: list[str] = Field(
        default_factory=list, description="Model predicted classes"
    )


class CurationDecision(BaseModel):
    """Reviewer decision record submitted to Human Review Queue."""

    review_id: str = Field(description="Unique review record ID")
    sample_id: str = Field(description="Reviewed sample identifier")
    issue_id: str | None = Field(default=None, description="Optional associated issue ID")
    decision: str = Field(
        description="'ACCEPT', 'REJECT', 'NEEDS_CORRECTION', 'NOT_A_PROBLEM', 'DUPLICATE', 'INVALID', 'UNCERTAIN'"
    )
    category: str = Field(
        default="annotation_review",
        description="'duplicate_review', 'leakage_review', 'annotation_review', 'outlier_review', 'hard_sample_review'",
    )
    notes: str = Field(default="", description="Reviewer notes or correction guidance")
    reviewer: str = Field(default="Researcher", description="User identity who made the decision")
    decided_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DatasetProfile(BaseModel):
    """Comprehensive, machine-readable dataset profile snapshot."""

    dataset_id: str = Field(description="Dataset identifier")
    dataset_version: str = Field(description="Dataset version identifier")
    dataset_fingerprint: str = Field(description="Cryptographic SHA-256 fingerprint hash")
    total_samples: int = Field(description="Total image samples in dataset")
    total_annotations: int = Field(description="Total bounding box annotations")
    total_classes: int = Field(description="Total distinct category classes")
    class_distribution: list[ClassDistributionItem] = Field(
        description="Detailed class representation statistics"
    )
    split_distribution: dict[str, int] = Field(description="Sample counts per split partition")
    split_percentages: dict[str, float] = Field(description="Percentage distribution across splits")
    image_statistics: ImageStatistics = Field(description="Resolution and format telemetry")
    annotation_statistics: AnnotationStatistics = Field(
        description="Bounding box geometry telemetry"
    )
    class_cooccurrence: list[ClassCooccurrence] = Field(
        description="Pairwise class co-occurrence frequencies"
    )
    health_summary: DatasetHealthSummary = Field(description="Categorical health scorecard")
    profile_generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DatasetVersionRecord(BaseModel):
    """Immutable dataset version snapshot record."""

    version_id: str = Field(description="Version snapshot identifier ('v1.0.0', 'v2.0.0')")
    dataset_id: str = Field(description="Dataset identifier")
    parent_version_id: str | None = Field(default=None, description="Parent version ID if iterated")
    dataset_fingerprint: str = Field(description="SHA-256 cryptographic manifest hash")
    changes_summary: str = Field(description="Summary of additions, removals, and corrections")
    total_samples: int = Field(description="Total sample count in this version")
    total_annotations: int = Field(description="Total annotation count in this version")
    review_decisions_count: int = Field(
        default=0, description="Total curation review decisions applied"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DatasetDiffResult(BaseModel):
    """Granular comparison between two dataset versions."""

    dataset_id: str = Field(description="Dataset identifier")
    version_a: str = Field(description="Baseline version identifier")
    version_b: str = Field(description="Comparison version identifier")
    samples_added: list[str] = Field(
        default_factory=list, description="Samples present in B but not A"
    )
    samples_removed: list[str] = Field(
        default_factory=list, description="Samples present in A but not B"
    )
    classes_added: list[str] = Field(
        default_factory=list, description="Classes introduced in version B"
    )
    classes_removed: list[str] = Field(
        default_factory=list, description="Classes deprecated in version B"
    )
    annotations_count_delta: int = Field(description="Net change in total bounding boxes")
    leakage_pairs_delta: int = Field(description="Net change in detected leakage pairs")
    class_distribution_deltas: dict[str, int] = Field(
        default_factory=dict, description="Annotation count changes per class"
    )
    summary: str = Field(description="Executive summary of dataset changes")
