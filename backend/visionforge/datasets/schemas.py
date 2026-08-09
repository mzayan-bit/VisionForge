"""VisionForge Dataset Preparation Pipeline Data Models & Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class PreparationStatus(StrEnum):
    """Lifecycle state of a dataset preparation run."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    SPLITTING = "SPLITTING"
    VERIFYING = "VERIFYING"
    MATERIALIZING = "MATERIALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SplitStrategy(StrEnum):
    """Dataset splitting strategy options."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP_AWARE = "group_aware"


class IssueSeverity(StrEnum):
    """Severity classification for validation findings."""

    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    """Individual issue found during dataset validation."""

    sample_id: str = Field(description="Sample record ID associated with issue")
    issue_type: str = Field(description="Category classification of the issue")
    message: str = Field(description="Detailed explanation message")
    severity: IssueSeverity = Field(default=IssueSeverity.WARNING, description="Issue severity level")


class ValidationReport(BaseModel):
    """Comprehensive validation report produced before dataset partitioning."""

    status: str = Field(description="Overall validation status ('PASSED', 'PASSED_WITH_WARNINGS', 'FAILED')")
    total_samples: int = Field(description="Total input dataset samples checked")
    valid_samples: int = Field(description="Total valid samples ready for splitting")
    corrupted_samples_count: int = Field(default=0, description="Count of unreadable/corrupted files")
    missing_embeddings_count: int = Field(default=0, description="Count of samples lacking vector embeddings")
    issues: list[ValidationIssue] = Field(default_factory=list, description="Detailed validation issues list")


class LeakageFinding(BaseModel):
    """Descriptor for a detected data leakage group (exact duplicate or near-duplicate)."""

    group_id: str = Field(description="Unique leakage cluster group ID")
    leakage_type: str = Field(description="'EXACT_DUPLICATE' or 'POSSIBLE_NEAR_DUPLICATE'")
    sample_ids: list[str] = Field(description="List of sample IDs in this leakage group")
    similarity_score: float = Field(default=1.0, description="Similarity score or hash match confidence")


class SplitConfig(BaseModel):
    """Configuration settings for partitioning a dataset."""

    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0, description="Training set fraction")
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0, description="Validation set fraction")
    test_ratio: float = Field(default=0.15, ge=0.0, lt=1.0, description="Test set fraction")
    random_seed: int = Field(default=42, ge=0, description="Random seed for 100% reproducible splitting")
    strategy: SplitStrategy = Field(default=SplitStrategy.RANDOM, description="Partitioning algorithm")
    group_by_field: str | None = Field(default=None, description="Metadata field for group-aware splitting")
    stratify_by_field: str | None = Field(default=None, description="Metadata field for stratified splitting")


class SampleRef(BaseModel):
    """Prepared sample reference metadata."""

    id: str = Field(description="Sample record ID")
    split: str = Field(description="Assigned split partition ('train', 'validation', 'test')")
    file_path: str = Field(default="", description="Source file location or reference")
    content_hash: str = Field(default="", description="SHA-256 content hash")
    image_metadata: dict[str, Any] = Field(default_factory=dict, description="Image dimensions and format")
    tags: list[str] = Field(default_factory=list, description="Classification tags or labels")
    leakage_group_id: str | None = Field(default=None, description="Associated leakage group ID if any")


class SplitStats(BaseModel):
    """Statistics descriptor for a single split partition."""

    split_name: str = Field(description="'train', 'validation', or 'test'")
    count: int = Field(description="Number of samples in partition")
    ratio: float = Field(description="Actual percentage ratio achieved")
    format_distribution: dict[str, int] = Field(default_factory=dict, description="File format breakdown")
    category_distribution: dict[str, int] = Field(default_factory=dict, description="Label distribution if labels exist")


class DatasetPreparationManifest(BaseModel):
    """Machine-readable, fully reproducible dataset split manifest."""

    manifest_version: str = Field(default="1.0.0", description="Manifest format version")
    preparation_id: str = Field(description="Unique preparation transaction ID ('prep_...')")
    dataset_id: str = Field(description="Source dataset identifier")
    dataset_version: str = Field(description="Source dataset version")
    creation_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(), description="ISO timestamp")
    random_seed: int = Field(description="Random seed used for splitting")
    split_config: SplitConfig = Field(description="Configured split parameters")
    software_version: str = Field(default="VisionForge v0.1.0", description="System software version")
    total_samples: int = Field(description="Total samples prepared")
    train_count: int = Field(description="Training split count")
    val_count: int = Field(description="Validation split count")
    test_count: int = Field(description="Test split count")
    exact_duplicates_found: int = Field(description="Total exact duplicate samples")
    near_duplicates_found: int = Field(description="Total near-duplicate samples")
    samples: list[SampleRef] = Field(description="Complete sample list with split assignments")


class PreparationRun(BaseModel):
    """Full execution transaction data model for a dataset preparation run."""

    preparation_id: str = Field(description="Unique preparation transaction ID ('prep_...')")
    dataset_id: str = Field(description="Target dataset ID")
    dataset_version: str = Field(description="Target dataset version")
    status: PreparationStatus = Field(default=PreparationStatus.CREATED, description="Current lifecycle state")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(), description="ISO timestamp")
    completed_at: str | None = Field(default=None, description="ISO completion timestamp")
    split_config: SplitConfig = Field(description="Configured split parameters")
    validation_report: ValidationReport | None = Field(default=None, description="Pre-split validation findings")
    leakage_findings: list[LeakageFinding] = Field(default_factory=list, description="Leakage groups detected")
    split_stats: dict[str, SplitStats] = Field(default_factory=dict, description="Per-split partition statistics")
    manifest_path: str | None = Field(default=None, description="Path to generated manifest JSON file")
    error_message: str | None = Field(default=None, description="Error message if run failed")
