"""VisionForge Experiment Tracking, Data/Model Lineage, and Reproducibility Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ExperimentStatus(StrEnum):
    """Lifecycle status of an experiment."""

    DRAFT = "DRAFT"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class EnvironmentSnapshot(BaseModel):
    """Immutable environment telemetry snapshot for reproducibility auditing."""

    python_version: str = Field(description="Python runtime version")
    os_platform: str = Field(description="Operating System and architecture platform")
    cpu_architecture: str = Field(description="CPU model / architecture descriptor")
    gpu_device: str = Field(default="cpu", description="Compute target device (CPU, MPS, CUDA)")
    torch_version: str = Field(default="unknown", description="PyTorch framework version")
    cuda_version: str | None = Field(default=None, description="CUDA driver version if applicable")
    git_commit_sha: str = Field(default="unknown", description="Repository Git commit SHA-1 hash")
    git_branch: str = Field(default="main", description="Active Git branch")
    is_working_tree_clean: bool = Field(
        default=True, description="Whether git repository working tree was uncommitted"
    )


class RandomnessConfig(BaseModel):
    """Random seed configuration for reproducibility tracking."""

    random_seed: int = Field(default=42, ge=0, description="Master random seed")
    python_seed: int | None = Field(default=42, description="Python random module seed")
    numpy_seed: int | None = Field(default=42, description="NumPy RNG seed")
    torch_seed: int | None = Field(default=42, description="PyTorch manual seed")
    determinism_notes: str = Field(
        default="CUDA/MPS non-deterministic algorithm warnings noted.",
        description="Notes on framework determinism constraints",
    )


class ArtifactReference(BaseModel):
    """Reference metadata for an experimental artifact with cryptographic hash verification."""

    artifact_id: str = Field(description="Unique artifact ID ('art_...')")
    artifact_type: str = Field(
        description="Type of artifact (checkpoint, metrics, report, overlay, benchmark, manifest)"
    )
    name: str = Field(description="Human readable artifact display name")
    path: str = Field(description="Disk file path location")
    sha256_checksum: str | None = Field(
        default=None, description="SHA-256 cryptographic checksum for integrity verification"
    )
    size_bytes: int = Field(default=0, description="Artifact file size in bytes")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Creation timestamp"
    )


class DatasetFingerprint(BaseModel):
    """Stable cryptographic fingerprint for a dataset version manifest."""

    dataset_id: str = Field(description="Dataset identifier")
    version: str = Field(description="Dataset version string")
    preparation_id: str | None = Field(default=None, description="Dataset preparation ID")
    num_samples: int = Field(default=0, description="Total sample count")
    num_classes: int = Field(default=0, description="Total category class count")
    manifest_sha256: str = Field(description="SHA-256 hash of dataset manifest file")
    fingerprint_hash: str = Field(description="Combined unique dataset version fingerprint SHA-256")


class LineageNode(BaseModel):
    """Node in the experiment lineage graph representing a resource."""

    id: str = Field(description="Unique node entity ID")
    label: str = Field(description="Human readable node display title")
    type: str = Field(
        description="Resource type ('dataset', 'preparation', 'training_run', 'model', 'evaluation', 'benchmark', 'inference')"
    )
    status: str = Field(default="COMPLETED", description="Node resource status")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Key attributes")
    route_link: str = Field(description="Frontend route URI for navigation")


class LineageEdge(BaseModel):
    """Directed edge in the lineage graph representing dependency flow."""

    source_id: str = Field(description="Upstream source node ID")
    target_id: str = Field(description="Downstream target node ID")
    relationship_type: str = Field(description="Dependency relationship descriptor")


class LineageGraph(BaseModel):
    """Complete directed acyclic lineage graph."""

    nodes: list[LineageNode] = Field(default_factory=list, description="Graph nodes")
    edges: list[LineageEdge] = Field(default_factory=list, description="Directed edges")


class TimelineEvent(BaseModel):
    """Chronological event entry in an experiment timeline."""

    event_id: str = Field(description="Unique event ID")
    timestamp: str = Field(description="ISO timestamp")
    event_type: str = Field(description="Event category")
    title: str = Field(description="Event headline")
    description: str = Field(description="Event detailed explanation")
    entity_id: str | None = Field(default=None, description="Associated resource ID")


class Experiment(BaseModel):
    """Complete Research Experiment tracking entity with full lineage and metadata snapshots."""

    experiment_id: str = Field(description="Unique experiment ID ('exp_...')")
    name: str = Field(description="Experiment display title")
    description: str = Field(default="", description="High level experiment abstract")
    purpose: str = Field(default="", description="Research goal or problem statement")
    status: ExperimentStatus = Field(default=ExperimentStatus.DRAFT, description="Lifecycle status")
    hypothesis: str | None = Field(default=None, description="Research hypothesis being tested")
    observations: str | None = Field(default=None, description="Experimental observations")
    conclusions: str | None = Field(default=None, description="Conclusions and key findings")
    tags: list[str] = Field(default_factory=list, description="Categorical experiment tags")

    # Lineage references (IDs only to avoid redundancy)
    dataset_id: str | None = Field(default=None, description="Target dataset ID")
    dataset_version: str | None = Field(default=None, description="Target dataset version")
    dataset_fingerprint: DatasetFingerprint | None = Field(
        default=None, description="Immutable dataset fingerprint"
    )
    preparation_id: str | None = Field(default=None, description="Dataset preparation ID")
    training_run_ids: list[str] = Field(
        default_factory=list, description="Associated training run IDs"
    )
    model_ids: list[str] = Field(default_factory=list, description="Associated model version IDs")
    evaluation_ids: list[str] = Field(default_factory=list, description="Associated evaluation IDs")
    benchmark_ids: list[str] = Field(default_factory=list, description="Associated benchmark IDs")
    inference_ids: list[str] = Field(
        default_factory=list, description="Associated inference run IDs"
    )

    # Immutable Snapshots
    training_config_snapshot: dict[str, Any] | None = Field(
        default=None, description="Frozen snapshot of training configuration"
    )
    environment_snapshot: EnvironmentSnapshot = Field(
        description="Captured runtime environment snapshot"
    )
    randomness: RandomnessConfig = Field(
        default_factory=RandomnessConfig, description="Random seed configuration"
    )
    artifacts: list[ArtifactReference] = Field(
        default_factory=list, description="Tracked artifact references"
    )
    parent_experiment_id: str | None = Field(
        default=None, description="Parent experiment ID if this is a reproduction attempt"
    )

    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Creation ISO timestamp"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Last updated ISO timestamp",
    )


class ExperimentComparison(BaseModel):
    """Side-by-side comparison telemetry between two experiments."""

    experiment_a_id: str = Field(description="First experiment ID")
    experiment_b_id: str = Field(description="Second experiment ID")
    config_diff: dict[str, list[Any]] = Field(
        default_factory=dict, description="Configuration differences dict: key -> [val_a, val_b]"
    )
    metric_diff: dict[str, list[Any]] = Field(
        default_factory=dict, description="Metric differences dict: key -> [val_a, val_b]"
    )
    summary_notes: str = Field(description="Comparative summary analysis")


class ReproducibilityReport(BaseModel):
    """Audit report validating the reproducibility of an experiment."""

    experiment_id: str = Field(description="Audited experiment ID")
    is_reproducible: bool = Field(description="Overall reproducibility verification pass status")
    checks_passed: list[str] = Field(
        default_factory=list, description="List of passed audit checks"
    )
    checks_failed: list[str] = Field(
        default_factory=list, description="List of failed audit checks"
    )
    missing_dependencies: list[str] = Field(
        default_factory=list, description="Missing file/checkpoint dependencies"
    )
    verified_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Verification timestamp"
    )


# ─── Research Benchmark & Ablation Lab Schemas ─────────────────────


class EvaluationProtocol(BaseModel):
    """Locked evaluation protocol configuration guaranteeing strict reproducibility."""

    dataset_split: str = Field(default="test", description="Target split partition ('test', 'val')")
    primary_metric: str = Field(
        default="map50", description="Primary objective metric ('map50', 'recall', 'precision')"
    )
    iou_threshold: float = Field(default=0.50, description="Evaluation IoU threshold")
    confidence_threshold: float = Field(default=0.25, description="Prediction confidence threshold")
    class_handling: str = Field(default="macro_average", description="Class aggregation strategy")
    is_locked: bool = Field(
        default=True, description="Whether protocol is locked to prevent protocol drift"
    )


class AggregatedMetricStats(BaseModel):
    """Descriptive statistics across multiple random seed runs."""

    metric_name: str = Field(description="Metric attribute name")
    count: int = Field(description="Number of valid evaluation runs evaluated")
    mean: float = Field(description="Sample mean value")
    std_dev: float = Field(description="Sample standard deviation (Bessel corrected)")
    min: float = Field(description="Minimum measured value")
    max: float = Field(description="Maximum measured value")
    confidence_interval_95: list[float] | None = Field(
        default=None, description="95% confidence interval [lower, upper] if sample count >= 3"
    )
    is_single_run: bool = Field(
        default=False, description="Whether metric is from a single unrepeated run"
    )
    warning: str | None = Field(
        default=None, description="Statistical limitation warning (e.g. Single run alert)"
    )


class ExperimentRunRecord(BaseModel):
    """Individual seed trial / evaluation run within a variant."""

    run_id: str = Field(description="Target TrainingRun or EvaluationRun ID")
    seed: int = Field(description="Explicit random seed")
    model_id: str = Field(description="Associated model checkpoint name")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Scalar metrics (map50, precision, recall, etc.)"
    )
    per_class_metrics: dict[str, float] = Field(
        default_factory=dict, description="Class-level metric scores"
    )
    error_counts: dict[str, int] = Field(
        default_factory=dict, description="Error taxonomy frequencies"
    )
    training_time_sec: float | None = Field(
        default=None, description="Training duration in seconds"
    )
    gpu_hours: float | None = Field(default=None, description="Measured GPU compute hours")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ExperimentVariant(BaseModel):
    """Specific experimental configuration branch compared against baseline."""

    variant_id: str = Field(description="Unique variant identifier ('var_...')")
    name: str = Field(description="Display title (e.g. 'Strong Augmentation', 'Resolution 1024')")
    description: str = Field(default="", description="Summary of changes in this branch")
    is_baseline: bool = Field(
        default=False, description="Whether this variant represents the control baseline"
    )
    config_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit configuration parameters changed relative to baseline",
    )
    dataset_id: str | None = Field(default=None, description="Dataset ID if dataset ablation")
    dataset_version: str | None = Field(
        default=None, description="Dataset version if dataset ablation"
    )
    runs: list[ExperimentRunRecord] = Field(
        default_factory=list, description="Trial runs across seeds"
    )
    aggregated_metrics: dict[str, AggregatedMetricStats] = Field(
        default_factory=dict, description="Aggregated metric statistics across runs"
    )
    aggregated_per_class: dict[str, AggregatedMetricStats] = Field(
        default_factory=dict, description="Aggregated per-class scores across runs"
    )
    aggregated_error_counts: dict[str, AggregatedMetricStats] = Field(
        default_factory=dict, description="Aggregated error counts across runs"
    )
    label_count: int | None = Field(
        default=None, description="Annotated label budget (for Active Learning)"
    )
    label_percentage: float | None = Field(default=None, description="Dataset percentage budget")


class VariableDiffItem(BaseModel):
    """Field-level configuration diff comparing Baseline vs Variant."""

    parameter: str = Field(description="Configuration parameter name")
    baseline_value: Any = Field(description="Baseline setting")
    variant_value: Any = Field(description="Variant setting")
    has_changed: bool = Field(description="Whether parameter was modified")
    component_type: str = Field(
        default="hyperparameter",
        description="Category: 'augmentation', 'resolution', 'architecture', 'dataset', 'active_learning', 'training'",
    )


class AblationRow(BaseModel):
    """Component contribution row in an Ablation Study matrix."""

    component: str = Field(
        description="Component name (e.g. 'Augmentation', 'Active-Learning', '1024-Resolution')"
    )
    baseline_present: bool = Field(description="Whether present in Baseline")
    variant_present: bool = Field(description="Whether present in Variant")
    measured_effect_delta: float | None = Field(
        default=None, description="Measured performance metric delta"
    )
    metric_name: str = Field(default="map50", description="Evaluation metric measured")


class AblationStudy(BaseModel):
    """Component ablation study representation."""

    ablation_id: str = Field(description="Ablation study ID ('abl_...')")
    name: str = Field(description="Ablation study title")
    hypothesis: str = Field(description="Component contribution hypothesis")
    components: list[str] = Field(
        default_factory=list, description="List of isolated system components"
    )
    matrix: list[AblationRow] = Field(
        default_factory=list, description="Component presence and measured effect matrix"
    )
    measured_effects: dict[str, float] = Field(
        default_factory=dict, description="Component -> measured performance delta"
    )


class ResearchExperiment(BaseModel):
    """Complete Research Benchmark, Controlled Experiment, & Ablation entity."""

    experiment_id: str = Field(description="Unique experiment ID ('rexp_...')")
    name: str = Field(description="Experiment title")
    description: str = Field(default="", description="Abstract or context")
    hypothesis: str = Field(description="Researcher-provided hypothesis text")
    baseline_variant_id: str = Field(description="Baseline variant identifier")
    variants: list[ExperimentVariant] = Field(
        default_factory=list, description="List of experimental variants"
    )
    dataset_id: str = Field(description="Benchmark dataset identifier")
    dataset_version: str = Field(description="Benchmark dataset version")
    evaluation_protocol: EvaluationProtocol = Field(
        default_factory=EvaluationProtocol, description="Locked evaluation protocol"
    )
    status: ExperimentStatus = Field(
        default=ExperimentStatus.DRAFT, description="Experiment lifecycle status"
    )
    ablation_study: AblationStudy | None = Field(
        default=None, description="Associated component ablation study"
    )
    conclusions: str | None = Field(default=None, description="Grounded findings and observations")
    limitations: str | None = Field(
        default=None, description="Methodological limitations or caveats"
    )
    reproducibility_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Git commit, seed list, environment hash, and protocol lock",
    )
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ResearchReport(BaseModel):
    """Formal structured research report with grounded conclusions."""

    experiment_id: str = Field(description="Associated ResearchExperiment ID")
    title: str = Field(description="Research report title")
    hypothesis: str = Field(description="Tested hypothesis")
    dataset_summary: str = Field(description="Dataset and split configuration")
    baseline_summary: str = Field(description="Baseline performance summary")
    variants_summary: str = Field(description="Summary of variant performance and deltas")
    performance_deltas: dict[str, float] = Field(
        default_factory=dict, description="Variant -> metric delta relative to baseline"
    )
    per_class_deltas: dict[str, float] = Field(
        default_factory=dict, description="Class label -> performance delta"
    )
    error_deltas: dict[str, float] = Field(
        default_factory=dict, description="Error taxonomy percentage changes"
    )
    statistical_conclusions: list[str] = Field(
        default_factory=list, description="Factually grounded statistical conclusions"
    )
    grounded_conclusions: str = Field(
        description="Executive research summary strictly derived from evidence"
    )
    limitations: list[str] = Field(default_factory=list, description="Limitations and warnings")
    markdown_report: str = Field(
        description="Complete GitHub-flavored markdown research report document"
    )
