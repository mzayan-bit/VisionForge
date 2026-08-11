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
    training_run_ids: list[str] = Field(default_factory=list, description="Associated training run IDs")
    model_ids: list[str] = Field(default_factory=list, description="Associated model version IDs")
    evaluation_ids: list[str] = Field(default_factory=list, description="Associated evaluation IDs")
    benchmark_ids: list[str] = Field(default_factory=list, description="Associated benchmark IDs")
    inference_ids: list[str] = Field(default_factory=list, description="Associated inference run IDs")

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
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Last updated ISO timestamp"
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
    checks_passed: list[str] = Field(default_factory=list, description="List of passed audit checks")
    checks_failed: list[str] = Field(default_factory=list, description="List of failed audit checks")
    missing_dependencies: list[str] = Field(
        default_factory=list, description="Missing file/checkpoint dependencies"
    )
    verified_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Verification timestamp"
    )
