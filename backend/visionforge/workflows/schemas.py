"""VisionForge End-to-End Computer Vision Research Workflow Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowStatus(StrEnum):
    """Lifecycle state machine for a controlled research workflow."""

    DRAFT = "DRAFT"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    WAITING_FOR_REVIEW = "WAITING_FOR_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class WorkflowStage(StrEnum):
    """The 8 finite sequential stages of a VisionForge research workflow."""

    RESEARCH_DEFINITION = "RESEARCH_DEFINITION"
    DATASET = "DATASET"
    EXPERIMENT = "EXPERIMENT"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    ERROR_ANALYSIS = "ERROR_ANALYSIS"
    COMPARISON = "COMPARISON"
    REPORT = "REPORT"


class WorkflowTemplateType(StrEnum):
    """Pre-configured research study templates."""

    ACTIVE_LEARNING_STUDY = "ACTIVE_LEARNING_STUDY"
    BASELINE_VS_VARIANT = "BASELINE_VS_VARIANT"
    MODEL_ARCHITECTURE_COMPARISON = "MODEL_ARCHITECTURE_COMPARISON"
    CUSTOM = "CUSTOM"


class DecisionType(StrEnum):
    """Researcher human decision gate outcomes."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INVESTIGATE = "INVESTIGATE"


class DecisionRecord(BaseModel):
    """Human decision record capturing scientific judgment at review gates."""

    decision_id: str = Field(description="Unique decision ID")
    decision: DecisionType = Field(description="ACCEPT, REJECT, or INVESTIGATE")
    reviewer: str = Field(default="Researcher", description="Reviewer identity")
    rationale: str = Field(description="Qualitative justification for decision")
    target_stage: WorkflowStage | None = Field(
        default=None, description="Destination stage if returning to investigation loop"
    )
    iteration: int = Field(default=1, description="Workflow iteration cycle number")
    decided_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ResearchDefinition(BaseModel):
    """Initial stage specification defining scientific questions and constraints."""

    research_question: str = Field(description="Core question being investigated")
    hypothesis: str = Field(description="Testable researcher-provided hypothesis")
    objective: str = Field(default="", description="Primary research objective")
    success_metrics: list[str] = Field(
        default_factory=lambda: ["map50"], description="Quantitative metrics defining success"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Methodological constraints (e.g. fixed test set, label budget)",
    )


class DatasetConfig(BaseModel):
    """Locked dataset version and split configuration."""

    dataset_id: str = Field(default="safety_v2", description="Dataset identifier")
    dataset_version: str = Field(default="v2.0.0", description="Dataset version string")
    train_split: str = Field(default="train", description="Training partition")
    val_split: str = Field(default="val", description="Validation partition")
    test_split: str = Field(default="test", description="Locked evaluation benchmark partition")
    is_locked: bool = Field(default=True, description="Whether dataset is locked against drift")
    dataset_fingerprint: str | None = Field(
        default=None, description="Cryptographic SHA-256 fingerprint"
    )


class StageNote(BaseModel):
    """Researcher observational notes attached to specific workflow stages."""

    note_id: str = Field(description="Unique note ID")
    stage: WorkflowStage = Field(description="Stage at which note was written")
    author: str = Field(default="Researcher", description="Author identity")
    text: str = Field(description="Observational notes or caveats")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class WorkflowEvent(BaseModel):
    """Chronological event telemetry record for auditability."""

    event_id: str = Field(description="Unique event ID")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    stage: WorkflowStage = Field(description="Workflow stage when event occurred")
    event_type: str = Field(
        description="Event classification (e.g. STAGE_COMPLETED, DECISION_MADE)"
    )
    message: str = Field(description="Human-readable event summary")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured attributes")


class WorkflowLineageNode(BaseModel):
    """Lineage node representing an entity in the workflow DAG."""

    id: str = Field(description="Entity identifier")
    label: str = Field(description="Display title")
    stage: WorkflowStage = Field(description="Associated workflow stage")
    entity_type: str = Field(
        description="Type: 'research', 'dataset', 'experiment', 'training_run', 'model', 'evaluation', 'analysis', 'report'"
    )
    status: str = Field(default="COMPLETED")
    route_link: str = Field(description="Frontend navigation link")


class WorkflowLineageEdge(BaseModel):
    """Directed dependency link in workflow DAG."""

    source_id: str = Field(description="Upstream entity ID")
    target_id: str = Field(description="Downstream entity ID")
    relationship: str = Field(description="Relationship name")


class WorkflowLineageGraph(BaseModel):
    """Complete workflow lineage graph."""

    nodes: list[WorkflowLineageNode] = Field(default_factory=list)
    edges: list[WorkflowLineageEdge] = Field(default_factory=list)


class ResearchWorkflow(BaseModel):
    """Complete End-to-End Computer Vision Research Workflow."""

    workflow_id: str = Field(description="Unique workflow ID ('wf_...')")
    name: str = Field(description="Workflow title")
    description: str = Field(default="", description="Summary of investigation")
    template_type: WorkflowTemplateType = Field(default=WorkflowTemplateType.CUSTOM)
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT)
    current_stage: WorkflowStage = Field(default=WorkflowStage.RESEARCH_DEFINITION)
    current_iteration: int = Field(default=1, description="Iteration cycle counter")
    research_definition: ResearchDefinition = Field(
        description="Scientific question and hypothesis"
    )
    dataset_config: DatasetConfig = Field(
        default_factory=DatasetConfig, description="Locked dataset configuration"
    )
    experiment_id: str | None = Field(default=None, description="Linked ResearchExperiment ID")
    baseline_run_id: str | None = Field(default=None, description="Linked Baseline TrainingRun ID")
    variant_run_ids: list[str] = Field(
        default_factory=list, description="Linked Variant TrainingRun IDs"
    )
    evaluation_ids: list[str] = Field(default_factory=list, description="Linked EvaluationRun IDs")
    error_analysis_ids: list[str] = Field(
        default_factory=list, description="Linked ErrorAnalysis snapshot IDs"
    )
    stage_notes: list[StageNote] = Field(
        default_factory=list, description="Researcher qualitative notes"
    )
    timeline_events: list[WorkflowEvent] = Field(
        default_factory=list, description="Audit log events"
    )
    decision_history: list[DecisionRecord] = Field(
        default_factory=list, description="Human decision gate records"
    )
    artifact_ids: list[str] = Field(
        default_factory=list, description="Associated artifact references"
    )
    generated_report_markdown: str | None = Field(
        default=None, description="Synthesized final research report"
    )
    reproducibility_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Reproducibility environment and seeds"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class WorkflowExportPackage(BaseModel):
    """Self-contained exportable research workflow package."""

    workflow: ResearchWorkflow = Field(description="Full workflow record")
    experiment_snapshot: dict[str, Any] | None = Field(default=None)
    evaluations_summary: list[dict[str, Any]] = Field(default_factory=list)
    report_markdown: str = Field(description="Markdown report document")
    reproducibility_hash: str = Field(description="Cryptographic validation hash")
    exported_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
