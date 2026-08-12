"""VisionForge Active Learning & Intelligent Sample Selection Schemas."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SelectionStrategy(StrEnum):
    """Supported active learning sample selection strategies."""

    UNCERTAINTY = "UNCERTAINTY"
    DIVERSITY = "DIVERSITY"
    UNCERTAINTY_DIVERSITY = "UNCERTAINTY_DIVERSITY"
    NOVELTY = "NOVELTY"


class ReviewStatus(StrEnum):
    """Human review decision status for recommended samples."""

    UNREVIEWED = "UNREVIEWED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"
    MARKED_FOR_LABELING = "MARKED_FOR_LABELING"


class SignalWeights(BaseModel):
    """Configurable weights for combining selection signals into composite rank score."""

    uncertainty: float = Field(
        default=0.40, ge=0.0, le=1.0, description="Model prediction uncertainty weight"
    )
    novelty: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Embedding space distance novelty weight"
    )
    diversity: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Visual coverage diversity weight"
    )
    failure: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Historical failure relevance weight"
    )
    quality: float = Field(
        default=0.00, ge=0.0, le=1.0, description="Image quality signal weight"
    )


class SampleSignals(BaseModel):
    """Individual normalized scoring signals for a candidate sample."""

    image_id: str = Field(description="Unique candidate sample image ID")
    image_path: str = Field(description="Candidate image file path")
    uncertainty_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Model prediction uncertainty proxy score"
    )
    novelty_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Embedding distance from dataset centroid"
    )
    diversity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Farthest-point visual coverage distance score"
    )
    failure_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance score to past model failures"
    )
    quality_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Image quality score (resolution, blur, contrast)"
    )
    composite_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Final weighted composite selection score"
    )


class RankedSample(BaseModel):
    """Individual recommended sample entry in the review queue."""

    rank: int = Field(description="Sample recommendation rank index (1 = top choice)")
    image_id: str = Field(description="Unique image identifier")
    image_path: str = Field(description="Image file location")
    composite_score: float = Field(description="Final composite rank score [0.0, 1.0]")
    signals: SampleSignals = Field(description="Detailed breakdown of individual signal scores")
    recommendation_reason: str = Field(description="Plain-English explanation of recommendation signal rationale")
    review_status: ReviewStatus = Field(
        default=ReviewStatus.UNREVIEWED, description="Human review status"
    )
    notes: str | None = Field(default=None, description="Optional researcher feedback note")


class CandidatePoolDescriptor(BaseModel):
    """Descriptor defining candidate image pool source and test-set exclusion tracking."""

    pool_id: str = Field(description="Unique candidate pool ID ('pool_...')")
    name: str = Field(description="Pool display name")
    source_dataset_id: str = Field(description="Source dataset identifier")
    source_version: str = Field(default="v1.0", description="Dataset version tag")
    candidate_paths: list[str] = Field(
        default_factory=list, description="Validated candidate image file paths"
    )
    total_candidates: int = Field(default=0, description="Total eligible candidate images")
    excluded_test_samples_count: int = Field(
        default=0, description="Number of test split images filtered out for protection"
    )


class ActiveLearningRun(BaseModel):
    """Complete record of an Active Learning sample selection execution."""

    run_id: str = Field(description="Unique active learning run ID ('al_run_...')")
    experiment_id: str | None = Field(default=None, description="Associated experiment ID")
    model_id: str = Field(description="Target model identifier used for uncertainty signals")
    model_version: str = Field(default="1.0.0", description="Model version tag")
    dataset_id: str = Field(description="Target dataset identifier")
    candidate_pool_id: str = Field(description="Candidate image pool identifier")
    strategy: SelectionStrategy = Field(description="Selected sample ranking strategy")
    weights: SignalWeights = Field(default_factory=SignalWeights, description="Signal combination weights")
    top_k: int = Field(default=25, ge=1, le=500, description="Number of top candidates requested")
    selected_samples: list[RankedSample] = Field(
        default_factory=list, description="Ranked candidate sample recommendations"
    )
    status: str = Field(default="COMPLETED", description="Run execution status")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Execution ISO timestamp"
    )


class ReviewDecisionRequest(BaseModel):
    """Payload for submitting a human review decision on a sample."""

    run_id: str = Field(description="Active learning run ID")
    image_id: str = Field(description="Sample image ID")
    status: ReviewStatus = Field(description="Review decision status")
    notes: str | None = Field(default=None, description="Researcher notes")


class SelectionBiasReport(BaseModel):
    """Telemetry report measuring potential selection bias across recommended samples."""

    run_id: str = Field(description="Active learning run ID")
    strategy: SelectionStrategy = Field(description="Selection strategy used")
    total_selected: int = Field(description="Total selected sample count")
    class_distribution: dict[str, int] = Field(
        default_factory=dict, description="Predicted class counts in selected set"
    )
    quality_distribution: dict[str, int] = Field(
        default_factory=dict, description="Image quality breakdown (high, medium, low)"
    )
    confidence_distribution: dict[str, float] = Field(
        default_factory=dict, description="Confidence quartile statistics"
    )
    bias_summary: str = Field(description="Qualitative summary of selection bias metrics")


class StrategyComparisonRequest(BaseModel):
    """Payload for comparing two active learning selection strategies."""

    dataset_id: str = Field(description="Target dataset ID")
    model_id: str = Field(description="Target model ID")
    candidate_pool_id: str = Field(description="Candidate pool ID")
    strategy_a: SelectionStrategy = Field(description="First selection strategy")
    strategy_b: SelectionStrategy = Field(description="Second selection strategy")
    top_k: int = Field(default=25, ge=1, le=200, description="Top-K sample count")


class StrategyComparisonResult(BaseModel):
    """Comparative analysis result between two selection strategies."""

    dataset_id: str = Field(description="Evaluated dataset ID")
    model_id: str = Field(description="Evaluated model ID")
    strategy_a: SelectionStrategy = Field(description="Strategy A")
    strategy_b: SelectionStrategy = Field(description="Strategy B")
    overlap_count: int = Field(description="Number of identical samples selected by both strategies")
    unique_a_count: int = Field(description="Samples unique to Strategy A")
    unique_b_count: int = Field(description="Samples unique to Strategy B")
    diversity_delta: float = Field(description="Visual coverage diversity difference")
    uncertainty_delta: float = Field(description="Average uncertainty difference")
    summary_notes: str = Field(description="Qualitative strategy comparison analysis")


class ImprovementVerdict(StrEnum):
    """Empirical verdict answering: Did performance actually improve?"""

    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    NEUTRAL = "NEUTRAL"


class MetricDelta(BaseModel):
    """Specific metric change before and after active learning retraining."""

    baseline_val: float = Field(description="Evaluation metric score on baseline model")
    retrained_val: float = Field(description="Evaluation metric score on retrained model")
    delta: float = Field(description="Absolute score difference (retrained - baseline)")
    percent_change: float = Field(description="Relative percentage change (%)")


class ActiveLearningIteration(BaseModel):
    """Complete closed-loop active learning retraining iteration record."""

    iteration_id: str = Field(description="Unique iteration ID ('iter_...')")
    baseline_dataset_id: str = Field(description="Baseline dataset ID (D0)")
    baseline_model_id: str = Field(description="Baseline model ID (M0)")
    baseline_evaluation_id: str = Field(description="Baseline evaluation ID (E0)")
    active_learning_run_id: str = Field(description="Active learning recommendation run ID")
    reviewed_samples_count: int = Field(description="Count of accepted human-reviewed samples")
    new_dataset_version: str = Field(description="New dataset version tag (D1)")
    retrained_run_id: str = Field(description="Controlled retraining run ID (Run M1)")
    retrained_model_id: str = Field(description="Retrained model ID (M1)")
    retrained_evaluation_id: str = Field(description="Retrained evaluation ID on untouched test split (E1)")
    map50_delta: MetricDelta = Field(description="mAP@50 delta telemetry")
    map50_95_delta: MetricDelta = Field(description="mAP@50:95 delta telemetry")
    precision_delta: MetricDelta = Field(description="Precision delta telemetry")
    recall_delta: MetricDelta = Field(description="Recall delta telemetry")
    verdict: ImprovementVerdict = Field(description="Final empirical performance verdict")
    verdict_summary: str = Field(description="Qualitative explanation of performance delta")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Execution ISO timestamp"
    )


class ExecuteLoopRequest(BaseModel):
    """Payload for executing an end-to-end active learning retraining iteration."""

    baseline_dataset_id: str = Field(default="safety_v2")
    baseline_model_id: str = Field(default="yolo11s.pt")
    active_learning_run_id: str = Field(description="Active learning run ID containing reviewed samples")
    new_version_tag: str | None = Field(default=None, description="Optional new dataset version tag (e.g. v2.1)")

