"""VisionForge Active Learning & Human-in-the-Loop Workflow Schemas.

Comprehensive domain models for:
- Uncertainty Sampling, Diversity Farthest-Point Clustering, and Hybrid Prioritization
- Evidence-Based Candidate Explanations
- Interactive Human Review Sessions with Keyboard Shortcuts
- Multi-Reviewer Consistency & Consensus Resolution
- Active Learning Cycles, Dataset Version Creation, and Retraining Integration
- Longitudinal Cycle Progression & Diminishing Returns Tracking
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SelectionStrategy(StrEnum):
    """Supported active learning sample selection strategies."""

    UNCERTAINTY = "UNCERTAINTY"
    DIVERSITY = "DIVERSITY"
    HYBRID = "HYBRID"
    UNCERTAINTY_DIVERSITY = "UNCERTAINTY_DIVERSITY"
    NOVELTY = "NOVELTY"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    CLASS_AWARE = "CLASS_AWARE"
    FAILURE_AWARE = "FAILURE_AWARE"


class ReviewStatus(StrEnum):
    """Lifecycle queue status for candidate samples."""

    UNREVIEWED = "UNREVIEWED"
    IN_REVIEW = "IN_REVIEW"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"
    FLAGGED = "FLAGGED"
    MARKED_FOR_LABELING = "MARKED_FOR_LABELING"


class ReviewDecisionType(StrEnum):
    """Granular human review decision taxonomy."""

    CONFIRMED = "CONFIRMED"
    INCORRECT_PREDICTION = "INCORRECT_PREDICTION"
    ANNOTATION_ISSUE = "ANNOTATION_ISSUE"
    VALID_HARD_EXAMPLE = "VALID_HARD_EXAMPLE"
    DUPLICATE = "DUPLICATE"
    NOT_USEFUL = "NOT_USEFUL"
    NEEDS_MORE_REVIEW = "NEEDS_MORE_REVIEW"
    SKIP = "SKIP"


class ReviewerAgreementStatus(StrEnum):
    """Consensus state across multiple reviewers for the same sample."""

    UNANIMOUS = "UNANIMOUS"
    SPLIT = "SPLIT"
    NEEDS_RESOLUTION = "NEEDS_RESOLUTION"


class SignalWeights(BaseModel):
    """Configurable weights for combining selection signals into composite rank score."""

    uncertainty: float = Field(
        default=0.40, ge=0.0, le=1.0, description="Model prediction uncertainty weight"
    )
    diversity: float = Field(
        default=0.40, ge=0.0, le=1.0, description="Visual coverage diversity weight"
    )
    failure: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Historical failure relevance weight"
    )
    novelty: float = Field(
        default=0.00, ge=0.0, le=1.0, description="Embedding space distance novelty weight"
    )
    quality: float = Field(default=0.00, ge=0.0, le=1.0, description="Image quality signal weight")


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


class CandidateExplanation(BaseModel):
    """Evidence-based explanation of why a sample was prioritized for review."""

    composite_priority: float = Field(description="Final composite priority [0.0, 1.0]")
    uncertainty_contribution: float = Field(
        description="Contribution from model uncertainty margin"
    )
    diversity_contribution: float = Field(
        description="Contribution from visual representation diversity"
    )
    failure_contribution: float = Field(description="Contribution from benchmark error relevance")
    class_rarity_flag: bool = Field(
        default=False, description="Flagged if belongs to under-represented class"
    )
    model_disagreement_flag: bool = Field(
        default=False, description="Flagged if baseline and candidate disagree"
    )
    plain_text_reasons: list[str] = Field(
        default_factory=list, description="Bullet-point plain language explanations for researcher"
    )


class CandidateSampleDetail(BaseModel):
    """Detailed candidate sample representation for human review cards."""

    rank: int = Field(description="Rank index in review queue (1 = top priority)")
    image_id: str = Field(description="Unique image identifier")
    image_path: str = Field(description="Image file location")
    split: str = Field(default="unlabeled", description="Source split or partition")
    composite_score: float = Field(description="Final composite priority score [0.0, 1.0]")
    signals: SampleSignals = Field(description="Detailed signal breakdown")
    explanation: CandidateExplanation = Field(description="Evidence-based selection rationale")
    recommendation_reason: str = Field(default="", description="Human-readable reason")
    ground_truth_boxes: list[dict[str, Any]] = Field(
        default_factory=list, description="Ground truth bounding boxes if available"
    )
    predicted_boxes: list[dict[str, Any]] = Field(
        default_factory=list, description="Model candidate predictions"
    )
    predicted_class: str | None = Field(default=None, description="Top predicted class")
    confidence: float | None = Field(default=None, description="Top prediction confidence")
    iou: float | None = Field(
        default=None, description="Intersection over Union with GT if available"
    )
    similar_sample_ids: list[str] = Field(
        default_factory=list, description="Nearest neighbor sample IDs from visual memory"
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.UNREVIEWED, description="Queue review state"
    )
    review_decision: ReviewDecisionType | None = Field(
        default=None, description="Recorded human review decision"
    )
    notes: str | None = Field(default=None, description="Reviewer feedback notes")


class ReviewerDecisionRecord(BaseModel):
    """Individual human review decision audit log record."""

    decision_id: str = Field(description="Unique decision record ID")
    cycle_id: str = Field(description="Associated Active Learning Cycle ID")
    sample_id: str = Field(description="Target image sample ID")
    reviewer_id: str = Field(description="Researcher / Reviewer identity")
    decision: ReviewDecisionType = Field(description="Human decision taxonomy")
    ground_truth_class: str | None = Field(default=None, description="Confirmed or corrected class")
    predicted_class: str | None = Field(default=None, description="Original model predicted class")
    confidence: float | None = Field(default=None, description="Original prediction confidence")
    notes: str = Field(default="", description="Reviewer notes and annotations")
    bbox_corrections: list[dict[str, Any]] = Field(
        default_factory=list, description="Adjusted bounding box geometry if corrected"
    )
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SampleReviewConsensus(BaseModel):
    """Multi-reviewer consensus evaluation for a sample."""

    sample_id: str = Field(description="Target sample ID")
    decisions: list[ReviewerDecisionRecord] = Field(default_factory=list)
    consensus_status: ReviewerAgreementStatus = Field(default=ReviewerAgreementStatus.UNANIMOUS)
    final_decision: ReviewDecisionType | None = Field(default=None)


class ActiveLearningCycle(BaseModel):
    """Complete cycle entity for Active Learning & Human-in-the-Loop workflow."""

    cycle_id: str = Field(description="Unique cycle ID ('al_cycle_...')")
    name: str = Field(description="Descriptive cycle name")
    dataset_id: str = Field(description="Source dataset identifier")
    dataset_version: str = Field(
        default="v1.0.0", description="Input dataset version tag (e.g. 'v12')"
    )
    model_id: str = Field(description="Target model identifier (e.g. 'yolo11s.pt')")
    model_version: str = Field(default="1.0.0", description="Model version tag")
    candidate_pool_id: str = Field(default="pool_01", description="Candidate image pool identifier")
    candidate_pool_size: int = Field(default=0, description="Total eligible candidate pool size")
    strategy: SelectionStrategy = Field(
        default=SelectionStrategy.HYBRID, description="Selection strategy"
    )
    budget: int = Field(default=50, ge=1, le=500, description="Exact human review sample budget")
    weights: SignalWeights = Field(
        default_factory=SignalWeights, description="Signal combination weights"
    )
    selected_samples: list[CandidateSampleDetail] = Field(
        default_factory=list, description="Prioritized candidates selected within budget"
    )
    review_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "pending": 0,
            "in_review": 0,
            "reviewed": 0,
            "skipped": 0,
            "flagged": 0,
        },
        description="Live queue count breakdown",
    )
    resulting_dataset_version: str | None = Field(
        default=None, description="New dataset version produced upon explicit commit (e.g. 'v13')"
    )
    benchmark_before_map50: float | None = Field(
        default=None, description="mAP@50 on baseline dataset before curation"
    )
    benchmark_after_map50: float | None = Field(
        default=None, description="mAP@50 on curated dataset after retraining"
    )
    status: str = Field(default="PLANNING", description="'PLANNING', 'IN_REVIEW', 'COMPLETED'")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = Field(default=None)
    # Backward compatibility
    run_id: str | None = Field(default=None, description="Legacy run ID alias")
    top_k: int | None = Field(default=None, description="Legacy top-K alias")
    experiment_id: str | None = Field(default=None, description="Optional experiment ID")

    def model_post_init(self, __context: Any) -> None:
        if not self.run_id:
            self.run_id = self.cycle_id
        if not self.top_k:
            self.top_k = self.budget


class ActiveLearningCycleHistoryItem(BaseModel):
    """Longitudinal progression milestone tracking diminishing returns across cycles."""

    cycle_id: str = Field(description="Cycle ID")
    name: str = Field(description="Cycle name")
    dataset_version_before: str = Field(description="Input dataset version")
    dataset_version_after: str | None = Field(default=None, description="Output dataset version")
    model_version_before: str = Field(description="Input model version")
    model_version_after: str | None = Field(default=None, description="Retrained model version")
    samples_reviewed: int = Field(description="Number of human-reviewed samples")
    strategy: SelectionStrategy = Field(description="Selection strategy used")
    budget: int = Field(description="Configured review budget")
    map50_before: float | None = Field(default=None, description="Baseline mAP@50")
    map50_after: float | None = Field(default=None, description="Retrained mAP@50")
    delta_map50: float | None = Field(
        default=None, description="Empirical mAP@50 gain (after - before)"
    )
    created_at: str = Field(description="Cycle creation ISO timestamp")


class StoppingCriteriaConfig(BaseModel):
    """Researcher-configurable criteria for terminating active learning iterations."""

    max_cycles: int = Field(default=5, ge=1, le=50, description="Maximum iterations")
    budget_per_cycle: int = Field(
        default=50, ge=1, le=500, description="Samples reviewed per cycle"
    )
    target_map50: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Target performance threshold"
    )
    min_improvement_threshold: float = Field(
        default=0.005, ge=0.0, le=0.1, description="Minimum acceptable mAP delta before stopping"
    )


# ─── Backward-Compatibility Aliases ──────────────────────────────────
RankedSample = CandidateSampleDetail
ActiveLearningRun = ActiveLearningCycle


class ReviewDecisionRequest(BaseModel):
    """Payload for submitting a human review decision on a sample."""

    cycle_id: str | None = Field(default=None, description="Active learning cycle ID")
    run_id: str | None = Field(default=None, description="Legacy run ID")
    image_id: str = Field(description="Sample image ID")
    decision: ReviewDecisionType | None = Field(
        default=None, description="Review decision taxonomy"
    )
    status: ReviewStatus | None = Field(default=None, description="Legacy review status")
    reviewer_id: str = Field(default="Researcher", description="Reviewer identity")
    ground_truth_class: str | None = Field(default=None, description="Corrected class")
    notes: str | None = Field(default=None, description="Reviewer notes")
    bbox_corrections: list[dict[str, Any]] = Field(default_factory=list)


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


class SelectionBiasReport(BaseModel):
    """Telemetry report measuring potential selection bias across recommended samples."""

    run_id: str = Field(description="Active learning run/cycle ID")
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

    dataset_id: str = Field(default="safety_v2", description="Target dataset ID")
    model_id: str = Field(default="yolo11s.pt", description="Target model ID")
    candidate_pool_id: str = Field(default="pool_01", description="Candidate pool ID")
    strategy_a: SelectionStrategy = Field(description="First selection strategy")
    strategy_b: SelectionStrategy = Field(description="Second selection strategy")
    top_k: int = Field(default=25, ge=1, le=200, description="Top-K sample count")


class StrategyComparisonResult(BaseModel):
    """Comparative analysis result between two selection strategies."""

    dataset_id: str = Field(description="Evaluated dataset ID")
    model_id: str = Field(description="Evaluated model ID")
    strategy_a: SelectionStrategy = Field(description="Strategy A")
    strategy_b: SelectionStrategy = Field(description="Strategy B")
    overlap_count: int = Field(
        description="Number of identical samples selected by both strategies"
    )
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
    retrained_evaluation_id: str = Field(
        description="Retrained evaluation ID on untouched test split (E1)"
    )
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
    active_learning_run_id: str = Field(
        description="Active learning run ID containing reviewed samples"
    )
    new_version_tag: str | None = Field(
        default=None, description="Optional new dataset version tag (e.g. v2.1)"
    )
