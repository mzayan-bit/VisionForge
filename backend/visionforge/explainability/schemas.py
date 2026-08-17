"""VisionForge Model Explainability & Visual Diagnostics Schemas.

Defines domain models for diagnostic spatial attribution:
- ExplanationRun
- ExplanationConfig
- AttributionArtifact
- ExplanationComparison
- Human Review & Researcher Notes
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from visionforge.inference.schemas import StandardPrediction


class ExplanationMethod(StrEnum):
    """Supported visual explainability and attribution algorithms."""

    GRAD_CAM = "GRAD_CAM"
    LAYER_CAM = "LAYER_CAM"
    INTEGRATED_GRADIENTS = "INTEGRATED_GRADIENTS"
    ATTENTION_MAP = "ATTENTION_MAP"
    PERTURBATION = "PERTURBATION"


class ExplanationStatus(StrEnum):
    """Lifecycle status of an explanation generation execution."""

    QUEUED = "QUEUED"
    GENERATING = "GENERATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"


class ReviewRating(StrEnum):
    """Human researcher rating of explanation utility."""

    UNREVIEWED = "UNREVIEWED"
    USEFUL = "USEFUL"
    NOT_USEFUL = "NOT_USEFUL"
    UNCLEAR = "UNCLEAR"
    NEEDS_INVESTIGATION = "NEEDS_INVESTIGATION"


class ExplanationConfig(BaseModel):
    """Configuration options for attribution map generation."""

    method: ExplanationMethod = Field(
        default=ExplanationMethod.GRAD_CAM, description="Attribution algorithm"
    )
    target_layer: str | None = Field(
        default=None, description="Model layer name to extract activations/gradients from"
    )
    target_class: str | None = Field(default=None, description="Category class label to attribute")
    target_prediction_id: str | None = Field(
        default=None, description="Specific prediction object ID to explain"
    )
    colormap: str = Field(
        default="jet", description="Heatmap colormap palette ('jet', 'viridis', 'inferno')"
    )
    opacity: float = Field(default=0.55, ge=0.0, le=1.0, description="Overlay blending opacity")
    num_steps: int = Field(default=25, ge=5, le=100, description="Integration / perturbation steps")
    show_prediction_box: bool = Field(
        default=True, description="Whether to render prediction bounding box outline"
    )


class AttributionArtifact(BaseModel):
    """Output spatial attribution payload and summary statistics."""

    grid_width: int = Field(description="Heatmap grid column count (e.g. 32)")
    grid_height: int = Field(description="Heatmap grid row count (e.g. 32)")
    heatmap_grid: list[list[float]] = Field(
        description="2D normalized attribution intensity matrix in [0.0, 1.0]"
    )
    peak_intensity_coords: list[float] = Field(
        description="Normalized coordinates [x, y] of maximum attribution intensity"
    )
    mean_intensity: float = Field(description="Average attribution intensity across image")
    object_concentration_score: float = Field(
        description="Fraction of attribution mass concentrated inside target bounding box [0.0, 1.0]"
    )
    background_concentration_score: float = Field(
        description="Fraction of attribution mass falling outside target bounding box [0.0, 1.0]"
    )
    colormap: str = Field(default="jet", description="Color mapping applied")


class ExplanationRun(BaseModel):
    """Complete record of an explainability generation transaction."""

    explanation_id: str = Field(description="Unique explanation run ID ('exp_...')")
    model_id: str = Field(description="Evaluated model checkpoint")
    model_version: str = Field(default="1.0.0", description="Model version tag")
    inference_id: str | None = Field(default=None, description="Linked inference run ID")
    sample_id: str = Field(description="Evaluated image sample ID")
    image_path: str = Field(description="Path to evaluated image file")
    dataset_id: str = Field(default="safety_v2", description="Associated dataset ID")
    dataset_version: str = Field(default="v1.0.0", description="Dataset version tag")
    split: str = Field(default="test", description="Dataset split evaluated")
    method: ExplanationMethod = Field(description="Attribution method utilized")
    status: ExplanationStatus = Field(
        default=ExplanationStatus.QUEUED, description="Execution status"
    )
    config: ExplanationConfig = Field(default_factory=ExplanationConfig)
    target_class: str = Field(description="Class category being explained")
    prediction: StandardPrediction | None = Field(
        default=None, description="Associated object detection prediction"
    )
    ground_truth_class: str | None = Field(
        default=None, description="Actual ground truth class label"
    )
    is_correct_prediction: bool | None = Field(
        default=None, description="Whether prediction matches ground truth"
    )
    artifact: AttributionArtifact | None = Field(
        default=None, description="Generated attribution heatmap artifact"
    )
    diagnostic_summary: str = Field(
        default="", description="Diagnostic evidence summary using non-causal language"
    )
    disclaimer: str = Field(
        default="Attribution visualizations indicate image regions associated with the model's output. They should not be interpreted as definitive evidence of causal reasoning.",
        description="Mandatory scientific validity disclaimer",
    )
    review_rating: ReviewRating = Field(
        default=ReviewRating.UNREVIEWED, description="Human utility assessment"
    )
    researcher_notes: list[str] = Field(
        default_factory=list, description="Observations logged by human researcher"
    )
    cache_hit: bool = Field(
        default=False, description="Whether result was served from deterministic cache"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = Field(default=None)
    error_message: str | None = Field(
        default=None, description="Technical diagnostics if status is FAILED or UNSUPPORTED"
    )
    environment: dict[str, Any] = Field(default_factory=dict)


class ExplanationComparison(BaseModel):
    """Side-by-side diagnostic comparison between two explanation runs."""

    comparison_id: str = Field(description="Unique comparison ID ('cmp_exp_...')")
    explanation_a: ExplanationRun = Field(
        description="First explanation run (e.g. Sample A / Model A)"
    )
    explanation_b: ExplanationRun = Field(
        description="Second explanation run (e.g. Sample B / Model B)"
    )
    attribution_difference_score: float = Field(
        description="Cosine / Mean Absolute Difference between attribution maps"
    )
    attribution_difference_grid: list[list[float]] = Field(
        description="2D difference map |Attribution_A - Attribution_B|"
    )
    diagnostic_notes: list[str] = Field(
        default_factory=list, description="Comparative observation notes"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class CreateExplanationRequest(BaseModel):
    """Payload to initiate or retrieve an explanation."""

    model_id: str = Field(default="yolo11s.pt", description="Target model identifier")
    model_version: str = Field(default="1.0.0", description="Model version")
    sample_id: str = Field(default="img_0007", description="Image sample ID")
    image_path: str | None = Field(default=None, description="Path to image file")
    inference_id: str | None = Field(default=None, description="Source inference ID")
    target_class: str = Field(default="helmet", description="Target category label to attribute")
    method: ExplanationMethod = Field(default=ExplanationMethod.GRAD_CAM)
    config: ExplanationConfig = Field(default_factory=ExplanationConfig)
    ground_truth_class: str | None = Field(default=None)
    is_correct_prediction: bool | None = Field(default=None)


class ReviewExplanationRequest(BaseModel):
    """Payload to record human assessment on explanation utility."""

    rating: ReviewRating = Field(description="Review assessment rating")
    note: str | None = Field(default=None, description="Optional accompanying observation note")


class AddResearcherNoteRequest(BaseModel):
    """Payload to append a researcher observation note."""

    note: str = Field(description="Observation text logged by researcher")


class CompareExplanationsRequest(BaseModel):
    """Payload to compare two explanation runs side-by-side."""

    explanation_id_a: str = Field(description="First explanation ID")
    explanation_id_b: str = Field(description="Second explanation ID")
