"""VisionForge Multimodal Vision-Language Layer Data Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class VisionQueryType(StrEnum):
    """Core supported query categories across the multimodal system."""

    IMAGE_QUERY = "IMAGE_QUERY"
    VIDEO_QUERY = "VIDEO_QUERY"
    OBJECT_QUERY = "OBJECT_QUERY"
    TRACK_QUERY = "TRACK_QUERY"
    EVENT_QUERY = "EVENT_QUERY"
    DATASET_QUERY = "DATASET_QUERY"
    MODEL_QUERY = "MODEL_QUERY"
    FAILURE_QUERY = "FAILURE_QUERY"
    SEARCH_QUERY = "SEARCH_QUERY"
    EVALUATION_QUERY = "EVALUATION_QUERY"


class EvidenceType(StrEnum):
    """Visual and structural evidence artifact classifications."""

    IMAGE_SAMPLE = "IMAGE_SAMPLE"
    DETECTION_BBOX = "DETECTION_BBOX"
    VIDEO_TIMESTAMP = "VIDEO_TIMESTAMP"
    TRACK_TRAJECTORY = "TRACK_TRAJECTORY"
    TEMPORAL_EVENT = "TEMPORAL_EVENT"
    FAILURE_SAMPLE = "FAILURE_SAMPLE"
    DATASET_PROFILE = "DATASET_PROFILE"
    MODEL_EVALUATION = "MODEL_EVALUATION"
    ATTRIBUTION_MAP = "ATTRIBUTION_MAP"
    SIMILAR_SAMPLE = "SIMILAR_SAMPLE"


class MultimodalQueryStatus(StrEnum):
    """Execution status of multimodal vision query."""

    SUCCESS = "SUCCESS"
    AMBIGUOUS = "AMBIGUOUS"
    NO_RESULTS = "NO_RESULTS"
    UNSUPPORTED = "UNSUPPORTED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"


class VisionEvidenceItem(BaseModel):
    """Structured evidence linking factual claims directly to visual artifacts."""

    evidence_id: str = Field(description="Unique evidence identifier ('evi_...')")
    evidence_type: EvidenceType = Field(description="Evidence classification category")
    title: str = Field(description="Human-readable title (e.g. 'Sample #1024 [Helmet FP]')")
    description: str = Field(description="Detailed verification summary of the evidence item")
    thumbnail_uri: str | None = Field(default=None, description="Visual preview image path or URI")
    sample_id: str | None = Field(default=None, description="Dataset sample ID if applicable")
    dataset_id: str | None = Field(default=None, description="Associated dataset ID")
    model_id: str | None = Field(default=None, description="Associated model checkpoint ID")
    video_id: str | None = Field(default=None, description="Associated video asset ID")
    timestamp_sec: float | None = Field(default=None, description="Video timestamp in seconds")
    frame_idx: int | None = Field(default=None, description="Video frame index")
    track_id: int | None = Field(default=None, description="Multi-object Track ID")
    event_id: str | None = Field(default=None, description="Temporal Event ID")
    bbox: list[float] | None = Field(
        default=None, description="Bounding box [x_min, y_min, x_max, y_max]"
    )
    confidence: float | None = Field(default=None, description="Model prediction confidence score")
    class_name: str | None = Field(default=None, description="Object class label")
    iou: float | None = Field(default=None, description="IoU with ground truth if failure analysis")
    action_link: str = Field(
        description="Direct UI deep link to verify evidence (e.g. '/video-lab?seek=12.4')"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional structured parameters"
    )


class MultiTurnContext(BaseModel):
    """Scoped session context maintaining active selection and conversational memory."""

    session_id: str = Field(description="Session identifier")
    selected_dataset: str | None = Field(default=None, description="Currently selected dataset ID")
    selected_model: str | None = Field(default=None, description="Currently selected model ID")
    selected_video: str | None = Field(
        default=None, description="Currently selected video asset ID"
    )
    selected_image: str | None = Field(
        default=None, description="Currently selected sample/image ID"
    )
    selected_time_range: list[float] | None = Field(
        default=None, description="Active video time window [start, end]"
    )
    previous_query: dict[str, Any] | None = Field(
        default=None, description="Previous structured query parameters"
    )
    history_turns: list[dict[str, Any]] = Field(
        default_factory=list, description="Recent conversation turns"
    )


class VisionQuery(BaseModel):
    """Comprehensive Multimodal Vision-Language Query descriptor."""

    query_id: str = Field(description="Unique query identifier ('vq_...')")
    user_query: str = Field(description="Original user natural language question")
    query_type: VisionQueryType = Field(description="Resolved query category")
    target: dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved target entities (dataset, model, video, sample_id, etc.)",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved query constraints (class_name, confidence, time_range, region, etc.)",
    )
    structured_query: dict[str, Any] = Field(
        default_factory=dict, description="Executable structured query DSL parameters"
    )
    execution_result: dict[str, Any] = Field(
        default_factory=dict, description="Raw structured results retrieved from domain services"
    )
    answer: str = Field(
        description="Factually grounded natural language answer summarizing actual vision data"
    )
    evidence: list[VisionEvidenceItem] = Field(
        default_factory=list, description="List of linked visual evidence references"
    )
    status: MultimodalQueryStatus = Field(
        default=MultimodalQueryStatus.SUCCESS, description="Execution and resolution status"
    )
    clarification_needed: str | None = Field(
        default=None, description="Question asked to user when query is ambiguous"
    )
    clarification_options: list[str] | None = Field(
        default=None, description="Available selectable options to resolve ambiguity"
    )
    grounding_verified: bool = Field(
        default=True,
        description="Whether all claims in answer have been verified against execution results",
    )
    reproducibility_hash: str = Field(
        default="", description="Deterministic cryptographic execution hash"
    )
    execution_time_ms: float = Field(
        default=0.0, description="Total execution latency in milliseconds"
    )
    created_timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Execution ISO timestamp"
    )


class VisionQueryHistoryItem(BaseModel):
    """Historical summary record for fast timeline retrieval."""

    query_id: str = Field(description="Unique query identifier")
    user_query: str = Field(description="Original question string")
    query_type: VisionQueryType = Field(description="Query category")
    status: MultimodalQueryStatus = Field(description="Execution status")
    results_count: int = Field(description="Number of matching records or evidence items")
    created_timestamp: str = Field(description="Execution ISO timestamp")
    execution_time_ms: float = Field(description="Total latency in ms")


class SuggestedQueryItem(BaseModel):
    """Context-aware suggested query recommendation for UI."""

    text: str = Field(description="Suggested question text")
    query_type: VisionQueryType = Field(description="Query type")
    page_context: str = Field(
        description="Target workspace page (e.g. 'failure_gallery', 'video_lab')"
    )
