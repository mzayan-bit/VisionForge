"""VisionForge Unified Visual Search Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from visionforge.search.similarity import SimilarityMetric


class VisualAssetType(StrEnum):
    """Classification of searchable visual assets across VisionForge."""

    IMAGE = "IMAGE"
    FRAME = "FRAME"
    OBJECT_CROP = "OBJECT_CROP"
    DATASET_SAMPLE = "DATASET_SAMPLE"
    EVENT_FRAME = "EVENT_FRAME"


class VisualAsset(BaseModel):
    """Unified representation of a searchable visual asset in VisionForge."""

    asset_id: str = Field(description="Unique asset identifier")
    asset_type: VisualAssetType = Field(description="Category of the visual asset")
    title: str = Field(description="Human-readable title or label")
    embedding_id: str = Field(description="Corresponding record ID in VisualMemoryIndex")
    embedding_model: str = Field(
        default="siglip-base-patch16-224", description="Embedding model used to encode asset"
    )
    embedding_version: str = Field(
        default="1.0.0", description="Embedding model architecture version"
    )
    source_video_id: str | None = Field(
        default=None, description="Video asset ID if frame/object/event"
    )
    source_dataset_id: str | None = Field(default=None, description="Dataset ID if dataset sample")
    source_run_id: str | None = Field(
        default=None, description="Inference run ID if video detection/track"
    )
    source_event_id: str | None = Field(
        default=None, description="Event ID if event moment/evidence"
    )
    timestamp_sec: float | None = Field(
        default=None, description="Timestamp in video timeline in seconds"
    )
    frame_idx: int | None = Field(default=None, description="Frame index in video sequence")
    track_id: int | None = Field(default=None, description="Persistent Track ID if object crop")
    bbox: list[float] | None = Field(
        default=None, description="Bounding box [x1, y1, x2, y2] in pixels if object crop"
    )
    class_name: str | None = Field(default=None, description="Class label if classified/detected")
    thumbnail_url: str | None = Field(
        default=None, description="Visual thumbnail URI or base64 preview"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional arbitrary metadata"
    )
    indexed_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 indexing timestamp",
    )


class UnifiedSearchRequest(BaseModel):
    """Unified query request supporting image, frame, object crop, event moment, or dataset sample."""

    query_type: VisualAssetType | str = Field(
        default=VisualAssetType.IMAGE, description="Source type of the query asset"
    )
    asset_id: str | None = Field(
        default=None, description="Existing VisualAsset ID to use as query"
    )
    video_id: str | None = Field(
        default=None, description="Target video asset ID if querying by frame"
    )
    timestamp_sec: float | None = Field(
        default=None, description="Target timestamp in seconds if querying by frame/moment"
    )
    run_id: str | None = Field(
        default=None, description="Target inference run ID if querying by track/object"
    )
    track_id: int | None = Field(
        default=None, description="Target track ID if querying by object crop"
    )
    event_id: str | None = Field(
        default=None, description="Target event ID if querying by event moment"
    )
    dataset_id: str | None = Field(
        default=None, description="Target dataset ID if querying by dataset sample"
    )
    sample_id: str | None = Field(default=None, description="Target sample ID in dataset")
    vector: list[float] | None = Field(default=None, description="Direct dense query vector (768D)")
    filter_asset_types: list[VisualAssetType] | None = Field(
        default=None, description="Filter candidate results by asset types"
    )
    filter_dataset_id: str | None = Field(
        default=None, description="Filter candidate results to specific dataset ID"
    )
    filter_video_id: str | None = Field(
        default=None, description="Filter candidate results to specific video ID"
    )
    filter_class_name: str | None = Field(
        default=None, description="Filter candidate results to specific class name"
    )
    filter_event_type: str | None = Field(
        default=None, description="Filter candidate results to specific event type"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Top-K maximum results to return")
    threshold: float = Field(
        default=0.0, ge=0.0, le=0.98, description="Minimum similarity score threshold [0.0, 0.98]"
    )
    metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Distance/similarity metric"
    )


class UnifiedSearchResultItem(BaseModel):
    """Ranked visual search match with complete source traceability."""

    rank: int = Field(description="1-indexed similarity rank order")
    asset: VisualAsset = Field(description="Matched visual asset details")
    similarity_score: float = Field(description="Calculated similarity score in range [0.0, 1.0]")
    distance: float = Field(description="Calculated vector distance")
    source_traceability: dict[str, Any] = Field(
        default_factory=dict,
        description="Source provenance metadata (dataset, video, run, event, frame)",
    )
    action_link: str = Field(
        description="Frontend navigation link (e.g. '/video-lab?seek=14.2&track=17')"
    )
    evidence_notes: str = Field(description="Transparent description of matching provenance")


class UnifiedSearchResponse(BaseModel):
    """Complete response document from unified visual search execution."""

    search_id: str = Field(description="Unique transaction ID for this search execution")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO query execution timestamp",
    )
    query_summary: str = Field(description="Human-readable description of the query input")
    query_asset: VisualAsset | None = Field(
        default=None, description="Resolved query asset details"
    )
    results: list[UnifiedSearchResultItem] = Field(description="Ranked top-K visual asset matches")
    candidate_count: int = Field(description="Total visual memory assets evaluated")
    returned_count: int = Field(description="Total matches satisfying filters and cutoff threshold")
    metric_used: SimilarityMetric = Field(description="Distance metric used for ranking")
    model_used: str = Field(
        default="siglip-base-patch16-224", description="Embedding model used for query"
    )
    embedding_time_ms: float = Field(
        default=0.0, description="Embedding extraction duration in milliseconds"
    )
    search_time_ms: float = Field(
        default=0.0, description="Vector similarity search duration in milliseconds"
    )
    filtering_time_ms: float = Field(
        default=0.0, description="Metadata filtering duration in milliseconds"
    )
    total_execution_time_ms: float = Field(
        description="Total end-to-end search duration in milliseconds"
    )
    explanation: str = Field(
        default="Ranked by dense embedding similarity (SigLIP-base-patch16-224). Metric: Cosine.",
        description="Transparent explanation of ranking methodology",
    )


class NearDuplicatePair(BaseModel):
    """Discovered near-duplicate visual asset candidate pair."""

    asset_a: VisualAsset = Field(description="First asset in candidate pair")
    asset_b: VisualAsset = Field(description="Second asset in candidate pair")
    similarity_score: float = Field(description="Pairwise cosine similarity score")
    distance: float = Field(description="Pairwise vector distance")
    recommendation: str = Field(description="Actionable quality guidance for researcher")


class NearDuplicateResponse(BaseModel):
    """Response document for near-duplicate candidate discovery."""

    total_evaluated: int = Field(description="Number of assets evaluated for near-duplicates")
    duplicate_pairs_found: int = Field(
        description="Number of candidate near-duplicate pairs identified"
    )
    pairs: list[NearDuplicatePair] = Field(description="List of candidate near-duplicate pairs")
    threshold_used: float = Field(description="Cosine similarity cutoff threshold used (e.g. 0.95)")
    execution_time_ms: float = Field(description="Total discovery latency in milliseconds")
