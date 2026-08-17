"""VisionForge Visual Query Layer Data Schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class QueryType(StrEnum):
    """Supported deterministic visual query types."""

    EVENT_SEARCH = "EVENT_SEARCH"
    TRACK_SEARCH = "TRACK_SEARCH"
    OBJECT_COUNT = "OBJECT_COUNT"
    TRACK_AGGREGATION = "TRACK_AGGREGATION"
    EVENT_AGGREGATION = "EVENT_AGGREGATION"
    TIME_RANGE_SEARCH = "TIME_RANGE_SEARCH"
    REGION_SEARCH = "REGION_SEARCH"


class AggregationType(StrEnum):
    """Aggregation functions supported over query results."""

    COUNT = "COUNT"
    MIN = "MIN"
    MAX = "MAX"
    AVERAGE = "AVERAGE"
    SUM = "SUM"


class SortBy(StrEnum):
    """Field attributes available for sorting query results."""

    DURATION = "duration"
    CONFIDENCE = "confidence"
    TIMESTAMP = "timestamp"
    DISTANCE = "distance"
    COUNT = "count"


class SortOrder(StrEnum):
    """Result sorting direction."""

    ASC = "ASC"
    DESC = "DESC"


class QueryStatus(StrEnum):
    """Execution status of a visual query."""

    SUCCESS = "SUCCESS"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class ResultType(StrEnum):
    """Data shape of returned query records."""

    EVENT_LIST = "EVENT_LIST"
    TRACK_LIST = "TRACK_LIST"
    COUNT_METRIC = "COUNT_METRIC"
    AGGREGATION_METRIC = "AGGREGATION_METRIC"
    TIMELINE_LIST = "TIMELINE_LIST"


class VisualQuery(BaseModel):
    """Internal deterministic Structured Query DSL representation."""

    query_id: str = Field(description="Unique query identifier ('vq_...')")
    run_id: str = Field(description="Target VideoInferenceRun ID")
    video_id: str | None = Field(default=None, description="Target Video Asset ID")
    query_type: QueryType = Field(
        default=QueryType.EVENT_SEARCH, description="Target query execution type"
    )
    event_type: str | None = Field(
        default=None, description="Optional EventType filter (e.g. 'OBJECT_ENTERED_REGION')"
    )
    track_id: int | None = Field(default=None, description="Optional persistent Track ID filter")
    object_class: str | None = Field(
        default=None, description="Optional object class name filter (e.g. 'person', 'car')"
    )
    region_name: str | None = Field(
        default=None, description="Optional Region ROI name filter (e.g. 'Loading Zone A')"
    )
    region_id: str | None = Field(default=None, description="Optional Region ROI ID filter")
    time_range: list[float] | None = Field(
        default=None, description="[start_sec, end_sec] timestamp window filter"
    )
    at_timestamp_sec: float | None = Field(
        default=None, description="Exact timestamp target (e.g. 10.0 for 'at 10 seconds')"
    )
    min_duration_sec: float | None = Field(
        default=None, description="Minimum duration threshold filter in seconds"
    )
    max_duration_sec: float | None = Field(
        default=None, description="Maximum duration threshold filter in seconds"
    )
    min_confidence: float | None = Field(
        default=None, description="Minimum confidence threshold [0.0, 1.0]"
    )
    aggregation: AggregationType | None = Field(
        default=None, description="Aggregation function if requested"
    )
    sort_by: SortBy = Field(default=SortBy.TIMESTAMP, description="Result sorting attribute")
    sort_order: SortOrder = Field(default=SortOrder.ASC, description="Result sorting direction")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum matching records limit")
    original_text: str | None = Field(
        default=None, description="Original natural language query string"
    )


class QueryEvidenceItem(BaseModel):
    """Visual evidence reference connecting query results to video player frames."""

    event_id: str | None = Field(default=None, description="Associated event ID if applicable")
    track_id: int | None = Field(default=None, description="Associated Track ID if applicable")
    timestamp_sec: float = Field(description="Exact timestamp in video timeline (seconds)")
    frame_idx: int = Field(description="Corresponding frame index in video")
    region_id: str | None = Field(
        default=None, description="Associated Region ROI ID if applicable"
    )
    description: str = Field(description="Visual evidence verification text")
    action_link: str = Field(
        description="Frontend action link (e.g. '/video-lab?seek=5.2&track=4')"
    )


class QueryResult(BaseModel):
    """Complete, evidence-backed query result document."""

    query_id: str = Field(description="Unique query identifier ('vq_...')")
    original_query: str = Field(description="Original user question string")
    structured_query: VisualQuery = Field(
        description="Internal structured query DSL representation"
    )
    status: QueryStatus = Field(description="Execution status")
    result_type: ResultType = Field(description="Result record category")
    records: list[dict[str, Any]] = Field(
        default_factory=list, description="Matching structured data records"
    )
    summary: str = Field(
        description="Evidence-backed human-readable answer generated ONLY from records"
    )
    evidence: list[QueryEvidenceItem] = Field(
        default_factory=list, description="Visual evidence links to video frames"
    )
    interpretation_explanation: str = Field(
        description="Transparent breakdown explaining how the query was interpreted"
    )
    interpretation_time_ms: float = Field(description="Parsing/interpretation latency in ms")
    execution_time_ms: float = Field(description="Query execution latency in ms")
    total_query_time_ms: float = Field(description="Total query latency in ms")
    source_run_id: str = Field(description="Target VideoInferenceRun ID")
    reproducibility_hash: str = Field(description="Deterministic query execution hash")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Execution ISO timestamp"
    )


class QueryHistoryItem(BaseModel):
    """Summary record stored in query history."""

    query_id: str = Field(description="Query ID")
    original_query: str = Field(description="User question string")
    query_type: QueryType = Field(description="Parsed query type")
    run_id: str = Field(description="Target run ID")
    status: QueryStatus = Field(description="Status")
    results_count: int = Field(description="Number of matching records")
    total_query_time_ms: float = Field(description="Execution duration ms")
    created_at: str = Field(description="ISO timestamp")
