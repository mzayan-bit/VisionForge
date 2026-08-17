"""VisionForge Natural Language Query Interpreter for Visual Query Layer."""

import re
import uuid

from visionforge.events.schemas import EventType
from visionforge.query.schemas import (
    AggregationType,
    QueryStatus,
    QueryType,
    SortBy,
    SortOrder,
    VisualQuery,
)

# Known object class synonyms mapping
CLASS_SYNONYMS: dict[str, str] = {
    "person": "person",
    "people": "person",
    "human": "person",
    "humans": "person",
    "car": "car",
    "cars": "car",
    "vehicle": "car",
    "vehicles": "car",
    "helmet": "helmet",
    "helmets": "helmet",
    "truck": "car",
}

# Known event phrase mappings
EVENT_PHRASE_MAP: list[tuple[list[str], EventType]] = [
    (["entered", "enter", "entering", "came into"], EventType.OBJECT_ENTERED_REGION),
    (["left", "exit", "exited", "exiting", "departed"], EventType.OBJECT_LEFT_REGION),
    (["dwelled", "dwell", "dwelling", "stayed", "remained", "waited"], EventType.OBJECT_DWELLED),
    (["stopped", "halted", "stationary", "parked"], EventType.OBJECT_STOPPED),
    (["moved", "moving", "displaced"], EventType.OBJECT_MOVED),
    (["became close", "near", "close", "approached"], EventType.OBJECTS_BECAME_CLOSE),
    (["moved apart", "separated", "split"], EventType.OBJECTS_MOVED_APART),
    (["started", "began", "appeared"], EventType.TRACK_STARTED),
    (["ended", "disappeared", "terminated"], EventType.TRACK_ENDED),
    (["count changed", "count increased", "count decreased"], EventType.OBJECT_COUNT_CHANGED),
]

# Known unsupported action/semantic intent patterns
UNSUPPORTED_PATTERNS: list[str] = [
    r"what (was|is) (the|a) (person|object|who) doing",
    r"action",
    r"emotion|angry|happy|sad",
    r"facial|face|identity of person",
    r"re-identification|reid",
    r"predict future|will happen",
    r"speech|talking|saying",
]


class InterpretationResult:
    """Wrapper holding interpreted query DSL and metadata."""

    def __init__(
        self,
        query: VisualQuery,
        status: QueryStatus = QueryStatus.SUCCESS,
        explanation: str = "",
    ):
        self.query = query
        self.status = status
        self.explanation = explanation


class QueryInterpreter:
    """Deterministic, rule-based natural language parser mapping questions to VisualQuery DSL."""

    def parse_query(
        self, text: str, run_id: str, active_region_names: list[str] | None = None
    ) -> InterpretationResult:
        """Parse natural language query string into VisualQuery DSL."""
        clean_text = text.strip()
        lower_text = clean_text.lower()
        active_regions = active_region_names or []

        # 1. Check for Unsupported Query Intent
        for pattern in UNSUPPORTED_PATTERNS:
            if re.search(pattern, lower_text):
                vq = VisualQuery(
                    query_id=f"vq_{uuid.uuid4().hex[:10]}",
                    run_id=run_id,
                    original_text=clean_text,
                )
                return InterpretationResult(
                    query=vq,
                    status=QueryStatus.UNSUPPORTED,
                    explanation=(
                        "Unsupported Query: Question requests deep action recognition, facial identification, or "
                        "semantic intent not available for this run. VisionForge currently supports querying "
                        "tracks, trajectories, spatial regions, counts, timestamps, and observable temporal events."
                    ),
                )

        # 2. Check for Ambiguous Region Query
        if ("in the zone" in lower_text or "in the region" in lower_text) and not any(
            r.lower() in lower_text for r in active_regions
        ):
            if len(active_regions) > 1:
                vq = VisualQuery(
                    query_id=f"vq_{uuid.uuid4().hex[:10]}",
                    run_id=run_id,
                    original_text=clean_text,
                )
                reg_list = ", ".join(f"'{r}'" for r in active_regions)
                return InterpretationResult(
                    query=vq,
                    status=QueryStatus.AMBIGUOUS,
                    explanation=f"Ambiguous Query: Multiple active regions exist ({reg_list}). Please specify which region ROI to search.",
                )

        # Initialize base Structured Query
        vq = VisualQuery(
            query_id=f"vq_{uuid.uuid4().hex[:10]}",
            run_id=run_id,
            original_text=clean_text,
        )

        explanation_parts: list[str] = []

        # 3. Extract Track ID filter (e.g. "Track 4", "Track #4", "track 7")
        track_match = re.search(r"track\s*#?\s*(\d+)", lower_text)
        if track_match:
            vq.track_id = int(track_match.group(1))
            explanation_parts.append(f"Track ID: #{vq.track_id}")

        # 4. Extract Object Class filter (e.g. "person", "car", "people")
        for word, mapped_cls in CLASS_SYNONYMS.items():
            if re.search(r"\b" + re.escape(word) + r"\b", lower_text):
                vq.object_class = mapped_cls
                explanation_parts.append(f"Class: '{mapped_cls}'")
                break

        # 5. Extract Region Name filter
        matched_region: str | None = None
        for r_name in active_regions:
            if r_name.lower() in lower_text:
                matched_region = r_name
                break
        if not matched_region:
            # Fallback regex for Zone A / Zone B / Loading Zone
            zone_match = re.search(
                r"(zone\s+[a-z0-9_]+|loading zone|restricted corridor)", lower_text
            )
            if zone_match:
                matched_region = zone_match.group(1).title()

        if matched_region:
            vq.region_name = matched_region
            explanation_parts.append(f"Region: '{matched_region}'")

        # 6. Extract Event Type filter
        for phrases, evt_enum in EVENT_PHRASE_MAP:
            if any(p in lower_text for p in phrases):
                vq.event_type = evt_enum.value
                explanation_parts.append(f"Event: '{evt_enum.value}'")
                break

        # 7. Extract Exact Timestamp target (e.g. "at 10 seconds", "at 10s", "at 15.5s")
        at_match = re.search(r"at\s+(\d+(?:\.\d+)?)\s*(?:seconds|sec|s\b)", lower_text)
        if at_match:
            vq.at_timestamp_sec = float(at_match.group(1))
            explanation_parts.append(f"At Timestamp: {vq.at_timestamp_sec}s")

        # 8. Extract Time Range window (e.g. "between 5 and 15 seconds", "from 5s to 15s")
        range_match = re.search(
            r"(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)\s*(?:seconds|sec|s\b)",
            lower_text,
        )
        if range_match:
            vq.time_range = [float(range_match.group(1)), float(range_match.group(2))]
            explanation_parts.append(f"Time Range: [{vq.time_range[0]}s, {vq.time_range[1]}s]")
        elif "after" in lower_text:
            after_match = re.search(r"after\s+(\d+(?:\.\d+)?)\s*(?:seconds|sec|s\b)", lower_text)
            if after_match:
                vq.time_range = [float(after_match.group(1)), 9999.0]
                explanation_parts.append(f"Time Range: >{vq.time_range[0]}s")
        elif "before" in lower_text:
            before_match = re.search(r"before\s+(\d+(?:\.\d+)?)\s*(?:seconds|sec|s\b)", lower_text)
            if before_match:
                vq.time_range = [0.0, float(before_match.group(1))]
                explanation_parts.append(f"Time Range: <{vq.time_range[1]}s")

        # 9. Extract Duration Threshold filter (e.g. "longer than 5 seconds", "more than 5s")
        dur_match = re.search(
            r"(?:longer than|more than|exceeding|>\s+|>)\s*(\d+(?:\.\d+)?)\s*(?:seconds|sec|s\b)",
            lower_text,
        )
        if dur_match:
            vq.min_duration_sec = float(dur_match.group(1))
            explanation_parts.append(f"Min Duration: >{vq.min_duration_sec}s")

        # 10. Infer Query Type & Aggregations
        if "how many" in lower_text or "count" in lower_text:
            if (
                vq.at_timestamp_sec is not None
                or "visible" in lower_text
                or "present" in lower_text
            ):
                vq.query_type = QueryType.OBJECT_COUNT
                vq.aggregation = AggregationType.COUNT
            else:
                vq.query_type = QueryType.EVENT_SEARCH
                vq.aggregation = AggregationType.COUNT
        elif "longest" in lower_text:
            vq.aggregation = AggregationType.MAX
            vq.sort_by = SortBy.DURATION
            vq.sort_order = SortOrder.DESC
            if vq.event_type:
                vq.query_type = QueryType.EVENT_AGGREGATION
            else:
                vq.query_type = QueryType.TRACK_AGGREGATION
            explanation_parts.append("Aggregation: MAX (Longest Duration)")
        elif "most" in lower_text or "highest" in lower_text:
            vq.aggregation = AggregationType.MAX
            vq.sort_by = SortBy.COUNT
            vq.sort_order = SortOrder.DESC
            vq.query_type = QueryType.TRACK_AGGREGATION
            explanation_parts.append("Aggregation: MAX (Highest Count)")
        elif vq.time_range and not vq.event_type:
            vq.query_type = QueryType.TIME_RANGE_SEARCH
        elif vq.region_name and not vq.event_type:
            vq.query_type = QueryType.REGION_SEARCH
        elif vq.object_class and not vq.event_type:
            vq.query_type = QueryType.TRACK_SEARCH
        elif vq.track_id is not None:
            vq.query_type = QueryType.EVENT_SEARCH
        else:
            vq.query_type = QueryType.EVENT_SEARCH

        explanation_str = (
            f"Parsed Intent: {vq.query_type.value} | " + ", ".join(explanation_parts)
            if explanation_parts
            else f"Parsed Intent: {vq.query_type.value}"
        )

        return InterpretationResult(
            query=vq,
            status=QueryStatus.SUCCESS,
            explanation=explanation_str,
        )
