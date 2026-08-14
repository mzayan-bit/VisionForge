"""VisionForge Query Executor for Visual Query Layer."""

import hashlib
import json
import logging
import time
from typing import Any

from visionforge.events.service import get_temporal_event_service
from visionforge.query.schemas import (
    AggregationType,
    QueryEvidenceItem,
    QueryResult,
    QueryStatus,
    QueryType,
    ResultType,
    SortOrder,
    VisualQuery,
)
from visionforge.video.schemas import Track
from visionforge.video.service import get_video_intelligence_service

logger = logging.getLogger("visionforge.query.executor")


class QueryExecutor:
    """Deterministic Query Executor operating over pre-computed video tracks, events, and regions."""

    def execute_query(
        self, query: VisualQuery, original_text: str, interpretation_explanation: str, interp_time_ms: float
    ) -> QueryResult:
        """Execute structured VisualQuery DSL against stored facts and return evidence-backed QueryResult."""
        t_start = time.perf_counter()

        video_svc = get_video_intelligence_service()
        event_svc = get_temporal_event_service()

        run = video_svc.get_run(query.run_id)
        events = event_svc.get_events_for_run(query.run_id)

        records: list[dict[str, Any]] = []
        evidence: list[QueryEvidenceItem] = []
        res_type = ResultType.EVENT_LIST
        summary_text = ""

        # Dispatch execution by QueryType
        if query.query_type == QueryType.OBJECT_COUNT:
            res_type = ResultType.COUNT_METRIC
            records, evidence, summary_text = self._execute_object_count(query, run)

        elif query.query_type in (QueryType.TRACK_SEARCH, QueryType.TRACK_AGGREGATION):
            res_type = ResultType.TRACK_LIST
            records, evidence, summary_text = self._execute_track_query(query, run, events)

        elif query.query_type in (QueryType.EVENT_SEARCH, QueryType.EVENT_AGGREGATION, QueryType.TIME_RANGE_SEARCH, QueryType.REGION_SEARCH):
            res_type = ResultType.EVENT_LIST
            records, evidence, summary_text = self._execute_event_query(query, run, events)

        else:
            res_type = ResultType.EVENT_LIST
            records, evidence, summary_text = self._execute_event_query(query, run, events)

        # Apply Sorting & Limit
        records = self._apply_sorting_and_limit(records, query)

        exec_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        total_time_ms = round(interp_time_ms + exec_time_ms, 2)

        # Compute Deterministic Reproducibility Hash
        repr_data = {
            "run_id": query.run_id,
            "query_type": query.query_type.value,
            "records_count": len(records),
        }
        repro_hash = hashlib.sha256(json.dumps(repr_data, sort_keys=True).encode()).hexdigest()[:16]

        return QueryResult(
            query_id=query.query_id,
            original_query=original_text,
            structured_query=query,
            status=QueryStatus.SUCCESS,
            result_type=res_type,
            records=records,
            summary=summary_text,
            evidence=evidence,
            interpretation_explanation=interpretation_explanation,
            interpretation_time_ms=interp_time_ms,
            execution_time_ms=exec_time_ms,
            total_query_time_ms=total_time_ms,
            source_run_id=query.run_id,
            reproducibility_hash=repro_hash,
        )

    def _execute_object_count(
        self, query: VisualQuery, run: Any
    ) -> tuple[list[dict[str, Any]], list[QueryEvidenceItem], str]:
        ts = query.at_timestamp_sec if query.at_timestamp_sec is not None else 0.0

        # Filter active tracks at target timestamp
        active_tracks: list[Track] = [
            t
            for t in run.tracks
            if t.first_timestamp_sec <= ts <= t.last_timestamp_sec + 0.5
        ]

        if query.object_class:
            active_tracks = [t for t in active_tracks if t.class_name.lower() == query.object_class.lower()]

        cnt = len(active_tracks)
        cls_str = f"'{query.object_class}' " if query.object_class else ""
        summary = f"{cnt} active {cls_str}object(s) visible at t={ts:.1f}s in run '{run.run_id}'."

        records = [
            {
                "timestamp_sec": ts,
                "active_count": cnt,
                "object_class": query.object_class or "all",
                "tracks": [t.model_dump() for t in active_tracks],
            }
        ]

        evidence = [
            QueryEvidenceItem(
                track_id=t.track_id,
                timestamp_sec=ts,
                frame_idx=int(ts * run.sampling_config.total_sampled_frames / max(1.0, run.duration_sec)),
                description=f"Track #{t.track_id} ({t.class_name}) visible at t={ts:.1f}s.",
                action_link=f"/video-lab?seek={ts:.1f}&track={t.track_id}",
            )
            for t in active_tracks
        ]

        return records, evidence, summary

    def _execute_track_query(
        self, query: VisualQuery, run: Any, events: list[Any]
    ) -> tuple[list[dict[str, Any]], list[QueryEvidenceItem], str]:
        matching_tracks = list(run.tracks)

        if query.object_class:
            matching_tracks = [t for t in matching_tracks if t.class_name.lower() == query.object_class.lower()]

        if query.track_id is not None:
            matching_tracks = [t for t in matching_tracks if t.track_id == query.track_id]

        if query.min_duration_sec is not None:
            matching_tracks = [t for t in matching_tracks if t.visibility_duration_sec >= query.min_duration_sec]

        if query.min_confidence is not None:
            matching_tracks = [t for t in matching_tracks if t.avg_confidence >= query.min_confidence]

        if query.region_name:
            # Find tracks involved in events for that region
            reg_evts = [e for e in events if query.region_name.lower() in e.description.lower()]
            reg_track_ids = set()
            for e in reg_evts:
                reg_track_ids.update(e.source_track_ids)
            matching_tracks = [t for t in matching_tracks if t.track_id in reg_track_ids]

        records = [t.model_dump() for t in matching_tracks]

        evidence = [
            QueryEvidenceItem(
                track_id=t.track_id,
                timestamp_sec=t.first_timestamp_sec,
                frame_idx=t.first_frame,
                description=f"Track #{t.track_id} ({t.class_name}): duration {t.visibility_duration_sec:.1f}s, distance {t.total_distance_px:.0f}px.",
                action_link=f"/video-lab?seek={t.first_timestamp_sec:.1f}&track={t.track_id}",
            )
            for t in matching_tracks
        ]

        cls_str = f"'{query.object_class}' " if query.object_class else ""
        dur_str = f" (duration ≥ {query.min_duration_sec}s)" if query.min_duration_sec else ""
        reg_str = f" in region '{query.region_name}'" if query.region_name else ""

        if query.aggregation == AggregationType.MAX and matching_tracks:
            longest_track = max(matching_tracks, key=lambda t: t.visibility_duration_sec)
            summary = (
                f"Longest {cls_str}track: Track #{longest_track.track_id} ({longest_track.class_name}) "
                f"stayed for {longest_track.visibility_duration_sec:.1f}s{reg_str}."
            )
        else:
            summary = f"Found {len(matching_tracks)} matching {cls_str}track(s){dur_str}{reg_str}."

        return records, evidence, summary

    def _execute_event_query(
        self, query: VisualQuery, run: Any, events: list[Any]
    ) -> tuple[list[dict[str, Any]], list[QueryEvidenceItem], str]:
        matching_events = list(events)

        if query.event_type:
            matching_events = [
                e for e in matching_events if e.event_type.value.lower() == query.event_type.lower()
            ]

        if query.track_id is not None:
            matching_events = [e for e in matching_events if query.track_id in e.source_track_ids]

        if query.region_name:
            matching_events = [
                e
                for e in matching_events
                if query.region_name.lower() in e.description.lower()
                or query.region_name.lower() == e.event_params.get("region_name", "").lower()
            ]

        if query.object_class:
            matching_events = [
                e
                for e in matching_events
                if query.object_class.lower() == e.event_params.get("class_name", "").lower()
            ]

        if query.time_range:
            start_w, end_w = query.time_range
            matching_events = [
                e for e in matching_events if start_w <= e.start_timestamp_sec <= end_w
            ]

        if query.min_duration_sec is not None:
            matching_events = [e for e in matching_events if e.duration_sec >= query.min_duration_sec]

        records = [e.model_dump() for e in matching_events]

        evidence = [
            QueryEvidenceItem(
                event_id=e.event_id,
                track_id=e.source_track_ids[0] if e.source_track_ids else None,
                timestamp_sec=e.start_timestamp_sec,
                frame_idx=e.source_frame_range[0] if e.source_frame_range else 0,
                region_id=e.event_params.get("region_id"),
                description=e.description,
                action_link=f"/video-lab?seek={e.start_timestamp_sec:.1f}&event={e.event_id}",
            )
            for e in matching_events
        ]

        evt_str = f"'{query.event_type}' " if query.event_type else ""
        reg_str = f" in region '{query.region_name}'" if query.region_name else ""
        dur_str = f" (duration ≥ {query.min_duration_sec}s)" if query.min_duration_sec else ""

        if query.aggregation == AggregationType.MAX and matching_events:
            longest_evt = max(matching_events, key=lambda e: e.duration_sec)
            summary = (
                f"Longest {evt_str}event: {longest_evt.description} "
                f"(Duration: {longest_evt.duration_sec:.1f}s)."
            )
        else:
            summary = f"Found {len(matching_events)} matching {evt_str}event(s){reg_str}{dur_str}."

        return records, evidence, summary

    def _apply_sorting_and_limit(self, records: list[dict[str, Any]], query: VisualQuery) -> list[dict[str, Any]]:
        if not records:
            return []

        reverse = query.sort_order == SortOrder.DESC
        sort_key = query.sort_by.value

        def extract_val(r: dict[str, Any]) -> float:
            if sort_key in r:
                return float(r[sort_key])
            if sort_key == "timestamp" and "start_timestamp_sec" in r:
                return float(r["start_timestamp_sec"])
            if sort_key == "timestamp" and "first_timestamp_sec" in r:
                return float(r["first_timestamp_sec"])
            if sort_key == "duration" and "visibility_duration_sec" in r:
                return float(r["visibility_duration_sec"])
            return 0.0

        try:
            records.sort(key=extract_val, reverse=reverse)
        except Exception:
            pass

        return records[: query.limit]
