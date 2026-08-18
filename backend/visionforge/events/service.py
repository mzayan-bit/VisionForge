"""VisionForge Temporal Event Intelligence Service."""

import json
import logging
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.events.detector import TemporalEventDetector
from visionforge.events.schemas import (
    CoordinateSystem,
    EventAnalytics,
    EventEvidence,
    EventRuleConfig,
    EventType,
    RegionOfInterest,
    RegionShape,
    SceneSummary,
    TemporalEvent,
)
from visionforge.video.service import get_video_intelligence_service

logger = logging.getLogger("visionforge.events.service")


class EventNotFoundError(VisionForgeException):
    """Raised when looking up an event ID that does not exist."""

    def __init__(self, event_id: str):
        super().__init__(
            message=f"Temporal event '{event_id}' was not found.",
            code="EVENT_NOT_FOUND",
            status_code=404,
        )


class RegionNotFoundError(VisionForgeException):
    """Raised when looking up a region ID that does not exist."""

    def __init__(self, region_id: str):
        super().__init__(
            message=f"Region of Interest '{region_id}' was not found.",
            code="REGION_NOT_FOUND",
            status_code=404,
        )


class TemporalEventService:
    """Service managing Region ROI definitions, event detection generation, evidence extraction, and analytics."""

    def __init__(self, storage_dir: Path | None = None):
        cache_root = Path(get_settings().model_cache_dir).expanduser().resolve()
        raw_path = storage_dir or (cache_root.parent / "events")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._regions_file = self._storage_dir / "regions.json"
        self._events_file = self._storage_dir / "events.json"

        self._is_custom_storage = storage_dir is not None
        self._regions: dict[str, RegionOfInterest] = {}
        self._events: dict[str, TemporalEvent] = {}
        self.load_from_disk()

    # ─── Region ROI Management ─────────────────────────────────────────

    def create_region(
        self,
        video_id: str,
        name: str,
        coordinates: list[list[float]],
        shape_type: RegionShape = RegionShape.RECTANGLE,
        coordinate_system: CoordinateSystem = CoordinateSystem.PIXEL,
        color: str = "#3b82f6",
    ) -> RegionOfInterest:
        """Create and store a Region of Interest for a video asset."""
        region_id = f"reg_{uuid.uuid4().hex[:10]}"
        region = RegionOfInterest(
            region_id=region_id,
            video_id=video_id,
            name=name,
            shape_type=shape_type,
            coordinates=coordinates,
            coordinate_system=coordinate_system,
            color=color,
        )
        self._regions[region_id] = region
        self.save_to_disk()
        logger.info(
            "Created Region of Interest '%s' (%s) for video '%s'", region_id, name, video_id
        )
        return region

    def get_region(self, region_id: str) -> RegionOfInterest:
        """Retrieve a specific Region of Interest."""
        if region_id not in self._regions:
            raise RegionNotFoundError(region_id)
        return self._regions[region_id]

    def update_region(
        self,
        region_id: str,
        name: str | None = None,
        coordinates: list[list[float]] | None = None,
        shape_type: RegionShape | None = None,
        color: str | None = None,
    ) -> RegionOfInterest:
        """Update properties or geometry of an existing Region of Interest."""
        if region_id not in self._regions:
            raise RegionNotFoundError(region_id)
        region = self._regions[region_id]

        if name is not None:
            region.name = name.strip()
        if coordinates is not None:
            region.coordinates = coordinates
        if shape_type is not None:
            region.shape_type = shape_type
        if color is not None:
            region.color = color

        self.save_to_disk()
        logger.info("Updated Region of Interest '%s' (%s)", region_id, region.name)
        return region

    def duplicate_region(self, region_id: str, offset_px: float = 30.0) -> RegionOfInterest:
        """Duplicate a region with a distinct ID, name, and coordinate offset."""
        source = self.get_region(region_id)
        new_coords: list[list[float]] = []

        if source.shape_type == RegionShape.RECTANGLE:
            if len(source.coordinates) == 2 and isinstance(source.coordinates[0], list):
                new_coords = [
                    [source.coordinates[0][0] + offset_px, source.coordinates[0][1] + offset_px],
                    [source.coordinates[1][0] + offset_px, source.coordinates[1][1] + offset_px],
                ]
            else:
                new_coords = [[pt[0] + offset_px, pt[1] + offset_px] for pt in source.coordinates]
        else:
            new_coords = [[pt[0] + offset_px, pt[1] + offset_px] for pt in source.coordinates]

        new_name = f"{source.name} (Copy)"
        return self.create_region(
            video_id=source.video_id,
            name=new_name,
            coordinates=new_coords,
            shape_type=source.shape_type,
            coordinate_system=source.coordinate_system,
            color=source.color,
        )

    def list_regions(self, video_id: str | None = None) -> list[RegionOfInterest]:
        """List all defined regions, optionally filtered by video ID."""
        regs = list(self._regions.values())
        if video_id:
            regs = [r for r in regs if r.video_id == video_id]
        return sorted(regs, key=lambda r: r.created_at, reverse=True)

    def delete_region(self, region_id: str) -> None:
        """Delete a region of interest."""
        if region_id not in self._regions:
            raise RegionNotFoundError(region_id)
        del self._regions[region_id]
        self.save_to_disk()

    # ─── Event Detection & Generation ──────────────────────────────────

    def generate_events_for_run(
        self, run_id: str, config: EventRuleConfig | None = None
    ) -> list[TemporalEvent]:
        """Detect and store all temporal events for a VideoInferenceRun."""
        video_svc = get_video_intelligence_service()
        run = video_svc.get_run(run_id)
        regions = self.list_regions(run.video_id)

        detector = TemporalEventDetector(config=config)
        detected = detector.detect_events(run, regions)

        # Clear prior events for this run to avoid stale duplicate events
        self._events = {k: v for k, v in self._events.items() if v.run_id != run_id}

        # Store generated events
        for evt in detected:
            self._events[evt.event_id] = evt

        self.save_to_disk()
        logger.info("Generated %d temporal events for run '%s'", len(detected), run_id)
        return detected

    def get_events_for_run(
        self,
        run_id: str,
        event_type: EventType | None = None,
        track_id: int | None = None,
        region_id: str | None = None,
    ) -> list[TemporalEvent]:
        """Retrieve stored events for a run with optional filtering."""
        evts = [e for e in self._events.values() if e.run_id == run_id]

        if not evts:
            # Auto-generate events if none stored yet
            evts = self.generate_events_for_run(run_id)

        if event_type:
            evts = [e for e in evts if e.event_type == event_type]
        if track_id is not None:
            evts = [e for e in evts if track_id in e.source_track_ids]
        if region_id:
            evts = [e for e in evts if e.event_params.get("region_id") == region_id]

        return sorted(evts, key=lambda e: e.start_timestamp_sec)

    def get_event_detail(self, event_id: str) -> TemporalEvent:
        if event_id not in self._events:
            raise EventNotFoundError(event_id)
        return self._events[event_id]

    # ─── Evidence & Analytics Extraction ─────────────────────────────

    def get_event_evidence(self, event_id: str) -> EventEvidence:
        """Extract 3-frame visual verification evidence indices for an event."""
        evt = self.get_event_detail(event_id)
        start_f = evt.source_frame_range[0] if evt.source_frame_range else 0
        end_f = evt.source_frame_range[-1] if evt.source_frame_range else start_f

        frame_before = max(0, start_f - 2)
        event_frame = start_f
        frame_after = end_f + 2

        notes = f"Evidence verification for {evt.event_type.value}: {evt.description}"

        return EventEvidence(
            event_id=evt.event_id,
            frame_before_idx=frame_before,
            event_frame_idx=event_frame,
            frame_after_idx=frame_after,
            highlight_track_ids=evt.source_track_ids,
            highlight_region_id=evt.event_params.get("region_id"),
            snapshot_notes=notes,
        )

    def get_event_analytics(self, run_id: str) -> EventAnalytics:
        """Compute aggregate event telemetry and timeline for a run."""
        evts = self.get_events_for_run(run_id)

        type_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        region_counts: dict[str, int] = {}
        dwell_durations: list[float] = []
        prox_count = 0

        timeline_entries: list[dict[str, Any]] = []

        for e in evts:
            t_str = e.event_type.value
            type_counts[t_str] = type_counts.get(t_str, 0) + 1

            cls_name = e.event_params.get("class_name")
            if cls_name:
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            reg_name = e.event_params.get("region_name")
            if reg_name:
                region_counts[reg_name] = region_counts.get(reg_name, 0) + 1

            if e.event_type == EventType.OBJECT_DWELLED:
                dwell_durations.append(e.duration_sec)

            if e.event_type in (EventType.OBJECTS_BECAME_CLOSE, EventType.OBJECTS_MOVED_APART):
                prox_count += 1

            timeline_entries.append(
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "timestamp_sec": e.start_timestamp_sec,
                    "duration_sec": e.duration_sec,
                    "description": e.description,
                    "tracks": e.source_track_ids,
                    "region": reg_name,
                }
            )

        avg_dwell = sum(dwell_durations) / max(1, len(dwell_durations)) if dwell_durations else 0.0
        max_dwell = max(dwell_durations) if dwell_durations else 0.0

        return EventAnalytics(
            total_events=len(evts),
            events_by_type=type_counts,
            events_by_class=class_counts,
            events_by_region=region_counts,
            avg_dwell_time_sec=round(avg_dwell, 2),
            longest_dwell_sec=round(max_dwell, 2),
            proximity_events_count=prox_count,
            timeline=timeline_entries,
        )

    def get_scene_summary(self, run_id: str) -> SceneSummary:
        """Construct deterministic scene summary for a video run."""
        video_svc = get_video_intelligence_service()
        run = video_svc.get_run(run_id)
        evts = self.get_events_for_run(run_id)
        regions = self.list_regions(run.video_id)

        analytics = self.get_event_analytics(run_id)

        # Most active region
        most_active = "None"
        if analytics.events_by_region:
            most_active = max(analytics.events_by_region.items(), key=lambda x: x[1])[0]

        # Longest dwell event
        dwell_evts = [e for e in evts if e.event_type == EventType.OBJECT_DWELLED]
        longest_dwell_obj = None
        if dwell_evts:
            longest_e = max(dwell_evts, key=lambda e: e.duration_sec)
            longest_dwell_obj = {
                "event_id": longest_e.event_id,
                "track_id": longest_e.source_track_ids[0] if longest_e.source_track_ids else 0,
                "region_name": longest_e.event_params.get("region_name", "Unknown"),
                "duration_sec": longest_e.duration_sec,
            }

        return SceneSummary(
            video_id=run.video_id,
            run_id=run.run_id,
            duration_sec=run.duration_sec,
            total_tracks=run.total_tracks,
            total_regions=len(regions),
            total_events=len(evts),
            most_active_region=most_active,
            longest_dwell_event=longest_dwell_obj,
        )

    def export_events_csv(self, run_id: str) -> str:
        """Export temporal event stream as CSV string."""
        evts = self.get_events_for_run(run_id)
        lines = [
            "event_id,run_id,video_id,event_type,start_sec,end_sec,duration_sec,tracks,description"
        ]

        for e in evts:
            tr_str = ";".join(str(t) for t in e.source_track_ids)
            desc_clean = e.description.replace(",", ";")
            lines.append(
                f"{e.event_id},{e.run_id},{e.video_id},{e.event_type.value},{e.start_timestamp_sec},{e.end_timestamp_sec},{e.duration_sec},{tr_str},{desc_clean}"
            )

        return "\n".join(lines)

    # ─── Persistence Helpers ──────────────────────────────────────────

    def save_to_disk(self) -> None:
        regs_serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "regions": [r.model_dump() for r in self._regions.values()],
        }
        self._regions_file.write_text(
            json.dumps(regs_serializable, indent=2, default=str), encoding="utf-8"
        )

        evts_serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "events": [e.model_dump() for e in self._events.values()],
        }
        self._events_file.write_text(
            json.dumps(evts_serializable, indent=2, default=str), encoding="utf-8"
        )

    def load_from_disk(self) -> None:
        if self._is_custom_storage:
            valid_video_ids: set[str] = set()
            valid_run_ids: set[str] = set()
        else:
            video_svc = get_video_intelligence_service()
            valid_video_ids = set(video_svc._videos.keys())
            valid_run_ids = set(video_svc._runs.keys())

        if self._regions_file.is_file():
            try:
                raw_regs = json.loads(self._regions_file.read_text(encoding="utf-8"))
                for item in raw_regs.get("regions", []):
                    reg = RegionOfInterest(**item)
                    if not valid_video_ids or reg.video_id in valid_video_ids:
                        self._regions[reg.region_id] = reg
            except Exception as exc:
                logger.warning("Failed to restore regions from disk: %s", str(exc))

        if self._events_file.is_file():
            try:
                raw_evts = json.loads(self._events_file.read_text(encoding="utf-8"))
                for item in raw_evts.get("events", []):
                    e = TemporalEvent(**item)
                    if not valid_run_ids or e.run_id in valid_run_ids:
                        self._events[e.event_id] = e
            except Exception as exc:
                logger.warning("Failed to restore events from disk: %s", str(exc))

        self.save_to_disk()


@lru_cache
def get_temporal_event_service() -> TemporalEventService:
    """Return singleton instance of TemporalEventService."""
    return TemporalEventService()
