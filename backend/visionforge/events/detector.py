"""VisionForge Rule-Based Temporal Event Detector."""

import math
import uuid

from visionforge.events.schemas import (
    EventEvidence,
    EventReliability,
    EventRuleConfig,
    EventType,
    RegionOfInterest,
    RegionShape,
    TemporalEvent,
)
from visionforge.video.schemas import RegionVisit, Track, VideoInferenceRun


def is_point_in_rectangle(pt: tuple[float, float], rect: list[list[float]]) -> bool:
    """Check if point (x, y) lies inside rectangle [[x_min, y_min], [x_max, y_max]] or [x_min, y_min, x_max, y_max]."""
    x, y = pt
    if len(rect) == 2 and isinstance(rect[0], list):
        x_min, y_min = rect[0]
        x_max, y_max = rect[1]
    elif len(rect) == 4 and isinstance(rect[0], (int, float)):
        x_min, y_min, x_max, y_max = rect  # type: ignore
    else:
        x_min, y_min, x_max, y_max = 0.0, 0.0, 1920.0, 1080.0

    return x_min <= x <= x_max and y_min <= y <= y_max


def is_point_in_polygon(pt: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray-casting algorithm to test if point (x, y) lies inside polygon vertices."""
    x, y = pt
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def is_point_in_region(pt: tuple[float, float], region: RegionOfInterest) -> bool:
    """Check if point (x, y) is inside Region of Interest."""
    if region.shape_type == RegionShape.RECTANGLE:
        return is_point_in_rectangle(pt, region.coordinates)
    return is_point_in_polygon(pt, region.coordinates)


class TemporalEventDetector:
    """Deterministic, rule-based detector converting object trajectories into explainable temporal events."""

    def __init__(self, config: EventRuleConfig | None = None):
        self.config = config or EventRuleConfig()

    def detect_events(
        self, run: VideoInferenceRun, regions: list[RegionOfInterest] | None = None
    ) -> list[TemporalEvent]:
        """Process video inference run tracks and regions to derive chronological temporal event stream."""
        events: list[TemporalEvent] = []
        active_regions = regions or []

        # 1. Track Lifecycle Events: OBJECT_APPEARED & OBJECT_DISAPPEARED
        events.extend(self._detect_track_lifecycle_events(run))

        # 2. Region-Based Events: ENTER, EXIT, DWELL, & CROSSING
        for region in active_regions:
            events.extend(self._detect_region_events(run, region))

        # 3. Image-Plane Movement State Events: OBJECT_STOPPED & OBJECT_MOVED
        events.extend(self._detect_movement_state_events(run))

        # 4. Proximity & Separation Events: OBJECTS_BECAME_CLOSE & PROLONGED_PROXIMITY
        events.extend(self._detect_proximity_events(run))

        # 5. Active Object Count Change Events: OBJECT_COUNT_CHANGED
        events.extend(self._detect_count_change_events(run))

        # Populate region visits and associated events on tracks
        self._populate_track_event_associations(run.tracks, events)

        # Sort all derived events chronologically by start timestamp
        events.sort(key=lambda e: (e.start_timestamp_sec, e.event_type.value))
        return events

    def _detect_track_lifecycle_events(self, run: VideoInferenceRun) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []

        for track in run.tracks:
            # TRACK_STARTED / OBJECT_APPEARED
            evt_id = f"evt_{uuid.uuid4().hex[:10]}"
            rule_app = f"Track #{track.track_id} ({track.class_name}) initiated observation at t={track.first_timestamp_sec:.1f}s."
            evidence_app = EventEvidence(
                event_id=evt_id,
                frame_before_idx=max(0, track.first_frame - 1),
                event_frame_idx=track.first_frame,
                frame_after_idx=min(run.processed_frames, track.first_frame + 1),
                start_timestamp_sec=track.first_timestamp_sec,
                representative_timestamp_sec=track.first_timestamp_sec,
                end_timestamp_sec=track.first_timestamp_sec,
                highlight_track_ids=[track.track_id],
                trigger_rule=rule_app,
                snapshot_notes=f"Object appeared on video canvas at frame #{track.first_frame}.",
            )

            evt_start = TemporalEvent(
                event_id=evt_id,
                run_id=run.run_id,
                video_id=run.video_id,
                event_type=EventType.TRACK_STARTED,
                start_timestamp_sec=track.first_timestamp_sec,
                end_timestamp_sec=track.first_timestamp_sec,
                duration_sec=0.0,
                source_track_ids=[track.track_id],
                source_frame_range=[track.first_frame, track.first_frame],
                reliability=EventReliability.HIGH,
                event_params={
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "initial_confidence": track.avg_confidence,
                    "initial_position": [
                        track.trajectory[0].x_center_px,
                        track.trajectory[0].y_center_px,
                    ]
                    if track.trajectory
                    else [0, 0],
                },
                description=f"Track #{track.track_id} ({track.class_name}) started at t={track.first_timestamp_sec:.1f}s.",
                trigger_rule=rule_app,
                evidence=evidence_app,
            )
            events.append(evt_start)

            # TRACK_ENDED (if active for multiple frames)
            if track.visibility_duration_sec > 0.0:
                evt_end_id = f"evt_{uuid.uuid4().hex[:10]}"
                rule_dis = f"Track #{track.track_id} ({track.class_name}) observation terminated at t={track.last_timestamp_sec:.1f}s."
                evidence_dis = EventEvidence(
                    event_id=evt_end_id,
                    frame_before_idx=max(0, track.last_frame - 1),
                    event_frame_idx=track.last_frame,
                    frame_after_idx=track.last_frame,
                    start_timestamp_sec=track.last_timestamp_sec,
                    representative_timestamp_sec=track.last_timestamp_sec,
                    end_timestamp_sec=track.last_timestamp_sec,
                    highlight_track_ids=[track.track_id],
                    trigger_rule=rule_dis,
                    snapshot_notes=f"Object disappeared from frame #{track.last_frame}.",
                )

                evt_end = TemporalEvent(
                    event_id=evt_end_id,
                    run_id=run.run_id,
                    video_id=run.video_id,
                    event_type=EventType.TRACK_ENDED,
                    start_timestamp_sec=track.last_timestamp_sec,
                    end_timestamp_sec=track.last_timestamp_sec,
                    duration_sec=0.0,
                    source_track_ids=[track.track_id],
                    source_frame_range=[track.last_frame, track.last_frame],
                    reliability=EventReliability.HIGH,
                    event_params={
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "total_visibility_sec": track.visibility_duration_sec,
                    },
                    description=f"Track #{track.track_id} ({track.class_name}) ended at t={track.last_timestamp_sec:.1f}s.",
                    trigger_rule=rule_dis,
                    evidence=evidence_dis,
                )
                events.append(evt_end)

        return events

    def _detect_region_events(
        self, run: VideoInferenceRun, region: RegionOfInterest
    ) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []

        for track in run.tracks:
            if not track.trajectory:
                continue

            inside_states: list[bool] = [
                is_point_in_region((pt.x_center_px, pt.y_center_px), region) for pt in track.trajectory
            ]

            # Apply debouncing window
            debounced_inside: list[bool] = list(inside_states)
            for i in range(1, len(debounced_inside) - 1):
                if inside_states[i - 1] == inside_states[i + 1] and inside_states[i] != inside_states[i - 1]:
                    debounced_inside[i] = inside_states[i - 1]

            in_region = False
            entry_idx = -1

            for i, is_in in enumerate(debounced_inside):
                pt = track.trajectory[i]

                if is_in and not in_region:
                    # Transition: OBJECT_ENTERED_REGION
                    in_region = True
                    entry_idx = i
                    evt_id = f"evt_{uuid.uuid4().hex[:10]}"
                    rule = f"Track #{track.track_id} ({track.class_name}) centroid crossed inside region '{region.name}' boundary."

                    evidence = EventEvidence(
                        event_id=evt_id,
                        frame_before_idx=max(0, pt.frame_index - 1),
                        event_frame_idx=pt.frame_index,
                        frame_after_idx=min(run.processed_frames, pt.frame_index + 1),
                        start_timestamp_sec=pt.timestamp_sec,
                        representative_timestamp_sec=pt.timestamp_sec,
                        end_timestamp_sec=pt.timestamp_sec,
                        highlight_track_ids=[track.track_id],
                        highlight_region_id=region.region_id,
                        trigger_rule=rule,
                        snapshot_notes=f"Track entered region '{region.name}'.",
                    )

                    events.append(
                        TemporalEvent(
                            event_id=evt_id,
                            run_id=run.run_id,
                            video_id=run.video_id,
                            event_type=EventType.OBJECT_ENTERED_REGION,
                            start_timestamp_sec=pt.timestamp_sec,
                            end_timestamp_sec=pt.timestamp_sec,
                            duration_sec=0.0,
                            source_track_ids=[track.track_id],
                            source_frame_range=[pt.frame_index, pt.frame_index],
                            reliability=EventReliability.HIGH,
                            event_params={
                                "track_id": track.track_id,
                                "class_name": track.class_name,
                                "region_id": region.region_id,
                                "region_name": region.name,
                                "entry_position": [pt.x_center_px, pt.y_center_px],
                            },
                            description=f"Track #{track.track_id} ({track.class_name}) entered region '{region.name}' at t={pt.timestamp_sec:.1f}s.",
                            trigger_rule=rule,
                            evidence=evidence,
                        )
                    )

                elif not is_in and in_region:
                    # Transition: OBJECT_LEFT_REGION
                    in_region = False
                    entry_pt = track.trajectory[entry_idx]
                    dwell_sec = pt.timestamp_sec - entry_pt.timestamp_sec
                    evt_id = f"evt_{uuid.uuid4().hex[:10]}"
                    rule = f"Track #{track.track_id} ({track.class_name}) crossed outside region '{region.name}' boundary."

                    evidence = EventEvidence(
                        event_id=evt_id,
                        frame_before_idx=max(0, pt.frame_index - 1),
                        event_frame_idx=pt.frame_index,
                        frame_after_idx=min(run.processed_frames, pt.frame_index + 1),
                        start_timestamp_sec=pt.timestamp_sec,
                        representative_timestamp_sec=pt.timestamp_sec,
                        end_timestamp_sec=pt.timestamp_sec,
                        highlight_track_ids=[track.track_id],
                        highlight_region_id=region.region_id,
                        trigger_rule=rule,
                        snapshot_notes=f"Track left region '{region.name}' after {dwell_sec:.1f}s.",
                    )

                    events.append(
                        TemporalEvent(
                            event_id=evt_id,
                            run_id=run.run_id,
                            video_id=run.video_id,
                            event_type=EventType.OBJECT_LEFT_REGION,
                            start_timestamp_sec=pt.timestamp_sec,
                            end_timestamp_sec=pt.timestamp_sec,
                            duration_sec=0.0,
                            source_track_ids=[track.track_id],
                            source_frame_range=[pt.frame_index, pt.frame_index],
                            reliability=EventReliability.HIGH,
                            event_params={
                                "track_id": track.track_id,
                                "class_name": track.class_name,
                                "region_id": region.region_id,
                                "region_name": region.name,
                                "exit_position": [pt.x_center_px, pt.y_center_px],
                                "total_dwell_sec": round(dwell_sec, 2),
                            },
                            description=f"Track #{track.track_id} ({track.class_name}) left region '{region.name}' at t={pt.timestamp_sec:.1f}s.",
                            trigger_rule=rule,
                            evidence=evidence,
                        )
                    )

                    # OBJECT_DWELLED / OBJECT_STAYED_IN_REGION
                    if dwell_sec >= self.config.dwell_threshold_sec:
                        dwell_evt_id = f"evt_{uuid.uuid4().hex[:10]}"
                        dwell_rule = f"Track #{track.track_id} dwelled in '{region.name}' for {dwell_sec:.1f}s (threshold: {self.config.dwell_threshold_sec:.1f}s)."
                        rep_idx = (entry_pt.frame_index + pt.frame_index) // 2
                        rep_time = (entry_pt.timestamp_sec + pt.timestamp_sec) / 2.0

                        dwell_ev = EventEvidence(
                            event_id=dwell_evt_id,
                            frame_before_idx=entry_pt.frame_index,
                            event_frame_idx=rep_idx,
                            frame_after_idx=pt.frame_index,
                            start_timestamp_sec=entry_pt.timestamp_sec,
                            representative_timestamp_sec=rep_time,
                            end_timestamp_sec=pt.timestamp_sec,
                            highlight_track_ids=[track.track_id],
                            highlight_region_id=region.region_id,
                            trigger_rule=dwell_rule,
                            snapshot_notes=f"Track dwelled in region for {dwell_sec:.1f}s.",
                        )

                        events.append(
                            TemporalEvent(
                                event_id=dwell_evt_id,
                                run_id=run.run_id,
                                video_id=run.video_id,
                                event_type=EventType.OBJECT_DWELLED,
                                start_timestamp_sec=entry_pt.timestamp_sec,
                                end_timestamp_sec=pt.timestamp_sec,
                                duration_sec=round(dwell_sec, 2),
                                source_track_ids=[track.track_id],
                                source_frame_range=[entry_pt.frame_index, pt.frame_index],
                                reliability=EventReliability.HIGH,
                                event_params={
                                    "track_id": track.track_id,
                                    "class_name": track.class_name,
                                    "region_id": region.region_id,
                                    "region_name": region.name,
                                    "dwell_duration_sec": round(dwell_sec, 2),
                                    "entry_position": [entry_pt.x_center_px, entry_pt.y_center_px],
                                    "exit_position": [pt.x_center_px, pt.y_center_px],
                                },
                                description=f"Track #{track.track_id} ({track.class_name}) stayed in region '{region.name}' for {dwell_sec:.1f}s.",
                                trigger_rule=dwell_rule,
                                evidence=dwell_ev,
                            )
                        )

            # Handle object still inside region at track end
            if in_region and entry_idx >= 0:
                entry_pt = track.trajectory[entry_idx]
                end_pt = track.trajectory[-1]
                dwell_sec = end_pt.timestamp_sec - entry_pt.timestamp_sec

                if dwell_sec >= self.config.dwell_threshold_sec:
                    dwell_evt_id = f"evt_{uuid.uuid4().hex[:10]}"
                    dwell_rule = f"Track #{track.track_id} remained in '{region.name}' for {dwell_sec:.1f}s through track end."
                    dwell_ev = EventEvidence(
                        event_id=dwell_evt_id,
                        frame_before_idx=entry_pt.frame_index,
                        event_frame_idx=(entry_pt.frame_index + end_pt.frame_index) // 2,
                        frame_after_idx=end_pt.frame_index,
                        start_timestamp_sec=entry_pt.timestamp_sec,
                        representative_timestamp_sec=(entry_pt.timestamp_sec + end_pt.timestamp_sec) / 2.0,
                        end_timestamp_sec=end_pt.timestamp_sec,
                        highlight_track_ids=[track.track_id],
                        highlight_region_id=region.region_id,
                        trigger_rule=dwell_rule,
                        snapshot_notes=f"Track dwelled in region for {dwell_sec:.1f}s.",
                    )

                    events.append(
                        TemporalEvent(
                            event_id=dwell_evt_id,
                            run_id=run.run_id,
                            video_id=run.video_id,
                            event_type=EventType.OBJECT_DWELLED,
                            start_timestamp_sec=entry_pt.timestamp_sec,
                            end_timestamp_sec=end_pt.timestamp_sec,
                            duration_sec=round(dwell_sec, 2),
                            source_track_ids=[track.track_id],
                            source_frame_range=[entry_pt.frame_index, end_pt.frame_index],
                            reliability=EventReliability.HIGH,
                            event_params={
                                "track_id": track.track_id,
                                "class_name": track.class_name,
                                "region_id": region.region_id,
                                "region_name": region.name,
                                "dwell_duration_sec": round(dwell_sec, 2),
                            },
                            description=f"Track #{track.track_id} ({track.class_name}) stayed in region '{region.name}' for {dwell_sec:.1f}s.",
                            trigger_rule=dwell_rule,
                            evidence=dwell_ev,
                        )
                    )

        return events

    def _detect_movement_state_events(self, run: VideoInferenceRun) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []
        stopped_thresh = self.config.stopped_speed_threshold_px_s

        for track in run.tracks:
            if len(track.trajectory) < 4:
                continue

            speeds: list[float] = []
            for i in range(1, len(track.trajectory)):
                p1 = track.trajectory[i - 1]
                p2 = track.trajectory[i]
                dt = p2.timestamp_sec - p1.timestamp_sec
                if dt > 0:
                    dist = math.hypot(p2.x_center_px - p1.x_center_px, p2.y_center_px - p1.y_center_px)
                    spd = dist / dt
                    speeds.append(spd)
                    p2.instantaneous_speed_px_s = round(spd, 2)
                else:
                    speeds.append(0.0)

            is_stopped = False
            stop_start_idx = -1

            for i, spd in enumerate(speeds):
                pt = track.trajectory[i + 1]

                if spd < stopped_thresh and not is_stopped:
                    is_stopped = True
                    stop_start_idx = i
                elif spd >= stopped_thresh and is_stopped:
                    is_stopped = False
                    start_pt = track.trajectory[stop_start_idx]
                    dur = pt.timestamp_sec - start_pt.timestamp_sec

                    if dur >= 2.0:
                        evt_id = f"evt_{uuid.uuid4().hex[:10]}"
                        rule = f"Track #{track.track_id} speed dropped below {stopped_thresh} px/s for {dur:.1f}s."
                        ev = EventEvidence(
                            event_id=evt_id,
                            frame_before_idx=start_pt.frame_index,
                            event_frame_idx=(start_pt.frame_index + pt.frame_index) // 2,
                            frame_after_idx=pt.frame_index,
                            start_timestamp_sec=start_pt.timestamp_sec,
                            representative_timestamp_sec=(start_pt.timestamp_sec + pt.timestamp_sec) / 2.0,
                            end_timestamp_sec=pt.timestamp_sec,
                            highlight_track_ids=[track.track_id],
                            trigger_rule=rule,
                            snapshot_notes=f"Track stopped for {dur:.1f}s.",
                        )

                        events.append(
                            TemporalEvent(
                                event_id=evt_id,
                                run_id=run.run_id,
                                video_id=run.video_id,
                                event_type=EventType.OBJECT_STOPPED,
                                start_timestamp_sec=start_pt.timestamp_sec,
                                end_timestamp_sec=pt.timestamp_sec,
                                duration_sec=round(dur, 2),
                                source_track_ids=[track.track_id],
                                source_frame_range=[start_pt.frame_index, pt.frame_index],
                                reliability=EventReliability.MEDIUM,
                                event_params={
                                    "track_id": track.track_id,
                                    "class_name": track.class_name,
                                    "stopped_duration_sec": round(dur, 2),
                                    "average_speed_px_s": round(sum(speeds[stop_start_idx:i]) / max(1, i - stop_start_idx), 2),
                                },
                                description=f"Track #{track.track_id} ({track.class_name}) stopped in place for {dur:.1f}s.",
                                trigger_rule=rule,
                                evidence=ev,
                            )
                        )

        return events

    def _detect_proximity_events(self, run: VideoInferenceRun) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []
        prox_thresh = self.config.proximity_threshold_px
        tracks = run.tracks
        n = len(tracks)

        for i in range(n):
            for j in range(i + 1, n):
                t1 = tracks[i]
                t2 = tracks[j]

                # Map trajectory points by frame index
                t1_frames = {pt.frame_index: pt for pt in t1.trajectory}
                t2_frames = {pt.frame_index: pt for pt in t2.trajectory}
                common_frames = sorted(set(t1_frames.keys()).intersection(set(t2_frames.keys())))

                if len(common_frames) < 3:
                    continue

                in_proximity = False
                prox_start_frame = -1
                prox_start_time = 0.0

                for f_idx in common_frames:
                    p1 = t1_frames[f_idx]
                    p2 = t2_frames[f_idx]
                    dist = math.hypot(p1.x_center_px - p2.x_center_px, p1.y_center_px - p2.y_center_px)

                    if dist <= prox_thresh and not in_proximity:
                        in_proximity = True
                        prox_start_frame = f_idx
                        prox_start_time = p1.timestamp_sec
                    elif dist > prox_thresh and in_proximity:
                        in_proximity = False
                        dur = p1.timestamp_sec - prox_start_time

                        if dur >= 1.5:
                            evt_id = f"evt_{uuid.uuid4().hex[:10]}"
                            rule = f"Distance between Track #{t1.track_id} ({t1.class_name}) and Track #{t2.track_id} ({t2.class_name}) remained <= {prox_thresh}px for {dur:.1f}s."
                            ev = EventEvidence(
                                event_id=evt_id,
                                frame_before_idx=prox_start_frame,
                                event_frame_idx=(prox_start_frame + f_idx) // 2,
                                frame_after_idx=f_idx,
                                start_timestamp_sec=prox_start_time,
                                representative_timestamp_sec=(prox_start_time + p1.timestamp_sec) / 2.0,
                                end_timestamp_sec=p1.timestamp_sec,
                                highlight_track_ids=[t1.track_id, t2.track_id],
                                trigger_rule=rule,
                                snapshot_notes=f"Prolonged proximity ({dur:.1f}s) between Track #{t1.track_id} and Track #{t2.track_id}.",
                            )

                            events.append(
                                TemporalEvent(
                                    event_id=evt_id,
                                    run_id=run.run_id,
                                    video_id=run.video_id,
                                    event_type=EventType.OBJECTS_BECAME_CLOSE,
                                    start_timestamp_sec=prox_start_time,
                                    end_timestamp_sec=p1.timestamp_sec,
                                    duration_sec=round(dur, 2),
                                    source_track_ids=[t1.track_id, t2.track_id],
                                    source_frame_range=[prox_start_frame, f_idx],
                                    reliability=EventReliability.MEDIUM,
                                    event_params={
                                        "track_a_id": t1.track_id,
                                        "track_b_id": t2.track_id,
                                        "class_a": t1.class_name,
                                        "class_b": t2.class_name,
                                        "proximity_distance_px": round(dist, 1),
                                        "duration_sec": round(dur, 2),
                                    },
                                    description=f"Track #{t1.track_id} ({t1.class_name}) in prolonged proximity with Track #{t2.track_id} ({t2.class_name}) for {dur:.1f}s.",
                                    trigger_rule=rule,
                                    evidence=ev,
                                )
                            )

        return events

    def _detect_count_change_events(self, run: VideoInferenceRun) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []
        if not run.analytics.active_objects_over_time:
            return events

        prev_count = None
        for entry in run.analytics.active_objects_over_time:
            t = entry.get("second", 0.0)
            cnt = entry.get("count", 0)
            if prev_count is not None and cnt != prev_count:
                delta = cnt - prev_count
                evt_id = f"evt_{uuid.uuid4().hex[:10]}"
                rule = f"Active object count changed from {prev_count} to {cnt} (Δ {delta:+d}) at t={t:.1f}s."
                events.append(
                    TemporalEvent(
                        event_id=evt_id,
                        run_id=run.run_id,
                        video_id=run.video_id,
                        event_type=EventType.OBJECT_COUNT_CHANGED,
                        start_timestamp_sec=float(t),
                        end_timestamp_sec=float(t),
                        duration_sec=0.0,
                        source_track_ids=[],
                        source_frame_range=[int(t * 30), int(t * 30)],
                        reliability=EventReliability.HIGH,
                        event_params={"previous_count": prev_count, "new_count": cnt, "delta": delta},
                        description=f"Active object count changed from {prev_count} to {cnt} at t={t:.1f}s.",
                        trigger_rule=rule,
                        evidence=None,
                    )
                )
            prev_count = cnt

        return events

    def _populate_track_event_associations(
        self, tracks: list[Track], events: list[TemporalEvent]
    ) -> None:
        track_map = {t.track_id: t for t in tracks}
        for evt in events:
            for tid in evt.source_track_ids:
                if tid in track_map:
                    t = track_map[tid]
                    if evt.event_id not in t.associated_events:
                        t.associated_events.append(evt.event_id)

                    # Extract region visit
                    if (
                        evt.event_type in (EventType.OBJECT_ENTERED_REGION, EventType.OBJECT_STAYED_IN_REGION)
                        and "region_id" in evt.event_params
                    ):
                        rid = evt.event_params["region_id"]
                        rname = evt.event_params.get("region_name", "Zone")
                        existing = next((rv for rv in t.regions_visited if rv.region_id == rid), None)
                        if not existing:
                            t.regions_visited.append(
                                RegionVisit(
                                    region_id=rid,
                                    region_name=rname,
                                    entered_sec=evt.start_timestamp_sec,
                                    exited_sec=evt.end_timestamp_sec if evt.duration_sec > 0 else None,
                                    dwell_duration_sec=evt.duration_sec,
                                    visit_count=1,
                                )
                            )
                        else:
                            existing.visit_count += 1
                            existing.dwell_duration_sec += evt.duration_sec
