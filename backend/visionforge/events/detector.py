"""VisionForge Rule-Based Temporal Event Detector."""

import math
import uuid

from visionforge.events.schemas import (
    EventReliability,
    EventRuleConfig,
    EventType,
    RegionOfInterest,
    RegionShape,
    TemporalEvent,
)
from visionforge.video.schemas import Track, VideoInferenceRun


def is_point_in_rectangle(pt: tuple[float, float], rect: list[list[float]]) -> bool:
    """Check if point (x, y) lies inside rectangle [[x_min, y_min], [x_max, y_max]] or [x_min, y_min, x_max, y_max]."""
    x, y = pt
    if len(rect) == 2 and isinstance(rect[0], list):
        x_min, y_min = rect[0]
        x_max, y_max = rect[1]
    elif len(rect) == 4 and isinstance(rect[0], (int, float)):
        x_min, y_min, x_max, y_max = rect  # type: ignore
    else:
        # Fallback default box
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

        # 1. Track Lifecycle Events: TRACK_STARTED & TRACK_ENDED
        events.extend(self._detect_track_lifecycle_events(run))

        # 2. Region-Based Events: ENTER, EXIT, & DWELL
        for region in active_regions:
            events.extend(self._detect_region_events(run, region))

        # 3. Image-Plane Movement State Events: OBJECT_STOPPED & OBJECT_MOVED
        events.extend(self._detect_movement_state_events(run))

        # 4. Proximity & Separation Events: OBJECTS_BECAME_CLOSE & OBJECTS_MOVED_APART
        events.extend(self._detect_proximity_events(run))

        # 5. Active Object Count Change Events: OBJECT_COUNT_CHANGED
        events.extend(self._detect_count_change_events(run))

        # Sort all derived events chronologically by start timestamp
        events.sort(key=lambda e: (e.start_timestamp_sec, e.event_type))
        return events

    def _detect_track_lifecycle_events(self, run: VideoInferenceRun) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []

        for track in run.tracks:
            # TRACK_STARTED
            evt_start = TemporalEvent(
                event_id=f"evt_{uuid.uuid4().hex[:10]}",
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
                description=f"Track #{track.track_id} ({track.class_name}) started at t={track.first_timestamp_sec:.1f}s (Frame #{track.first_frame}).",
            )
            events.append(evt_start)

            # TRACK_ENDED (only if track was active for multiple frames)
            if track.visibility_duration_sec > 0.0:
                evt_end = TemporalEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:10]}",
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
                        "final_position": [
                            track.trajectory[-1].x_center_px,
                            track.trajectory[-1].y_center_px,
                        ]
                        if track.trajectory
                        else [0, 0],
                    },
                    description=f"Track #{track.track_id} ({track.class_name}) ended at t={track.last_timestamp_sec:.1f}s (Frame #{track.last_frame}).",
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

            # Track inside/outside status sequence
            inside_states: list[bool] = [
                is_point_in_region((pt.x_center_px, pt.y_center_px), region) for pt in track.trajectory
            ]

            # Apply debouncing window to eliminate single-frame boundary noise
            debounced_inside: list[bool] = list(inside_states)
            for i in range(1, len(debounced_inside) - 1):
                if inside_states[i - 1] == inside_states[i + 1] and inside_states[i] != inside_states[i - 1]:
                    debounced_inside[i] = inside_states[i - 1]

            # Detect enter/exit transitions and continuous dwell intervals
            in_region = False
            entry_idx = -1

            for i, is_in in enumerate(debounced_inside):
                pt = track.trajectory[i]

                if is_in and not in_region:
                    # Transition: ENTER REGION
                    in_region = True
                    entry_idx = i

                    events.append(
                        TemporalEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:10]}",
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
                        )
                    )

                elif not is_in and in_region:
                    # Transition: EXIT REGION
                    in_region = False
                    entry_pt = track.trajectory[entry_idx]
                    dwell_sec = pt.timestamp_sec - entry_pt.timestamp_sec

                    events.append(
                        TemporalEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:10]}",
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
                        )
                    )

                    # OBJECT_DWELLED if dwell time exceeds threshold
                    if dwell_sec >= self.config.dwell_threshold_sec:
                        events.append(
                            TemporalEvent(
                                event_id=f"evt_{uuid.uuid4().hex[:10]}",
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
                                description=f"Track #{track.track_id} ({track.class_name}) dwelled in region '{region.name}' for {dwell_sec:.1f}s.",
                            )
                        )

            # Handle object still inside region at track end
            if in_region and entry_idx >= 0:
                entry_pt = track.trajectory[entry_idx]
                end_pt = track.trajectory[-1]
                dwell_sec = end_pt.timestamp_sec - entry_pt.timestamp_sec

                if dwell_sec >= self.config.dwell_threshold_sec:
                    events.append(
                        TemporalEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:10]}",
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
                            description=f"Track #{track.track_id} ({track.class_name}) dwelled in region '{region.name}' for {dwell_sec:.1f}s.",
                        )
                    )

        return events

    def _detect_movement_state_events(self, run: VideoInferenceRun) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []

        for track in run.tracks:
            if len(track.trajectory) < 2:
                continue

            # Compute per-point speed
            speeds: list[float] = []
            for i in range(1, len(track.trajectory)):
                pt1 = track.trajectory[i - 1]
                pt2 = track.trajectory[i]
                dt = max(0.01, pt2.timestamp_sec - pt1.timestamp_sec)
                dist = math.sqrt(
                    (pt2.x_center_px - pt1.x_center_px) ** 2
                    + (pt2.y_center_px - pt1.y_center_px) ** 2
                )
                speeds.append(dist / dt)

            # Classify into STOPPED or MOVED intervals
            is_stopped: list[bool] = [s < self.config.stopped_speed_threshold_px_s for s in speeds]

            # Merge continuous intervals
            start_i = 0
            curr_state = is_stopped[0]

            for i in range(1, len(is_stopped)):
                if is_stopped[i] != curr_state:
                    self._create_movement_event(
                        events, run, track, start_i, i, curr_state, speeds[start_i:i]
                    )
                    start_i = i
                    curr_state = is_stopped[i]

            self._create_movement_event(
                events, run, track, start_i, len(is_stopped), curr_state, speeds[start_i:]
            )

        return events

    def _create_movement_event(
        self,
        events: list[TemporalEvent],
        run: VideoInferenceRun,
        track: Track,
        start_idx: int,
        end_idx: int,
        is_stopped: bool,
        sub_speeds: list[float],
    ) -> None:
        start_pt = track.trajectory[start_idx]
        end_pt = track.trajectory[min(end_idx, len(track.trajectory) - 1)]
        duration = max(0.0, end_pt.timestamp_sec - start_pt.timestamp_sec)

        if duration < 0.5:
            return  # Ignore trivial sub-second state flickers

        evt_type = EventType.OBJECT_STOPPED if is_stopped else EventType.OBJECT_MOVED
        avg_speed = sum(sub_speeds) / max(1, len(sub_speeds))

        events.append(
            TemporalEvent(
                event_id=f"evt_{uuid.uuid4().hex[:10]}",
                run_id=run.run_id,
                video_id=run.video_id,
                event_type=evt_type,
                start_timestamp_sec=start_pt.timestamp_sec,
                end_timestamp_sec=end_pt.timestamp_sec,
                duration_sec=round(duration, 2),
                source_track_ids=[track.track_id],
                source_frame_range=[start_pt.frame_index, end_pt.frame_index],
                reliability=EventReliability.HIGH,
                event_params={
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "avg_speed_px_s": round(avg_speed, 2),
                    "movement_state": "STOPPED" if is_stopped else "MOVING",
                },
                description=f"Track #{track.track_id} ({track.class_name}) {'stopped' if is_stopped else 'moved'} for {duration:.1f}s (avg speed: {avg_speed:.1f} px/s).",
            )
        )

    def _detect_proximity_events(self, run: VideoInferenceRun) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []
        tracks = run.tracks
        n = len(tracks)

        for i in range(n):
            for j in range(i + 1, n):
                t1 = tracks[i]
                t2 = tracks[j]

                # Find overlapping timestamps between t1 and t2
                t1_pts = {round(pt.timestamp_sec, 2): pt for pt in t1.trajectory}
                t2_pts = {round(pt.timestamp_sec, 2): pt for pt in t2.trajectory}
                common_ts = sorted(set(t1_pts.keys()).intersection(t2_pts.keys()))

                if not common_ts:
                    continue

                is_close = False
                close_start_ts = 0.0
                close_start_frame = 0

                for ts in common_ts:
                    pt1 = t1_pts[ts]
                    pt2 = t2_pts[ts]
                    dist = math.sqrt(
                        (pt1.x_center_px - pt2.x_center_px) ** 2
                        + (pt1.y_center_px - pt2.y_center_px) ** 2
                    )

                    if dist <= self.config.proximity_threshold_px and not is_close:
                        is_close = True
                        close_start_ts = ts
                        close_start_frame = pt1.frame_index

                        events.append(
                            TemporalEvent(
                                event_id=f"evt_{uuid.uuid4().hex[:10]}",
                                run_id=run.run_id,
                                video_id=run.video_id,
                                event_type=EventType.OBJECTS_BECAME_CLOSE,
                                start_timestamp_sec=ts,
                                end_timestamp_sec=ts,
                                duration_sec=0.0,
                                source_track_ids=[t1.track_id, t2.track_id],
                                source_frame_range=[pt1.frame_index, pt1.frame_index],
                                reliability=EventReliability.HIGH,
                                event_params={
                                    "track_a": t1.track_id,
                                    "class_a": t1.class_name,
                                    "track_b": t2.track_id,
                                    "class_b": t2.class_name,
                                    "distance_px": round(dist, 1),
                                    "threshold_px": self.config.proximity_threshold_px,
                                },
                                description=f"Track #{t1.track_id} ({t1.class_name}) and Track #{t2.track_id} ({t2.class_name}) became close ({dist:.1f} px apart) at t={ts:.1f}s.",
                            )
                        )

                    elif dist >= self.config.separation_threshold_px and is_close:
                        is_close = False
                        events.append(
                            TemporalEvent(
                                event_id=f"evt_{uuid.uuid4().hex[:10]}",
                                run_id=run.run_id,
                                video_id=run.video_id,
                                event_type=EventType.OBJECTS_MOVED_APART,
                                start_timestamp_sec=ts,
                                end_timestamp_sec=ts,
                                duration_sec=round(ts - close_start_ts, 2),
                                source_track_ids=[t1.track_id, t2.track_id],
                                source_frame_range=[close_start_frame, pt1.frame_index],
                                reliability=EventReliability.HIGH,
                                event_params={
                                    "track_a": t1.track_id,
                                    "class_a": t1.class_name,
                                    "track_b": t2.track_id,
                                    "class_b": t2.class_name,
                                    "distance_px": round(dist, 1),
                                    "threshold_px": self.config.separation_threshold_px,
                                },
                                description=f"Track #{t1.track_id} ({t1.class_name}) and Track #{t2.track_id} ({t2.class_name}) moved apart ({dist:.1f} px apart) at t={ts:.1f}s.",
                            )
                        )

        return events

    def _detect_count_change_events(self, run: VideoInferenceRun) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []
        active_ts = run.analytics.active_objects_over_time

        if len(active_ts) < 2:
            return events

        prev_count = active_ts[0]["active_count"]
        for entry in active_ts[1:]:
            sec = float(entry["second"])
            curr_count = entry["active_count"]

            if curr_count != prev_count:
                delta = curr_count - prev_count
                events.append(
                    TemporalEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:10]}",
                        run_id=run.run_id,
                        video_id=run.video_id,
                        event_type=EventType.OBJECT_COUNT_CHANGED,
                        start_timestamp_sec=sec,
                        end_timestamp_sec=sec,
                        duration_sec=0.0,
                        source_track_ids=[],
                        source_frame_range=[int(sec * 30), int(sec * 30)],
                        reliability=EventReliability.HIGH,
                        event_params={
                            "previous_count": prev_count,
                            "new_count": curr_count,
                            "change_delta": delta,
                            "direction": "INCREASED" if delta > 0 else "DECREASED",
                        },
                        description=f"Active object count {'increased' if delta > 0 else 'decreased'} from {prev_count} to {curr_count} at t={sec:.1f}s.",
                    )
                )
                prev_count = curr_count

        return events
