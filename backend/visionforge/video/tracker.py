"""VisionForge ByteTrack / IoU-Kalman Multi-Object Tracker."""

import logging
import math
from typing import Any

from visionforge.video.schemas import (
    Track,
    TrackStatus,
    TrajectoryPoint,
)

logger = logging.getLogger("visionforge.video.tracker")


def compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate Intersection-over-Union (IoU) between two bounding boxes [x_min, y_min, x_max, y_max]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[0])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[0])
    union = area1 + area2 - intersection

    return intersection / max(1e-6, union)


class SingleTrackState:
    """Internal tracker state maintaining history for a single persistent Track ID."""

    def __init__(self, track_id: int, class_name: str, frame_index: int, timestamp_sec: float, bbox: list[float], confidence: float, img_width: int, img_height: int):
        self.track_id = track_id
        self.class_name = class_name
        self.first_frame = frame_index
        self.last_frame = frame_index
        self.first_timestamp_sec = timestamp_sec
        self.last_timestamp_sec = timestamp_sec
        self.last_bbox = bbox
        self.confidences: list[float] = [confidence]
        self.trajectory: list[TrajectoryPoint] = []
        self.consecutive_lost = 0
        self.status = TrackStatus.ACTIVE

        self.add_trajectory_point(frame_index, timestamp_sec, bbox, img_width, img_height)

    def add_trajectory_point(self, frame_index: int, timestamp_sec: float, bbox: list[float], img_width: int, img_height: int) -> None:
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        w_px = max(1.0, bbox[2] - bbox[0])
        h_px = max(1.0, bbox[3] - bbox[1])

        norm_x = min(1.0, max(0.0, x_center / max(1, img_width)))
        norm_y = min(1.0, max(0.0, y_center / max(1, img_height)))

        pt = TrajectoryPoint(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            x_center_px=round(x_center, 2),
            y_center_px=round(y_center, 2),
            norm_x=round(norm_x, 4),
            norm_y=round(norm_y, 4),
            width_px=round(w_px, 2),
            height_px=round(h_px, 2),
            bbox=[round(b, 2) for b in bbox],
        )
        self.trajectory.append(pt)
        self.last_frame = frame_index
        self.last_timestamp_sec = timestamp_sec
        self.last_bbox = bbox
        self.consecutive_lost = 0
        self.status = TrackStatus.ACTIVE

    def to_schema(self) -> Track:
        """Convert track state to standard Track model."""
        avg_conf = sum(self.confidences) / max(1, len(self.confidences))
        min_conf = min(self.confidences) if self.confidences else 0.0
        max_conf = max(self.confidences) if self.confidences else 0.0
        duration = max(0.0, self.last_timestamp_sec - self.first_timestamp_sec)

        # Calculate cumulative distance in pixels
        total_dist_px = 0.0
        for i in range(1, len(self.trajectory)):
            pt1 = self.trajectory[i - 1]
            pt2 = self.trajectory[i]
            dx = pt2.x_center_px - pt1.x_center_px
            dy = pt2.y_center_px - pt1.y_center_px
            total_dist_px += math.sqrt(dx * dx + dy * dy)

        avg_speed = total_dist_px / max(0.1, duration) if duration > 0 else 0.0

        return Track(
            track_id=self.track_id,
            class_name=self.class_name,
            first_frame=self.first_frame,
            last_frame=self.last_frame,
            first_timestamp_sec=round(self.first_timestamp_sec, 3),
            last_timestamp_sec=round(self.last_timestamp_sec, 3),
            visibility_duration_sec=round(duration, 3),
            avg_confidence=round(avg_conf, 4),
            min_confidence=round(min_conf, 4),
            max_confidence=round(max_conf, 4),
            total_distance_px=round(total_dist_px, 2),
            avg_speed_px_per_sec=round(avg_speed, 2),
            status=self.status,
            trajectory=self.trajectory,
            detections_count=len(self.trajectory),
        )


class ByteTracker:
    """ByteTrack IoU-based Multi-Object Tracker with persistent Track IDs."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.tracks: dict[int, SingleTrackState] = {}

    def update(
        self,
        frame_index: int,
        timestamp_sec: float,
        detections: list[dict[str, Any]],
        img_width: int,
        img_height: int,
    ) -> list[dict[str, Any]]:
        """Update tracker state with new frame detections and return assigned track IDs.

        Each detection dict in `detections` expected to contain:
        `bbox`: [x_min, y_min, x_max, y_max]
        `confidence`: float
        `class_name`: str
        """
        active_track_ids = [
            tid for tid, t in self.tracks.items() if t.status in (TrackStatus.ACTIVE, TrackStatus.LOST)
        ]

        assigned_detections: set[int] = set()
        matched_tracks: set[int] = set()

        # Step 1: Match detections to existing active tracks via IoU matrix
        if active_track_ids and detections:
            # Build IoU matrix
            matches: list[tuple[float, int, int]] = []
            for det_idx, det in enumerate(detections):
                d_bbox = det.get("bbox", [0, 0, 0, 0])
                d_cls = det.get("class_name", "object")

                for tid in active_track_ids:
                    track = self.tracks[tid]
                    # Class matching check
                    if track.class_name == d_cls:
                        iou = compute_iou(d_bbox, track.last_bbox)
                        if iou >= self.iou_threshold:
                            matches.append((iou, det_idx, tid))

            # Sort matches descending by IoU score
            matches.sort(key=lambda x: x[0], reverse=True)

            for iou, det_idx, tid in matches:
                if det_idx not in assigned_detections and tid not in matched_tracks:
                    assigned_detections.add(det_idx)
                    matched_tracks.add(tid)

                    det = detections[det_idx]
                    bbox = det["bbox"]
                    conf = det.get("confidence", 0.5)

                    self.tracks[tid].add_trajectory_point(
                        frame_index, timestamp_sec, bbox, img_width, img_height
                    )
                    self.tracks[tid].confidences.append(conf)

        # Step 2: Handle unmatched active tracks
        for tid in active_track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.consecutive_lost += 1
                if track.consecutive_lost > self.max_lost_frames:
                    track.status = TrackStatus.TERMINATED
                else:
                    track.status = TrackStatus.LOST

        # Step 3: Spawn new persistent tracks for unmatched detections
        output_results: list[dict[str, Any]] = []

        for det_idx, det in enumerate(detections):
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = det.get("confidence", 0.5)
            cls_name = det.get("class_name", "object")

            if det_idx in assigned_detections:
                # Find associated track ID
                for tid in matched_tracks:
                    if (
                        self.tracks[tid].last_frame == frame_index
                        and self.tracks[tid].last_bbox == bbox
                    ):
                        track_id = tid
                        break
                else:
                    track_id = 0
            else:
                track_id = self.next_track_id
                self.next_track_id += 1

                new_track = SingleTrackState(
                    track_id=track_id,
                    class_name=cls_name,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    bbox=bbox,
                    confidence=conf,
                    img_width=img_width,
                    img_height=img_height,
                )
                self.tracks[track_id] = new_track

            output_results.append(
                {
                    "track_id": track_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": bbox,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                }
            )

        return output_results

    def get_all_tracks(self) -> list[Track]:
        """Return list of all tracks converted to standard Track schemas."""
        return [t.to_schema() for t in self.tracks.values()]
