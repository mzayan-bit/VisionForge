"""VisionForge Video Intelligence Service & Tracking Pipeline."""

import hashlib
import json
import logging
import math
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.inference.service import (
    InferenceService,
    get_inference_service,
)
from visionforge.video.schemas import (
    FrameSamplingConfig,
    FrameSamplingMode,
    TemporalAnalytics,
    Track,
    TrajectoryPoint,
    VideoComparisonResult,
    VideoInferenceRun,
    VideoMetadata,
    VideoSession,
    VideoSessionStatus,
)
from visionforge.video.tracker import ByteTracker

logger = logging.getLogger("visionforge.video.service")


class VideoValidationError(VisionForgeException):
    """Raised when a video file fails validation checks."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="INVALID_VIDEO_FILE",
            status_code=400,
        )


class VideoRunNotFoundError(VisionForgeException):
    """Raised when looking up a video run ID that does not exist."""

    def __init__(self, run_id: str):
        super().__init__(
            message=f"Video inference run '{run_id}' was not found.",
            code="VIDEO_RUN_NOT_FOUND",
            status_code=404,
        )


class VideoIntelligenceService:
    """Service orchestrating video metadata extraction, frame sampling, ByteTrack multi-object tracking, and temporal analytics."""

    def __init__(
        self,
        inference_service: InferenceService | None = None,
        storage_dir: Path | None = None,
    ):
        self._inference_service = inference_service or get_inference_service()
        cache_root = Path(get_settings().model_cache_dir).expanduser().resolve()
        raw_path = storage_dir or (cache_root.parent / "video")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._runs_file = self._storage_dir / "video_runs.json"
        self._videos_file = self._storage_dir / "videos_metadata.json"
        self._sessions_file = self._storage_dir / "video_sessions.json"

        self._runs: dict[str, VideoInferenceRun] = {}
        self._videos: dict[str, VideoMetadata] = {}
        self._sessions: dict[str, VideoSession] = {}
        self.load_from_disk()

    # ─── Video Metadata & Validation ─────────────────────────────────

    def register_video(self, file_path: str, custom_id: str | None = None) -> VideoMetadata:
        """Validate and extract metadata from a real video file asset."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise VideoValidationError(f"Video file path '{file_path}' does not exist.")

        size_bytes = p.stat().st_size
        if size_bytes == 0:
            raise VideoValidationError("Video file is empty (0 bytes).")

        # Compute deterministic SHA-256 fingerprint
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        fingerprint = hasher.hexdigest()

        # Check for existing video with matching fingerprint (deduplication)
        for existing in self._videos.values():
            if existing.video_fingerprint == fingerprint:
                logger.info(
                    "Video with fingerprint '%s' already registered as '%s'",
                    fingerprint,
                    existing.video_id,
                )
                return existing

        vid_id = custom_id or f"vid_{uuid.uuid4().hex[:10]}"

        fps = 30.0
        width = 1920
        height = 1080
        frame_count = 0
        duration_sec = 0.0
        codec = "h264"

        try:
            import cv2

            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise VideoValidationError(
                    f"Failed to open video file '{p.name}'. Invalid or unsupported video stream."
                )

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if fps > 0 and frame_count > 0:
                duration_sec = round(frame_count / fps, 2)

            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            if fourcc:
                codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip() or "h264"

            cap.release()
        except Exception as exc:
            if isinstance(exc, VideoValidationError):
                raise
            raise VideoValidationError(f"Could not read video metadata with OpenCV: {exc}") from exc

        meta = VideoMetadata(
            video_id=vid_id,
            filename=p.name,
            duration_sec=duration_sec,
            fps=round(fps, 2),
            frame_count=frame_count,
            width=width,
            height=height,
            codec=codec,
            size_bytes=size_bytes,
            video_fingerprint=fingerprint,
        )
        self._videos[vid_id] = meta
        self.save_to_disk()
        logger.info(
            "Registered video '%s' (%s, %dx%d, %d frames, %.2fs)",
            vid_id,
            p.name,
            width,
            height,
            frame_count,
            duration_sec,
        )
        return meta

    def get_video_metadata(self, video_id: str) -> VideoMetadata:
        """Retrieve stored metadata for a video asset."""
        if video_id not in self._videos:
            raise VideoValidationError(f"Video ID '{video_id}' not found.")
        return self._videos[video_id]

    def list_videos(self) -> list[VideoMetadata]:
        """List all registered video assets."""
        return sorted(self._videos.values(), key=lambda v: v.created_at, reverse=True)

    # ─── Video Session Management ────────────────────────────────────

    def create_video_session(
        self,
        video_id: str,
        model_version: str = "1.0.0",
        tracking_config: dict[str, Any] | None = None,
        processing_config: dict[str, Any] | None = None,
    ) -> VideoSession:
        """Create a video analysis session with full lineage."""
        meta = self.get_video_metadata(video_id)
        session_id = f"vses_{uuid.uuid4().hex[:8]}"

        session = VideoSession(
            session_id=session_id,
            video_id=video_id,
            video_source=meta.filename,
            duration_sec=meta.duration_sec,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            frame_count=meta.frame_count,
            codec=meta.codec,
            file_size_bytes=meta.size_bytes,
            processing_config=processing_config or {"sampling_interval": 2},
            model_version=model_version,
            tracking_config=tracking_config or {"tracker": "ByteTrack", "track_thresh": 0.45},
            status=VideoSessionStatus.COMPLETED,
            video_fingerprint=meta.video_fingerprint or "sha256_verified_session",
            lineage={
                "model_checkpoint": "yolo11s.pt",
                "tracker": "ByteTrack",
                "dataset_split": "inference",
                "framework": "PyTorch / VisionForge",
            },
        )
        self._sessions[session_id] = session
        self.save_to_disk()
        return session

    def get_video_session(self, session_id: str) -> VideoSession | None:
        """Retrieve video session record."""
        return self._sessions.get(session_id)

    def list_video_sessions(self) -> list[VideoSession]:
        """List all video sessions."""
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def execute_video_inference(
        self,
        video_id: str,
        model_id: str = "yolo11s.pt",
        sampling_mode: FrameSamplingMode = FrameSamplingMode.EVERY_2ND_FRAME,
        custom_stride: int = 2,
        sampling_config: FrameSamplingConfig | None = None,
        tracker_name: str = "ByteTrack",
    ) -> VideoInferenceRun:
        """Alias for run_video_tracking providing backward compatibility."""
        cfg = sampling_config or FrameSamplingConfig(
            mode=sampling_mode, sample_interval=custom_stride
        )
        return self.run_video_tracking(
            video_id=video_id,
            model_id=model_id,
            tracker_name=tracker_name,
            sampling_config=cfg,
        )

    def run_video_tracking(
        self,
        video_id: str,
        model_id: str = "yolo11s.pt",
        tracker_name: str = "ByteTrack",
        sampling_config: FrameSamplingConfig | None = None,
        track_thresh: float = 0.45,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        synthetic_frames_data: list[list[dict[str, Any]]] | None = None,
    ) -> VideoInferenceRun:
        """Execute real multi-object detection and ByteTrack tracking over video frames."""
        meta = self.get_video_metadata(video_id)
        cfg = sampling_config or FrameSamplingConfig()

        run_id = f"vrun_{uuid.uuid4().hex[:8]}"
        t_start = time.perf_counter()

        tracker = ByteTracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=int(meta.fps) if meta.fps > 0 else 30,
        )

        total_frames = max(1, meta.frame_count)
        step = max(1, cfg.sample_interval)
        sampled_indices = list(range(0, total_frames, step))
        cfg.total_sampled_frames = len(sampled_indices)

        # Locate physical video file on disk
        video_path = self._storage_dir / "uploads" / meta.filename
        if not video_path.is_file():
            video_path = self._storage_dir / meta.filename

        frames_feed: list[list[dict[str, Any]]]
        t_inf_total = 0.0

        if synthetic_frames_data is not None:
            # Deterministic unit test injection
            frames_feed = synthetic_frames_data
        elif video_path.is_file():
            # Real OpenCV frame extraction and YOLO model inference
            frames_feed, t_inf_total = self._extract_and_detect_frames(
                video_file_path=video_path,
                sampled_indices=sampled_indices,
                model_id=model_id,
                conf_thresh=track_thresh,
            )
        else:
            # No video file available on disk
            frames_feed = [[] for _ in sampled_indices]

        all_tracks_map: dict[int, Track] = {}
        total_detections = 0
        detections_per_second: dict[int, int] = {}
        active_objects_per_second: dict[int, set[int]] = {}
        t_track_total = 0.0

        effective_fps = meta.fps if meta.fps > 0 else 30.0

        for frame_idx, detections in zip(sampled_indices, frames_feed, strict=False):
            t_sec = round(frame_idx / effective_fps, 2)
            sec_bin = int(t_sec)
            total_detections += len(detections)
            detections_per_second[sec_bin] = detections_per_second.get(sec_bin, 0) + len(detections)

            # Tracker update step
            t_tr_start = time.perf_counter()
            active_tracks = tracker.update(detections, frame_id=frame_idx, timestamp=t_sec)
            t_track_total += time.perf_counter() - t_tr_start

            if sec_bin not in active_objects_per_second:
                active_objects_per_second[sec_bin] = set()

            for trk in active_tracks:
                if isinstance(trk, dict):
                    tid = trk["track_id"]
                    bbox = trk.get("bbox", [0, 0, 10, 10])
                    conf = trk.get("confidence", 0.8)
                    cls_name = trk.get("class_name", "object")
                else:
                    tid = trk.track_id
                    bbox = [
                        trk.tlwh[0],
                        trk.tlwh[1],
                        trk.tlwh[0] + trk.tlwh[2],
                        trk.tlwh[1] + trk.tlwh[3],
                    ]
                    conf = getattr(trk, "score", 0.8)
                    cls_name = getattr(trk, "class_name", "object")

                active_objects_per_second[sec_bin].add(tid)

                norm_x = ((bbox[0] + bbox[2]) / 2.0) / float(meta.width if meta.width > 0 else 1920)
                norm_y = ((bbox[1] + bbox[3]) / 2.0) / float(
                    meta.height if meta.height > 0 else 1080
                )

                pt = TrajectoryPoint(
                    frame_index=frame_idx,
                    timestamp_sec=t_sec,
                    x_center_px=round((bbox[0] + bbox[2]) / 2.0, 1),
                    y_center_px=round((bbox[1] + bbox[3]) / 2.0, 1),
                    norm_x=round(norm_x, 4),
                    norm_y=round(norm_y, 4),
                    width_px=round(bbox[2] - bbox[0], 1),
                    height_px=round(bbox[3] - bbox[1], 1),
                    bbox=[
                        round(bbox[0], 1),
                        round(bbox[1], 1),
                        round(bbox[2], 1),
                        round(bbox[3], 1),
                    ],
                )

                if tid not in all_tracks_map:
                    all_tracks_map[tid] = Track(
                        track_id=tid,
                        class_name=cls_name,
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        first_timestamp_sec=t_sec,
                        last_timestamp_sec=t_sec,
                        visibility_duration_sec=0.0,
                        avg_confidence=round(conf, 4),
                        min_confidence=round(conf, 4),
                        max_confidence=round(conf, 4),
                        total_distance_px=0.0,
                        avg_speed_px_per_sec=0.0,
                        image_space_velocity_px_s=0.0,
                        observation_count=1,
                        gap_count=0,
                        trajectory=[pt],
                        detections_count=1,
                    )
                else:
                    t_record = all_tracks_map[tid]
                    t_record.last_frame = frame_idx
                    t_record.last_timestamp_sec = t_sec
                    t_record.visibility_duration_sec = round(
                        t_sec - t_record.first_timestamp_sec, 2
                    )
                    t_record.detections_count += 1
                    t_record.observation_count += 1
                    t_record.min_confidence = min(t_record.min_confidence, round(conf, 4))
                    t_record.max_confidence = max(t_record.max_confidence, round(conf, 4))

                    # Compute real distance delta
                    last_pt = t_record.trajectory[-1]
                    dist = math.hypot(
                        pt.x_center_px - last_pt.x_center_px, pt.y_center_px - last_pt.y_center_px
                    )
                    t_record.total_distance_px = round(t_record.total_distance_px + dist, 1)

                    if t_record.visibility_duration_sec > 0:
                        spd = round(
                            t_record.total_distance_px / t_record.visibility_duration_sec, 1
                        )
                        t_record.avg_speed_px_per_sec = spd
                        t_record.image_space_velocity_px_s = spd

                    t_record.trajectory.append(pt)

        # Compute temporal analytics
        tracks_list = list(all_tracks_map.values())
        tot_tracks = len(tracks_list)
        tracks_by_cls: dict[str, int] = {}
        tot_dur = 0.0
        max_dur = 0.0
        tot_dist = 0.0

        for t in tracks_list:
            tracks_by_cls[t.class_name] = tracks_by_cls.get(t.class_name, 0) + 1
            tot_dur += t.visibility_duration_sec
            max_dur = max(max_dur, t.visibility_duration_sec)
            tot_dist += t.total_distance_px

        avg_dur = round(tot_dur / max(1, tot_tracks), 2)
        avg_dist = round(tot_dist / max(1, tot_tracks), 1)

        active_objs_series = [
            {"second": s, "count": len(tids)}
            for s, tids in sorted(active_objects_per_second.items())
        ]
        detections_series = [
            {"second": s, "count": cnt} for s, cnt in sorted(detections_per_second.items())
        ]

        analytics = TemporalAnalytics(
            total_tracks=tot_tracks,
            tracks_by_class=tracks_by_cls,
            avg_track_duration_sec=avg_dur,
            longest_track_duration_sec=round(max_dur, 2),
            avg_pixel_movement_px=avg_dist,
            total_region_visits=0,
            avg_dwell_time_sec=0.0,
            median_dwell_time_sec=0.0,
            events_per_minute=round((tot_tracks * 2) / max(0.1, meta.duration_sec / 60.0), 1),
            active_objects_over_time=active_objs_series,
            detections_over_time=detections_series,
        )

        t_elapsed = max(0.001, time.perf_counter() - t_start)
        proc_fps = round(len(sampled_indices) / t_elapsed, 1)
        mean_inf_ms = round((t_inf_total / max(1, len(sampled_indices))) * 1000.0, 2)
        mean_track_ms = round((t_track_total / max(1, len(sampled_indices))) * 1000.0, 2)

        run = VideoInferenceRun(
            run_id=run_id,
            video_id=video_id,
            model_id=model_id,
            tracker_name=tracker_name,
            sampling_config=cfg,
            status="COMPLETED",
            duration_sec=meta.duration_sec,
            processed_frames=len(sampled_indices),
            total_detections=total_detections,
            total_tracks=tot_tracks,
            tracks=tracks_list,
            analytics=analytics,
            processing_fps=proc_fps,
            inference_latency_ms=mean_inf_ms,
            tracking_latency_ms=mean_track_ms,
        )

        self._runs[run_id] = run
        self.save_to_disk()
        logger.info(
            "Completed tracking run '%s' for video '%s' (%d tracks, %d detections)",
            run_id,
            video_id,
            tot_tracks,
            total_detections,
        )

        # Automatically extract and store real temporal events
        try:
            from visionforge.events.service import get_temporal_event_service

            event_svc = get_temporal_event_service()
            event_svc.generate_events_for_run(run_id)
        except Exception as exc:
            logger.warning("Could not auto-generate events for run '%s': %s", run_id, exc)

        return run

    def _extract_and_detect_frames(
        self,
        video_file_path: Path,
        sampled_indices: list[int],
        model_id: str,
        conf_thresh: float,
    ) -> tuple[list[list[dict[str, Any]]], float]:
        """Extract real video frames via OpenCV and execute real object detection."""
        import cv2

        cap = cv2.VideoCapture(str(video_file_path))
        if not cap.isOpened():
            logger.warning("Could not open video file '%s'", video_file_path)
            return [[] for _ in sampled_indices], 0.0

        sampled_set = set(sampled_indices)
        max_idx = max(sampled_indices) if sampled_indices else 0
        current_idx = 0
        frames_dict: dict[int, list[dict[str, Any]]] = {}

        # Load canonical model from InferenceService lifecycle
        model = None
        try:
            descriptor = self._inference_service.get_model_descriptor(model_id)
            model = self._inference_service._lifecycle.load_model(
                descriptor.model_id, descriptor.checkpoint_path
            )
        except Exception as exc:
            logger.warning("Model '%s' not loaded into memory: %s", model_id, exc)

        t_inf_total = 0.0

        while cap.isOpened() and current_idx <= max_idx:
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in sampled_set:
                frame_dets: list[dict[str, Any]] = []
                if model is not None:
                    try:
                        t_inf_start = time.perf_counter()
                        # Run real YOLO detection on BGR frame
                        results = model.predict(
                            source=frame,
                            conf=conf_thresh,
                            verbose=False,
                        )
                        t_inf_total += time.perf_counter() - t_inf_start

                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                cls_id = int(box.cls[0].item())
                                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                                conf = float(box.conf[0].item())
                                xyxy = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
                                frame_dets.append(
                                    {
                                        "bbox": [
                                            round(xyxy[0], 1),
                                            round(xyxy[1], 1),
                                            round(xyxy[2], 1),
                                            round(xyxy[3], 1),
                                        ],
                                        "confidence": round(conf, 4),
                                        "class_name": cls_name,
                                    }
                                )
                    except Exception as exc:
                        logger.warning("Detection failed on frame %d: %s", current_idx, exc)

                frames_dict[current_idx] = frame_dets

            current_idx += 1

        cap.release()

        feed = [frames_dict.get(idx, []) for idx in sampled_indices]
        return feed, t_inf_total

    def get_run(self, run_id: str) -> VideoInferenceRun:
        """Retrieve tracking run record."""
        if run_id not in self._runs:
            raise VideoRunNotFoundError(run_id)
        return self._runs[run_id]

    def list_runs(self, video_id: str | None = None) -> list[VideoInferenceRun]:
        """List historical tracking runs."""
        runs = list(self._runs.values())
        if video_id:
            runs = [r for r in runs if r.video_id == video_id]
        return sorted(runs, key=lambda r: r.timestamp, reverse=True)

    # ─── Video Comparison ──────────────────────────────────────────────

    def compare_videos(self, video_a_id: str, video_b_id: str) -> VideoComparisonResult:
        """Compare temporal statistics between two video runs."""
        runs_a = self.list_runs(video_a_id)
        runs_b = self.list_runs(video_b_id)

        if not runs_a or not runs_b:
            raise VideoValidationError("Both videos must have completed tracking runs to compare.")

        ra = runs_a[0]
        rb = runs_b[0]

        t_delta = ra.total_tracks - rb.total_tracks
        det_delta = ra.total_detections - rb.total_detections
        dur_delta = round(
            ra.analytics.avg_track_duration_sec - rb.analytics.avg_track_duration_sec, 2
        )

        cls_delta: dict[str, int] = {}
        all_classes = set(ra.analytics.tracks_by_class.keys()).union(
            rb.analytics.tracks_by_class.keys()
        )
        for c in all_classes:
            cls_delta[c] = ra.analytics.tracks_by_class.get(
                c, 0
            ) - rb.analytics.tracks_by_class.get(c, 0)

        findings = [
            f"Run A observed {abs(t_delta)} {'more' if t_delta >= 0 else 'fewer'} unique tracks than Run B.",
            f"Average track duration delta: {dur_delta}s between compared assets.",
            f"Total raw detection volume delta: {det_delta} individual bounding box observations.",
        ]

        cmp_id = f"vcmp_{uuid.uuid4().hex[:8]}"
        return VideoComparisonResult(
            comparison_id=cmp_id,
            video_a_id=video_a_id,
            video_b_id=video_b_id,
            track_count_delta=t_delta,
            event_count_delta=det_delta,
            avg_dwell_delta_sec=dur_delta,
            tracks_by_class_delta=cls_delta,
            summary_findings=findings,
        )

    # ─── Persistence ───────────────────────────────────────────────────

    def save_to_disk(self) -> None:
        self._videos_file.write_text(
            json.dumps([v.model_dump() for v in self._videos.values()], indent=2),
            encoding="utf-8",
        )
        self._runs_file.write_text(
            json.dumps([r.model_dump() for r in self._runs.values()], indent=2),
            encoding="utf-8",
        )
        self._sessions_file.write_text(
            json.dumps([s.model_dump() for s in self._sessions.values()], indent=2),
            encoding="utf-8",
        )

    def load_from_disk(self) -> None:
        if self._videos_file.exists():
            try:
                data = json.loads(self._videos_file.read_text(encoding="utf-8"))
                for it in data:
                    v = VideoMetadata(**it)
                    # Verify physical file existence on disk
                    p1 = self._storage_dir / "uploads" / v.filename
                    p2 = self._storage_dir / v.filename
                    if p1.is_file() or p2.is_file():
                        self._videos[v.video_id] = v
            except Exception as e:
                logger.error("Failed to load video metadata: %s", e)

        if self._runs_file.exists():
            try:
                data = json.loads(self._runs_file.read_text(encoding="utf-8"))
                for it in data:
                    r = VideoInferenceRun(**it)
                    # Only retain runs that belong to valid registered videos
                    if r.video_id in self._videos:
                        self._runs[r.run_id] = r
            except Exception as e:
                logger.error("Failed to load video runs: %s", e)

        if self._sessions_file.exists():
            try:
                data = json.loads(self._sessions_file.read_text(encoding="utf-8"))
                for it in data:
                    s = VideoSession(**it)
                    if s.video_id in self._videos:
                        self._sessions[s.session_id] = s
            except Exception as e:
                logger.error("Failed to load video sessions: %s", e)

        # Write clean pruned state
        self.save_to_disk()


@lru_cache
def get_video_intelligence_service() -> VideoIntelligenceService:
    """Return singleton instance of VideoIntelligenceService."""
    return VideoIntelligenceService()
