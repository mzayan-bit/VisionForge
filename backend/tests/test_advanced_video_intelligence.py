"""Comprehensive unit tests for Advanced Video Understanding and Temporal Intelligence."""

from fastapi.testclient import TestClient

from tests.test_video_intelligence import create_real_test_video
from visionforge.events.detector import TemporalEventDetector
from visionforge.events.schemas import (
    EventRuleConfig,
    EventType,
    RegionOfInterest,
    RegionShape,
)
from visionforge.main import app
from visionforge.query.interpreter import QueryInterpreter
from visionforge.query.schemas import QueryStatus
from visionforge.video.schemas import (
    FrameSamplingConfig,
    FrameSamplingMode,
    Track,
    TrajectoryPoint,
    VideoInferenceRun,
)
from visionforge.video.service import VideoIntelligenceService

client = TestClient(app)


def build_synthetic_enter_dwell_exit_run() -> VideoInferenceRun:
    """Build a deterministic synthetic run where a track moves into a region, dwells, and exits."""
    # 30 frames, fps=10.0 (3 seconds total)
    # Region: [200, 200] to [600, 600]
    # Track 1:
    # Frame 0-5 (t=0.0 to 0.5s): at (100, 100) -> OUTSIDE
    # Frame 6-25 (t=0.6 to 2.5s): at (300, 300) -> INSIDE (dwells ~1.9s)
    # Frame 26-29 (t=2.6 to 2.9s): at (800, 800) -> OUTSIDE
    trajectory = []
    for f in range(30):
        t = round(f / 10.0, 2)
        if f <= 5:
            cx, cy = 100.0, 100.0
        elif f <= 25:
            cx, cy = 300.0, 300.0
        else:
            cx, cy = 800.0, 800.0

        trajectory.append(
            TrajectoryPoint(
                frame_index=f,
                timestamp_sec=t,
                x_center_px=cx,
                y_center_px=cy,
                norm_x=cx / 1920.0,
                norm_y=cy / 1080.0,
                width_px=50.0,
                height_px=100.0,
                bbox=[cx - 25.0, cy - 50.0, cx + 25.0, cy + 50.0],
            )
        )

    track = Track(
        track_id=101,
        class_name="person",
        first_frame=0,
        last_frame=29,
        first_timestamp_sec=0.0,
        last_timestamp_sec=2.9,
        visibility_duration_sec=2.9,
        avg_confidence=0.92,
        min_confidence=0.88,
        max_confidence=0.95,
        total_distance_px=900.0,
        avg_speed_px_per_sec=310.0,
        image_space_velocity_px_s=310.0,
        observation_count=30,
        gap_count=0,
        trajectory=trajectory,
        detections_count=30,
    )

    from visionforge.video.schemas import TemporalAnalytics

    analytics = TemporalAnalytics(
        total_tracks=1,
        tracks_by_class={"person": 1},
        avg_track_duration_sec=2.9,
        longest_track_duration_sec=2.9,
        avg_pixel_movement_px=900.0,
        total_region_visits=1,
        avg_dwell_time_sec=1.9,
        median_dwell_time_sec=1.9,
        events_per_minute=20.0,
        active_objects_over_time=[{"second": 0.0, "count": 1}],
        detections_over_time=[{"second": 0.0, "count": 1}],
    )

    return VideoInferenceRun(
        run_id="vrun_synthetic_test",
        video_id="vid_synthetic_zone",
        model_id="yolo11s.pt",
        tracker_name="ByteTrack",
        sampling_config=FrameSamplingConfig(mode=FrameSamplingMode.EVERY_FRAME, sample_interval=1),
        status="COMPLETED",
        duration_sec=3.0,
        processed_frames=30,
        total_detections=30,
        total_tracks=1,
        tracks=[track],
        analytics=analytics,
        processing_fps=30.0,
        inference_latency_ms=12.5,
        tracking_latency_ms=1.2,
    )


def test_zone_entry_dwell_exit_detection():
    """Verify exact trigger rules and evidence for enter, dwell, and exit."""
    run = build_synthetic_enter_dwell_exit_run()
    zone = RegionOfInterest(
        region_id="reg_staging",
        video_id="vid_synthetic_zone",
        name="Staging Area",
        shape_type=RegionShape.RECTANGLE,
        coordinates=[[200.0, 200.0], [600.0, 600.0]],
    )

    detector = TemporalEventDetector(config=EventRuleConfig(dwell_threshold_sec=1.5))
    events = detector.detect_events(run, [zone])

    types = [e.event_type for e in events]
    assert EventType.TRACK_STARTED in types
    assert EventType.OBJECT_ENTERED_REGION in types
    assert EventType.OBJECT_LEFT_REGION in types
    assert EventType.OBJECT_DWELLED in types
    assert EventType.TRACK_ENDED in types

    # Verify enter event properties
    enter_evt = next(e for e in events if e.event_type == EventType.OBJECT_ENTERED_REGION)
    assert enter_evt.source_track_ids == [101]
    assert enter_evt.start_timestamp_sec == 0.6
    assert enter_evt.evidence is not None
    assert enter_evt.evidence.trigger_rule != ""
    assert "entered region 'Staging Area'" in enter_evt.description

    # Verify dwell event properties
    dwell_evt = next(e for e in events if e.event_type == EventType.OBJECT_DWELLED)
    assert dwell_evt.source_track_ids == [101]
    assert dwell_evt.duration_sec >= 1.5
    assert dwell_evt.evidence is not None
    assert "dwelled" in dwell_evt.trigger_rule or "maintained" in dwell_evt.trigger_rule


def test_video_comparison_functionality(tmp_path):
    """Verify comparing two video runs side-by-side."""
    service = VideoIntelligenceService(storage_dir=tmp_path)

    # Create real test videos
    vid_a = tmp_path / "vid_a.mp4"
    vid_b = tmp_path / "vid_b.mp4"
    create_real_test_video(vid_a, width=320, height=240, fps=30, frames=10)
    create_real_test_video(vid_b, width=320, height=240, fps=30, frames=15)
    meta_a = service.register_video(str(vid_a))
    meta_b = service.register_video(str(vid_b))

    # Run tracking for both
    feed_a = [[{"class_name": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50]}]]
    feed_b = [[{"class_name": "car", "confidence": 0.9, "bbox": [20, 20, 80, 80]}]]
    service.run_video_tracking(meta_a.video_id, synthetic_frames_data=feed_a)
    service.run_video_tracking(meta_b.video_id, synthetic_frames_data=feed_b)

    result = service.compare_videos(meta_a.video_id, meta_b.video_id)
    assert result.comparison_id.startswith("vcmp_")
    assert result.video_a_id == meta_a.video_id
    assert result.video_b_id == meta_b.video_id
    assert len(result.summary_findings) >= 1


def test_temporal_query_interpretation():
    """Verify natural language to temporal query parsing."""
    interpreter = QueryInterpreter()

    # Query 1: Entered region
    res1 = interpreter.parse_query(
        "What objects entered Zone A?", run_id="vrun_1", active_region_names=["Zone A"]
    )
    assert res1.status == QueryStatus.SUCCESS
    assert res1.query.event_type == EventType.OBJECT_ENTERED_REGION

    # Query 2: Dwelled in region
    res2 = interpreter.parse_query(
        "Which person stayed in Zone A?", run_id="vrun_1", active_region_names=["Zone A"]
    )
    assert res2.status == QueryStatus.SUCCESS
    assert res2.query.object_class == "person"

    # Query 3: Unsupported action
    res3 = interpreter.parse_query("What was the person talking about?", run_id="vrun_1")
    assert res3.status == QueryStatus.UNSUPPORTED


def test_video_session_and_lineage_api(tmp_path):
    """Verify video sessions and lineage REST API."""
    vid_file = tmp_path / "sess_test.mp4"
    create_real_test_video(vid_file, width=320, height=240, fps=30, frames=10)
    with open(vid_file, "rb") as f:
        res_upload = client.post(
            "/api/v1/video/upload",
            files={"file": ("sess_test.mp4", f, "video/mp4")},
        )
    assert res_upload.status_code == 201
    meta = res_upload.json()

    res_create = client.post(
        "/api/v1/video/sessions",
        json={"video_id": meta["video_id"], "model_version": "1.0.0"},
    )
    assert res_create.status_code == 201

    res_sess = client.get("/api/v1/video/sessions")
    assert res_sess.status_code == 200
    sessions = res_sess.json()
    assert len(sessions) >= 1
    assert "video_fingerprint" in sessions[0]
    assert "lineage" in sessions[0]
