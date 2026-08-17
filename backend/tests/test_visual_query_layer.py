"""Unit and Integration Tests for Visual Query Layer System."""

from fastapi.testclient import TestClient

from visionforge.events.service import TemporalEventService
from visionforge.main import app
from visionforge.query.interpreter import QueryInterpreter
from visionforge.query.schemas import (
    AggregationType,
    QueryStatus,
    QueryType,
)
from visionforge.query.service import VisualQueryService
from visionforge.video.schemas import FrameSamplingConfig, Track, TrajectoryPoint, VideoInferenceRun
from visionforge.video.service import get_video_intelligence_service

client = TestClient(app)


def test_natural_language_query_interpreter():
    """Verify QueryInterpreter parses natural language questions into structured VisualQuery DSL."""
    interpreter = QueryInterpreter()
    active_regions = ["Loading Zone A", "Restricted Corridor"]

    # 1. Event Search Query
    res1 = interpreter.parse_query(
        "Which objects entered Loading Zone A?", "run_01", active_regions
    )
    assert res1.status == QueryStatus.SUCCESS
    assert res1.query.query_type == QueryType.EVENT_SEARCH
    assert res1.query.event_type == "OBJECT_ENTERED_REGION"
    assert res1.query.region_name == "Loading Zone A"

    # 2. Track Search Query with Duration
    res2 = interpreter.parse_query(
        "Which person tracks stayed longer than 5 seconds?", "run_01", active_regions
    )
    assert res2.status == QueryStatus.SUCCESS
    assert res2.query.object_class == "person"
    assert res2.query.min_duration_sec == 5.0

    # 3. Object Count at Timestamp Query
    res3 = interpreter.parse_query(
        "How many people were present at 10 seconds?", "run_01", active_regions
    )
    assert res3.status == QueryStatus.SUCCESS
    assert res3.query.query_type == QueryType.OBJECT_COUNT
    assert res3.query.object_class == "person"
    assert res3.query.at_timestamp_sec == 10.0

    # 4. Aggregation Query (Longest Dwell)
    res4 = interpreter.parse_query(
        "Which track stayed longest in Loading Zone A?", "run_01", active_regions
    )
    assert res4.status == QueryStatus.SUCCESS
    assert res4.query.aggregation == AggregationType.MAX

    # 5. Time Range Search Query
    res5 = interpreter.parse_query(
        "What happened between 5 and 15 seconds?", "run_01", active_regions
    )
    assert res5.status == QueryStatus.SUCCESS
    assert res5.query.time_range == [5.0, 15.0]


def test_ambiguity_and_unsupported_query_handling():
    """Verify handling of ambiguous region questions and unsupported action/semantic queries."""
    interpreter = QueryInterpreter()
    active_regions = ["Zone A", "Zone B"]

    # Ambiguous Region Query
    res_amb = interpreter.parse_query("Show objects in the zone.", "run_01", active_regions)
    assert res_amb.status == QueryStatus.AMBIGUOUS
    assert "Multiple active regions exist" in res_amb.explanation

    # Unsupported Action Recognition Query
    res_unsupported = interpreter.parse_query(
        "What was the person doing?", "run_01", active_regions
    )
    assert res_unsupported.status == QueryStatus.UNSUPPORTED
    assert "Unsupported Query" in res_unsupported.explanation


def create_test_video_run() -> VideoInferenceRun:
    """Create test VideoInferenceRun with deterministic tracks."""
    traj1 = [
        TrajectoryPoint(
            frame_index=f * 30,
            timestamp_sec=float(f),
            x_center_px=200.0,
            y_center_px=200.0,
            norm_x=0.1,
            norm_y=0.1,
            width_px=50.0,
            height_px=50.0,
            bbox=[175.0, 175.0, 225.0, 225.0],
        )
        for f in range(12)
    ]
    tr1 = Track(
        track_id=4,
        class_name="person",
        first_frame=0,
        last_frame=330,
        first_timestamp_sec=0.0,
        last_timestamp_sec=11.0,
        visibility_duration_sec=11.0,
        avg_confidence=0.92,
        min_confidence=0.88,
        max_confidence=0.96,
        total_distance_px=200.0,
        avg_speed_px_per_sec=18.0,
        status="TERMINATED",
        trajectory=traj1,
        detections_count=12,
    )

    return VideoInferenceRun(
        run_id="vrun_query_test_01",
        video_id="sample_traffic_01",
        model_id="yolo11s.pt",
        tracker_name="ByteTrack",
        sampling_config=FrameSamplingConfig(
            mode="EVERY_FRAME", sample_interval=1, total_sampled_frames=12
        ),
        duration_sec=12.0,
        processed_frames=12,
        total_detections=12,
        total_tracks=1,
        tracks=[tr1],
        analytics={
            "total_tracks": 1,
            "tracks_by_class": {"person": 1},
            "avg_track_duration_sec": 11.0,
            "longest_track_duration_sec": 11.0,
            "avg_pixel_movement_px": 200.0,
            "active_objects_over_time": [{"second": 5, "active_count": 1}],
            "detections_over_time": [{"second": 5, "detection_count": 1}],
        },
        processing_fps=120.0,
        inference_latency_ms=4.0,
        tracking_latency_ms=0.4,
    )


def test_visual_query_service_and_api_endpoints(tmp_path):
    """Test VisualQueryService and REST API endpoints end-to-end."""
    video_svc = get_video_intelligence_service()
    TemporalEventService(storage_dir=tmp_path)
    VisualQueryService(storage_dir=tmp_path)

    # Register test run and region on singleton service
    run = create_test_video_run()
    video_svc._runs[run.run_id] = run
    from visionforge.events.service import get_temporal_event_service

    global_event_svc = get_temporal_event_service()
    global_event_svc.create_region(
        video_id=run.video_id,
        name="Loading Zone A",
        coordinates=[[100.0, 100.0], [500.0, 500.0]],
    )
    global_event_svc.generate_events_for_run(run.run_id)

    # 1. API: Ask Question
    res_ask = client.post(
        "/api/v1/query/ask",
        json={
            "query_text": "Which objects entered Loading Zone A?",
            "run_id": run.run_id,
        },
    )
    assert res_ask.status_code == 200
    data_ask = res_ask.json()
    assert data_ask["status"] == "SUCCESS"
    assert data_ask["source_run_id"] == run.run_id
    assert len(data_ask["records"]) >= 0

    # 2. API: Execute Structured Query
    res_exec = client.post(
        "/api/v1/query/execute",
        json={
            "query": {
                "query_id": "vq_test_struct",
                "run_id": run.run_id,
                "query_type": "OBJECT_COUNT",
                "at_timestamp_sec": 5.0,
                "object_class": "person",
                "original_text": "Count at 5s",
            }
        },
    )
    assert res_exec.status_code == 200
    assert res_exec.json()["records"][0]["active_count"] == 1

    # 3. API: List History
    res_hist = client.get("/api/v1/query/history")
    assert res_hist.status_code == 200
    assert len(res_hist.json()) >= 2

    # 4. API: Get Evidence
    qid = data_ask["query_id"]
    res_evid = client.get(f"/api/v1/query/{qid}/evidence")
    assert res_evid.status_code == 200

    # 5. API: Rerun Query
    res_rerun = client.post(f"/api/v1/query/rerun/{qid}")
    assert res_rerun.status_code == 200
    assert res_rerun.json()["original_query"] == "Which objects entered Loading Zone A?"
