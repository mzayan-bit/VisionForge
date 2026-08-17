"""Unit and Integration Tests for Temporal Event Intelligence System."""

from fastapi.testclient import TestClient

from visionforge.events.detector import (
    TemporalEventDetector,
    is_point_in_polygon,
    is_point_in_rectangle,
)
from visionforge.events.schemas import (
    EventRuleConfig,
    EventType,
    RegionOfInterest,
    RegionShape,
)
from visionforge.main import app
from visionforge.video.schemas import FrameSamplingConfig, Track, TrajectoryPoint, VideoInferenceRun

client = TestClient(app)


def test_point_in_rectangle_and_polygon_geometry():
    """Verify point intersection in rectangle and polygon regions."""
    rect = [[100.0, 100.0], [500.0, 500.0]]
    assert is_point_in_rectangle((200.0, 200.0), rect) is True
    assert is_point_in_rectangle((50.0, 200.0), rect) is False

    poly = [[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]]
    assert is_point_in_polygon((200.0, 200.0), poly) is True
    assert is_point_in_polygon((500.0, 500.0), poly) is False


def create_synthetic_run() -> VideoInferenceRun:
    """Construct synthetic VideoInferenceRun with deterministic trajectories."""
    # Track 1: Starts outside, enters Region A at t=2.0s, stays until t=6.0s (dwell 4.0s), exits at t=7.0s
    traj_1: list[TrajectoryPoint] = []
    for f_idx in range(10):
        t_sec = float(f_idx)
        # Move from (50, 50) into (300, 300) inside rect [[100, 100], [500, 500]]
        x = 50.0 if f_idx < 2 else 300.0 if f_idx < 7 else 600.0
        y = 50.0 if f_idx < 2 else 300.0 if f_idx < 7 else 600.0
        traj_1.append(
            TrajectoryPoint(
                frame_index=f_idx * 30,
                timestamp_sec=t_sec,
                x_center_px=x,
                y_center_px=y,
                norm_x=x / 1920.0,
                norm_y=y / 1080.0,
                width_px=100.0,
                height_px=100.0,
                bbox=[x - 50.0, y - 50.0, x + 50.0, y + 50.0],
            )
        )

    tr1 = Track(
        track_id=1,
        class_name="person",
        first_frame=0,
        last_frame=270,
        first_timestamp_sec=0.0,
        last_timestamp_sec=9.0,
        visibility_duration_sec=9.0,
        avg_confidence=0.90,
        min_confidence=0.85,
        max_confidence=0.95,
        total_distance_px=500.0,
        avg_speed_px_per_sec=55.5,
        status="TERMINATED",
        trajectory=traj_1,
        detections_count=10,
    )

    # Track 2: Approaches Track 1 at t=3.0s (became close), separates at t=6.0s
    traj_2: list[TrajectoryPoint] = []
    for f_idx in range(10):
        t_sec = float(f_idx)
        x = 600.0 if f_idx < 3 else 320.0 if f_idx < 6 else 750.0
        y = 600.0 if f_idx < 3 else 320.0 if f_idx < 6 else 750.0
        traj_2.append(
            TrajectoryPoint(
                frame_index=f_idx * 30,
                timestamp_sec=t_sec,
                x_center_px=x,
                y_center_px=y,
                norm_x=x / 1920.0,
                norm_y=y / 1080.0,
                width_px=100.0,
                height_px=100.0,
                bbox=[x - 50.0, y - 50.0, x + 50.0, y + 50.0],
            )
        )

    tr2 = Track(
        track_id=2,
        class_name="person",
        first_frame=0,
        last_frame=270,
        first_timestamp_sec=0.0,
        last_timestamp_sec=9.0,
        visibility_duration_sec=9.0,
        avg_confidence=0.88,
        min_confidence=0.80,
        max_confidence=0.92,
        total_distance_px=500.0,
        avg_speed_px_per_sec=55.5,
        status="TERMINATED",
        trajectory=traj_2,
        detections_count=10,
    )

    return VideoInferenceRun(
        run_id="vrun_synthetic_001",
        video_id="vid_synth_01",
        model_id="yolo11s.pt",
        tracker_name="ByteTrack",
        sampling_config=FrameSamplingConfig(
            mode="EVERY_FRAME", sample_interval=1, total_sampled_frames=10
        ),
        duration_sec=10.0,
        processed_frames=10,
        total_detections=20,
        total_tracks=2,
        tracks=[tr1, tr2],
        analytics={
            "total_tracks": 2,
            "tracks_by_class": {"person": 2},
            "avg_track_duration_sec": 9.0,
            "longest_track_duration_sec": 9.0,
            "avg_pixel_movement_px": 500.0,
            "active_objects_over_time": [
                {"second": 0, "active_count": 2},
                {"second": 5, "active_count": 2},
                {"second": 8, "active_count": 1},
            ],
            "detections_over_time": [{"second": 0, "detection_count": 2}],
        },
        processing_fps=120.0,
        inference_latency_ms=5.0,
        tracking_latency_ms=0.5,
    )


def test_temporal_event_detector_rule_logic():
    """Verify rule-based TemporalEventDetector generates expected events."""
    run = create_synthetic_run()
    region_a = RegionOfInterest(
        region_id="reg_a",
        video_id="vid_synth_01",
        name="Loading Zone A",
        shape_type=RegionShape.RECTANGLE,
        coordinates=[[100.0, 100.0], [500.0, 500.0]],
    )

    cfg = EventRuleConfig(
        dwell_threshold_sec=3.0, proximity_threshold_px=50.0, separation_threshold_px=150.0
    )
    detector = TemporalEventDetector(config=cfg)

    events = detector.detect_events(run, [region_a])
    assert len(events) > 0

    event_types = [e.event_type for e in events]
    assert EventType.TRACK_STARTED in event_types
    assert EventType.OBJECT_ENTERED_REGION in event_types
    assert EventType.OBJECT_DWELLED in event_types
    assert EventType.OBJECTS_BECAME_CLOSE in event_types

    # Verify OBJECT_DWELLED event details
    dwell_evts = [e for e in events if e.event_type == EventType.OBJECT_DWELLED]
    assert len(dwell_evts) >= 1
    assert dwell_evts[0].duration_sec >= 3.0
    assert dwell_evts[0].event_params["region_name"] == "Loading Zone A"


def test_temporal_event_service_and_api_endpoints(tmp_path):
    """Test TemporalEventService and REST API endpoints."""
    # 1. API: Create Region 1
    res_reg1 = client.post(
        "/api/v1/events/regions",
        json={
            "video_id": "sample_traffic_01",
            "name": "Zone B",
            "coordinates": [[200.0, 200.0], [800.0, 800.0]],
        },
    )
    assert res_reg1.status_code == 201

    # 2. API: Create Region 2
    res_reg2 = client.post(
        "/api/v1/events/regions",
        json={
            "video_id": "sample_traffic_01",
            "name": "Restricted Corridor",
            "coordinates": [[100.0, 100.0], [600.0, 600.0]],
        },
    )
    assert res_reg2.status_code == 201
    assert res_reg2.json()["name"] == "Restricted Corridor"

    # 3. API: List Regions
    res_list_reg = client.get("/api/v1/events/regions?video_id=sample_traffic_01")
    assert res_list_reg.status_code == 200
    assert len(res_list_reg.json()) >= 2

    # 4. First run video pipeline via API to generate run
    client.post(
        "/api/v1/video/runs",
        json={
            "video_id": "sample_traffic_01",
            "model_id": "yolo11s.pt",
            "sampling_mode": "EVERY_2ND_FRAME",
        },
    )
    res_vruns = client.get("/api/v1/video/runs")
    run_id = res_vruns.json()[0]["run_id"]

    # 5. API: Generate Events
    res_gen = client.post(
        "/api/v1/events/generate",
        json={"run_id": run_id},
    )
    assert res_gen.status_code == 201
    evts_data = res_gen.json()
    assert len(evts_data) > 0
    evt_id = evts_data[0]["event_id"]

    # 6. API: Get Event Detail
    res_evt = client.get(f"/api/v1/events/{evt_id}")
    assert res_evt.status_code == 200
    assert res_evt.json()["event_id"] == evt_id

    # 7. API: Get Evidence
    res_evid = client.get(f"/api/v1/events/{evt_id}/evidence")
    assert res_evid.status_code == 200
    assert "frame_before_idx" in res_evid.json()

    # 8. API: Get Analytics & Summary
    res_analytics = client.get(f"/api/v1/events/runs/{run_id}/analytics")
    assert res_analytics.status_code == 200
    assert "events_by_type" in res_analytics.json()

    res_summary = client.get(f"/api/v1/events/runs/{run_id}/summary")
    assert res_summary.status_code == 200
    assert res_summary.json()["total_events"] > 0

    # 9. API: Export Events CSV
    res_exp = client.get(f"/api/v1/events/runs/{run_id}/export")
    assert res_exp.status_code == 200
    assert "event_id,run_id,video_id" in res_exp.json()["data"]
