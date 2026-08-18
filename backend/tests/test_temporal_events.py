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
from visionforge.video.schemas import (
    FrameSamplingConfig,
    TemporalAnalytics,
    Track,
    TrajectoryPoint,
    VideoInferenceRun,
)

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
    from tests.test_video_intelligence import create_real_test_video

    vid_file = tmp_path / "events_test.mp4"
    create_real_test_video(vid_file, width=320, height=240, fps=30, frames=10)
    with open(vid_file, "rb") as f:
        res_upload = client.post(
            "/api/v1/video/upload",
            files={"file": ("events_test.mp4", f, "video/mp4")},
        )
    assert res_upload.status_code == 201
    meta = res_upload.json()
    video_id = meta["video_id"]

    # 1. API: Create Region 1
    res_reg1 = client.post(
        "/api/v1/events/regions",
        json={
            "video_id": video_id,
            "name": "Zone B",
            "coordinates": [[200.0, 200.0], [800.0, 800.0]],
        },
    )
    assert res_reg1.status_code == 201

    # 2. API: Create Region 2
    res_reg2 = client.post(
        "/api/v1/events/regions",
        json={
            "video_id": video_id,
            "name": "Restricted Corridor",
            "coordinates": [[100.0, 100.0], [600.0, 600.0]],
        },
    )
    assert res_reg2.status_code == 201
    assert res_reg2.json()["name"] == "Restricted Corridor"

    # 3. API: List Regions
    res_list_reg = client.get(f"/api/v1/events/regions?video_id={video_id}")
    assert res_list_reg.status_code == 200
    assert len(res_list_reg.json()) >= 2

    # 4. First register run with tracks to test event generation
    from visionforge.video.service import get_video_intelligence_service

    video_svc = get_video_intelligence_service()
    run = create_synthetic_run()
    run.video_id = video_id
    video_svc._runs[run.run_id] = run
    run_id = run.run_id

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


def test_zone_crud_move_resize_duplicate_and_persistence(tmp_path):
    """Verify that zones support create, move, resize, rename, duplicate with offset, and delete."""
    from visionforge.events.service import TemporalEventService

    svc = TemporalEventService(storage_dir=tmp_path)

    # 1. Create Rectangle Zone
    reg_rect = svc.create_region(
        video_id="vid_test_01",
        name="Loading Bay Alpha",
        coordinates=[[100.0, 100.0], [500.0, 400.0]],
        shape_type=RegionShape.RECTANGLE,
        color="#3b82f6",
    )
    assert reg_rect.region_id.startswith("reg_")
    assert reg_rect.name == "Loading Bay Alpha"

    # 2. Create Polygon Zone
    poly_coords = [[200.0, 100.0], [600.0, 150.0], [700.0, 500.0], [300.0, 450.0]]
    reg_poly = svc.create_region(
        video_id="vid_test_01",
        name="Restricted Hexagon",
        coordinates=poly_coords,
        shape_type=RegionShape.POLYGON,
        color="#8b5cf6",
    )
    assert reg_poly.shape_type == RegionShape.POLYGON
    assert len(reg_poly.coordinates) == 4

    # 3. Move Zone (Update Coordinates)
    moved_coords = [[150.0, 150.0], [550.0, 450.0]]
    updated = svc.update_region(reg_rect.region_id, coordinates=moved_coords)
    assert updated.coordinates == moved_coords

    # 4. Rename Zone
    renamed = svc.update_region(reg_rect.region_id, name="Primary Freight Zone")
    assert renamed.name == "Primary Freight Zone"

    # 5. Duplicate Zone (verifying distinct ID, (Copy) suffix, and +30px offset)
    dup = svc.duplicate_region(reg_rect.region_id, offset_px=30.0)
    assert dup.region_id != reg_rect.region_id
    assert dup.name == "Primary Freight Zone (Copy)"
    assert dup.coordinates == [[180.0, 180.0], [580.0, 480.0]]

    # 6. Delete Zone
    svc.delete_region(reg_rect.region_id)
    remaining = svc.list_regions("vid_test_01")
    assert len(remaining) == 2
    assert all(r.region_id != reg_rect.region_id for r in remaining)

    # 7. Reload from disk and verify persistence
    svc_reloaded = TemporalEventService(storage_dir=tmp_path)
    svc_reloaded.load_from_disk()
    assert dup.region_id in svc_reloaded._regions
    assert reg_poly.region_id in svc_reloaded._regions


def test_zone_spatial_analysis_and_negative_test():
    """Verify zone entry, dwell, exit events and negative test for unvisited zones."""
    detector = TemporalEventDetector(config=EventRuleConfig(dwell_threshold_sec=2.0))

    # Single track moving through Zone A (at x in [200, 400], y in [200, 400]) from t=0 to t=10
    # t=0..2: outside (100, 100)
    # t=3..7: inside (300, 300) -> 5 seconds dwell!
    # t=8..10: outside (600, 600)
    traj = []
    for f in range(11):
        if f <= 2:
            x, y = 100.0, 100.0
        elif 3 <= f <= 7:
            x, y = 300.0, 300.0
        else:
            x, y = 600.0, 600.0

        traj.append(
            TrajectoryPoint(
                frame_index=f * 30,
                timestamp_sec=float(f),
                x_center_px=x,
                y_center_px=y,
                norm_x=x / 1920.0,
                norm_y=y / 1080.0,
                width_px=40.0,
                height_px=40.0,
                bbox=[x - 20, y - 20, x + 20, y + 20],
            )
        )

    track = Track(
        track_id=1,
        class_id=0,
        class_name="person",
        first_frame=0,
        last_frame=300,
        first_timestamp_sec=0.0,
        last_timestamp_sec=10.0,
        visibility_duration_sec=10.0,
        total_distance_px=500.0,
        avg_speed_px_per_sec=50.0,
        image_space_velocity_px_s=50.0,
        avg_confidence=0.90,
        min_confidence=0.85,
        max_confidence=0.95,
        observation_count=11,
        trajectory=traj,
    )

    run = VideoInferenceRun(
        run_id="vrun_zone_test",
        video_id="vid_test_01",
        model_id="yolo11s.pt",
        tracker_name="ByteTrack",
        sampling_config=FrameSamplingConfig(
            mode="EVERY_FRAME", sample_interval=1, total_sampled_frames=11
        ),
        duration_sec=10.0,
        fps=30.0,
        processed_frames=11,
        total_detections=11,
        total_tracks=1,
        tracks=[track],
        analytics=TemporalAnalytics(
            total_tracks=1,
            tracks_by_class={"person": 1},
            avg_track_duration_sec=10.0,
            longest_track_duration_sec=10.0,
            avg_pixel_movement_px=500.0,
        ),
        processing_fps=60.0,
        inference_latency_ms=10.0,
        tracking_latency_ms=2.0,
    )

    # Active Zone A (which the track enters and dwells in)
    zone_a = RegionOfInterest(
        region_id="reg_zone_a",
        video_id="vid_test_01",
        name="Zone A",
        shape_type=RegionShape.RECTANGLE,
        coordinates=[[200.0, 200.0], [400.0, 400.0]],
    )

    # Negative Test Zone B (located far away at [1000, 1000] -> NO objects should ever trigger events)
    zone_b = RegionOfInterest(
        region_id="reg_zone_b",
        video_id="vid_test_01",
        name="Unvisited Zone B",
        shape_type=RegionShape.RECTANGLE,
        coordinates=[[1000.0, 1000.0], [1500.0, 1500.0]],
    )

    events = detector.detect_events(run, [zone_a, zone_b])

    # 1. Verify Zone A events
    events_a = [e for e in events if e.event_params.get("region_id") == "reg_zone_a"]
    assert any(e.event_type == EventType.OBJECT_ENTERED_REGION for e in events_a)
    assert any(e.event_type == EventType.OBJECT_DWELLED for e in events_a)
    assert any(e.event_type == EventType.OBJECT_LEFT_REGION for e in events_a)

    entry_evt = next(e for e in events_a if e.event_type == EventType.OBJECT_ENTERED_REGION)
    assert entry_evt.start_timestamp_sec == 3.0

    dwell_evt = next(e for e in events_a if e.event_type == EventType.OBJECT_DWELLED)
    assert dwell_evt.duration_sec >= 4.0

    exit_evt = next(e for e in events_a if e.event_type == EventType.OBJECT_LEFT_REGION)
    assert exit_evt.start_timestamp_sec == 8.0

    # 2. Negative Test Verification: Zone B MUST produce ZERO events!
    events_b = [e for e in events if e.event_params.get("region_id") == "reg_zone_b"]
    assert len(events_b) == 0, "Unvisited Zone B must not generate synthetic events!"
