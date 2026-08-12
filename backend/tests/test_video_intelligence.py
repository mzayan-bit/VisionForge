"""Unit and Integration Tests for Video Intelligence & Multi-Object Tracking System."""

import pytest
from fastapi.testclient import TestClient

from visionforge.main import app
from visionforge.video.schemas import FrameSamplingMode
from visionforge.video.service import VideoIntelligenceService, VideoValidationError
from visionforge.video.tracker import ByteTracker, compute_iou

client = TestClient(app)


def test_compute_iou():
    """Verify IoU calculation logic."""
    bbox1 = [100.0, 100.0, 200.0, 200.0]
    bbox2 = [150.0, 100.0, 250.0, 200.0]
    iou = compute_iou(bbox1, bbox2)
    assert iou > 0.3
    assert iou < 0.5

    # Non-overlapping boxes
    bbox3 = [300.0, 300.0, 400.0, 400.0]
    assert compute_iou(bbox1, bbox3) == 0.0


def test_bytetrack_tracker_persistent_ids():
    """Verify ByteTracker maintains persistent Track IDs across consecutive frames."""
    tracker = ByteTracker(iou_threshold=0.3, max_lost_frames=30)

    # Frame 1: Person at (100, 100) -> (200, 300)
    dets_f1 = [
        {"class_name": "person", "confidence": 0.90, "bbox": [100.0, 100.0, 200.0, 300.0]}
    ]
    res_f1 = tracker.update(
        frame_index=0, timestamp_sec=0.0, detections=dets_f1, img_width=1920, img_height=1080
    )
    assert len(res_f1) == 1
    track_id_1 = res_f1[0]["track_id"]
    assert track_id_1 == 1

    # Frame 2: Person moved slightly to (110, 100) -> (210, 300)
    dets_f2 = [
        {"class_name": "person", "confidence": 0.92, "bbox": [110.0, 100.0, 210.0, 300.0]}
    ]
    res_f2 = tracker.update(
        frame_index=1, timestamp_sec=0.033, detections=dets_f2, img_width=1920, img_height=1080
    )
    assert len(res_f2) == 1
    assert res_f2[0]["track_id"] == track_id_1  # Persistent Track ID maintained!


def test_trajectory_and_pixel_speed_calculation():
    """Verify spatial trajectory points, distance, and pixel speed calculation."""
    tracker = ByteTracker(iou_threshold=0.3)

    # Move object across 3 frames
    for idx in range(3):
        t_sec = idx * 1.0
        x_min = 100.0 + (idx * 30.0)  # Move 30px per frame (overlapping IoU)
        dets = [{"class_name": "car", "confidence": 0.88, "bbox": [x_min, 200.0, x_min + 100.0, 300.0]}]
        tracker.update(frame_index=idx, timestamp_sec=t_sec, detections=dets, img_width=1920, img_height=1080)

    tracks = tracker.get_all_tracks()
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.track_id == 1
    assert tr.class_name == "car"
    assert len(tr.trajectory) == 3
    assert tr.total_distance_px > 50.0  # Traversed ~60px
    assert tr.avg_speed_px_per_sec > 25.0  # ~30 px/sec speed


def test_video_intelligence_service_metadata_and_validation(tmp_path):
    """Verify video registration and metadata extraction."""
    service = VideoIntelligenceService(storage_dir=tmp_path)

    # Test non-existent file error
    with pytest.raises(VideoValidationError):
        service.register_video("/non/existent/video.mp4")

    # Create dummy video file
    dummy_vid = tmp_path / "test.mp4"
    dummy_vid.write_bytes(b"dummy video binary data")

    meta = service.register_video(str(dummy_vid))
    assert meta.filename == "test.mp4"
    assert meta.duration_sec > 0.0
    assert meta.fps > 0.0


def test_video_intelligence_service_execute_inference(tmp_path):
    """Verify complete video inference pipeline execution and temporal analytics."""
    service = VideoIntelligenceService(storage_dir=tmp_path)

    run = service.execute_video_inference(
        video_id="test_vid_01",
        model_id="yolo11s.pt",
        sampling_mode=FrameSamplingMode.EVERY_2ND_FRAME,
    )

    assert run.run_id.startswith("vrun_")
    assert run.video_id == "test_vid_01"
    assert run.tracker_name == "ByteTrack"
    assert run.processed_frames > 0
    assert run.total_tracks > 0
    assert len(run.tracks) > 0
    assert run.analytics.total_tracks == run.total_tracks
    assert run.processing_fps > 0.0


def test_video_api_endpoints():
    """Test video intelligence REST API routes."""
    # 1. Create Video Run
    res_create = client.post(
        "/api/v1/video/runs",
        json={
            "video_id": "sample_traffic_01",
            "model_id": "yolo11s.pt",
            "sampling_mode": "EVERY_2ND_FRAME",
        },
    )
    assert res_create.status_code == 201
    run_data = res_create.json()
    run_id = run_data["run_id"]
    assert run_data["tracker_name"] == "ByteTrack"

    # 2. List Runs
    res_list = client.get("/api/v1/video/runs")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1

    # 3. Get Run Detail
    res_get = client.get(f"/api/v1/video/runs/{run_id}")
    assert res_get.status_code == 200
    assert res_get.json()["run_id"] == run_id

    # 4. Get Tracks
    res_tracks = client.get(f"/api/v1/video/runs/{run_id}/tracks")
    assert res_tracks.status_code == 200
    assert len(res_tracks.json()) > 0

    # 5. Get Temporal Analytics
    res_analytics = client.get(f"/api/v1/video/runs/{run_id}/analytics")
    assert res_analytics.status_code == 200
    assert "total_tracks" in res_analytics.json()

    # 6. Export Trajectories CSV
    res_export = client.get(f"/api/v1/video/runs/{run_id}/export")
    assert res_export.status_code == 200
    assert "data" in res_export.json()
    assert "run_id,video_id,track_id" in res_export.json()["data"]
