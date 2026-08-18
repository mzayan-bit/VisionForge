"""Unit and Integration Tests for Video Intelligence & Multi-Object Tracking System."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.main import app
from visionforge.video.schemas import FrameSamplingConfig
from visionforge.video.service import VideoIntelligenceService, VideoValidationError
from visionforge.video.tracker import ByteTracker, compute_iou

client = TestClient(app)


def create_real_test_video(
    file_path: Path, width: int = 320, height: int = 240, fps: int = 30, frames: int = 15
) -> Path:
    """Create a valid real MP4 video file on disk for deterministic testing."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, float(fps), (width, height))
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw moving rectangle simulating real object motion
        x = 20 + i * 10
        cv2.rectangle(img, (x, 50), (x + 40, 120), (255, 255, 255), -1)
        out.write(img)
    out.release()
    return file_path


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
    dets_f1 = [{"class_name": "person", "confidence": 0.90, "bbox": [100.0, 100.0, 200.0, 300.0]}]
    res_f1 = tracker.update(
        frame_index=0, timestamp_sec=0.0, detections=dets_f1, img_width=1920, img_height=1080
    )
    assert len(res_f1) == 1
    track_id_1 = res_f1[0]["track_id"]
    assert track_id_1 == 1

    # Frame 2: Person moved slightly to (110, 100) -> (210, 300)
    dets_f2 = [{"class_name": "person", "confidence": 0.92, "bbox": [110.0, 100.0, 210.0, 300.0]}]
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
        dets = [
            {"class_name": "car", "confidence": 0.88, "bbox": [x_min, 200.0, x_min + 100.0, 300.0]}
        ]
        tracker.update(
            frame_index=idx, timestamp_sec=t_sec, detections=dets, img_width=1920, img_height=1080
        )

    tracks = tracker.get_all_tracks()
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.track_id == 1
    assert tr.class_name == "car"
    assert len(tr.trajectory) == 3
    assert tr.total_distance_px > 50.0  # Traversed ~60px
    assert tr.avg_speed_px_per_sec > 25.0  # ~30 px/sec speed


def test_video_intelligence_service_metadata_and_validation(tmp_path):
    """Verify real video registration and metadata extraction."""
    service = VideoIntelligenceService(storage_dir=tmp_path)

    # Test non-existent file error
    with pytest.raises(VideoValidationError):
        service.register_video("/non/existent/video.mp4")

    # Create real test video file
    vid_file = tmp_path / "test.mp4"
    create_real_test_video(vid_file, width=320, height=240, fps=30, frames=15)

    meta = service.register_video(str(vid_file))
    assert meta.filename == "test.mp4"
    assert meta.duration_sec == 0.5  # 15 frames / 30 fps = 0.5s
    assert meta.fps == 30.0
    assert meta.width == 320
    assert meta.height == 240
    assert meta.frame_count == 15
    assert len(meta.video_fingerprint) == 64  # SHA-256


def test_video_intelligence_service_execute_inference(tmp_path):
    """Verify complete video inference pipeline execution and temporal analytics."""
    service = VideoIntelligenceService(storage_dir=tmp_path)

    # Create real test video
    vid_file = tmp_path / "test_run.mp4"
    create_real_test_video(vid_file, width=320, height=240, fps=30, frames=10)
    meta = service.register_video(str(vid_file))

    # Feed deterministic detections to test ByteTrack association
    feed = [
        [
            {
                "class_name": "person",
                "confidence": 0.90,
                "bbox": [50.0 + i * 5, 50.0, 100.0 + i * 5, 150.0],
            }
        ]
        for i in range(5)
    ]

    run = service.run_video_tracking(
        video_id=meta.video_id,
        model_id="yolo11s.pt",
        sampling_config=FrameSamplingConfig(sample_interval=2),
        synthetic_frames_data=feed,
    )

    assert run.run_id.startswith("vrun_")
    assert run.video_id == meta.video_id
    assert run.tracker_name == "ByteTrack"
    assert run.processed_frames == 5
    assert run.total_tracks == 1
    assert len(run.tracks) == 1
    assert run.tracks[0].class_name == "person"
    assert run.analytics.total_tracks == 1
    assert run.processing_fps > 0.0


def test_video_api_endpoints(tmp_path):
    """Test video intelligence REST API routes with real video upload."""
    vid_file = tmp_path / "api_test.mp4"
    create_real_test_video(vid_file, width=320, height=240, fps=30, frames=10)

    # 1. Upload Video
    with open(vid_file, "rb") as f:
        res_upload = client.post(
            "/api/v1/video/upload",
            files={"file": ("api_test.mp4", f, "video/mp4")},
        )
    assert res_upload.status_code == 201
    meta = res_upload.json()
    video_id = meta["video_id"]
    assert meta["frame_count"] == 10

    # 2. Create Video Run
    res_create = client.post(
        "/api/v1/video/runs",
        json={
            "video_id": video_id,
            "model_id": "yolo11s.pt",
            "sampling_mode": "EVERY_2ND_FRAME",
            "custom_stride": 2,
        },
    )
    assert res_create.status_code == 201
    run_data = res_create.json()
    run_id = run_data["run_id"]
    assert run_data["tracker_name"] == "ByteTrack"

    # 3. List Runs
    res_list = client.get("/api/v1/video/runs")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1

    # 4. Get Run Detail
    res_get = client.get(f"/api/v1/video/runs/{run_id}")
    assert res_get.status_code == 200
    assert res_get.json()["run_id"] == run_id

    # 5. Get Stream
    res_stream = client.get(f"/api/v1/video/stream/{video_id}")
    assert res_stream.status_code == 200

    # 6. Export Trajectories CSV
    res_export = client.get(f"/api/v1/video/runs/{run_id}/export")
    assert res_export.status_code == 200
    assert "data" in res_export.json()
    assert "run_id,video_id,track_id,class_name,confidence" in res_export.json()["data"]
