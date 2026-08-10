"""Unit and Integration Tests for VisionForge Inference Studio Engine and API."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from visionforge.inference.schemas import (
    InferenceBenchmarkConfig,
    InferenceConfig,
    ModelLifecycleState,
    NormalizedBoundingBox,
    StandardPrediction,
)
from visionforge.inference.service import (
    ImageValidationError,
    InferenceService,
    ModelLifecycleManager,
)
from visionforge.main import app

client = TestClient(app)


# ─── Schema Tests ──────────────────────────────────────────────────

def test_inference_config_validation():
    """Verify InferenceConfig validation boundaries."""
    cfg = InferenceConfig(model_id="yolo11s.pt", confidence_threshold=0.3, iou_threshold=0.5)
    assert cfg.model_id == "yolo11s.pt"
    assert cfg.confidence_threshold == 0.3
    assert cfg.iou_threshold == 0.5
    assert cfg.imgsz == 640
    assert cfg.device == "auto"


def test_standard_prediction_schema():
    """Verify StandardPrediction and NormalizedBoundingBox schemas."""
    bbox = NormalizedBoundingBox(
        x_center=0.5,
        y_center=0.5,
        width=0.4,
        height=0.4,
        pixel_coords=[100.0, 100.0, 300.0, 300.0],
    )
    pred = StandardPrediction(
        prediction_id="pred_1",
        class_id=0,
        class_name="helmet",
        confidence=0.95,
        bbox=bbox,
        model_id="yolo11s.pt",
        model_version="1.0.0",
    )
    assert pred.class_name == "helmet"
    assert pred.confidence == 0.95
    assert pred.bbox.pixel_coords == [100.0, 100.0, 300.0, 300.0]


# ─── Lifecycle Manager Tests ─────────────────────────────────────────

def test_model_lifecycle_manager():
    """Verify model warming lifecycle transitions."""
    mgr = ModelLifecycleManager()
    assert mgr.get_state("yolo11s.pt") == ModelLifecycleState.NOT_LOADED

    # Mock ultralytics loading
    with patch("ultralytics.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance

        model = mgr.load_model("yolo11s.pt", "yolo11s.pt")
        assert model == mock_instance
        assert mgr.get_state("yolo11s.pt") == ModelLifecycleState.READY

        # Unload
        unloaded = mgr.unload_model("yolo11s.pt")
        assert unloaded is True
        assert mgr.get_state("yolo11s.pt") == ModelLifecycleState.NOT_LOADED


# ─── Service Execution Tests ─────────────────────────────────────────

def test_inference_service_run_and_history(tmp_path):
    """Test single image inference execution, overlay generation, and history persistence."""
    # Create sample image file
    sample_img = tmp_path / "test_sample.jpg"
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    img.save(sample_img, "JPEG")

    service = InferenceService()
    cfg = InferenceConfig(model_id="yolo11s.pt", confidence_threshold=0.2)

    result = service.run_inference(str(sample_img), cfg)

    assert result.inference_id.startswith("inf_")
    assert result.model_id == "yolo11s.pt"
    assert isinstance(result.predictions, list)
    assert result.summary.image_width == 640
    assert result.summary.image_height == 480
    assert result.visual_overlay_path is not None
    assert Path(result.visual_overlay_path).is_file()

    # Verify history persistence
    history = service.list_history()
    assert len(history) > 0
    rec = service.get_inference_record(result.inference_id)
    assert rec is not None
    assert rec.inference_id == result.inference_id


def test_inference_service_model_comparison(tmp_path):
    """Test side-by-side model comparison execution."""
    sample_img = tmp_path / "cmp_sample.jpg"
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(200, 200, 200))
    img.save(sample_img, "JPEG")

    service = InferenceService()
    cmp_res = service.run_comparison(
        image_path=str(sample_img),
        model_a_id="yolo11s.pt",
        model_b_id="rtdetr-l.pt",
    )

    assert cmp_res.comparison_id.startswith("cmp_")
    assert cmp_res.model_a_result.model_id == "yolo11s.pt"
    assert cmp_res.model_b_result.model_id == "rtdetr-l.pt"
    assert "Qualitative Comparison" in cmp_res.notes


def test_inference_service_latency_benchmark():
    """Test multi-pass inference latency benchmarking calculations."""
    service = InferenceService()
    bm_cfg = InferenceBenchmarkConfig(model_id="yolo11s.pt", runs=5, warmup_runs=2, imgsz=640)

    bm_res = service.run_benchmark(bm_cfg)

    assert bm_res.benchmark_id.startswith("bm_")
    assert bm_res.model_id == "yolo11s.pt"
    assert bm_res.runs == 5
    assert bm_res.average_latency_ms > 0
    assert bm_res.fps > 0
    assert bm_res.p95_latency_ms >= bm_res.median_latency_ms


def test_inference_service_invalid_image():
    """Verify error handling on non-existent image path."""
    service = InferenceService()
    cfg = InferenceConfig(model_id="yolo11s.pt")
    with pytest.raises(ImageValidationError):
        service.run_inference("/invalid/path/nonexistent.jpg", cfg)


# ─── API Router Endpoints Tests ─────────────────────────────────────

def test_api_list_inference_models():
    """Verify GET /api/v1/inference/models endpoint."""
    response = client.get("/api/v1/inference/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    model_ids = [m["model_id"] for m in data]
    assert "yolo11s.pt" in model_ids
    assert "rtdetr-l.pt" in model_ids


def test_api_run_inference_endpoint(tmp_path):
    """Verify POST /api/v1/inference/run endpoint."""
    sample_img = tmp_path / "api_sample.jpg"
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(150, 150, 150))
    img.save(sample_img, "JPEG")

    response = client.post(
        "/api/v1/inference/run",
        data={
            "image_path": str(sample_img),
            "model_id": "yolo11s.pt",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "imgsz": 640,
            "device": "auto",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "inference_id" in data
    assert data["model_id"] == "yolo11s.pt"
    assert isinstance(data["predictions"], list)


def test_api_inference_history_endpoint():
    """Verify GET /api/v1/inference/history endpoint."""
    response = client.get("/api/v1/inference/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_model_comparison_endpoint(tmp_path):
    """Verify POST /api/v1/inference/compare endpoint."""
    sample_img = tmp_path / "api_cmp_sample.jpg"
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(50, 50, 50))
    img.save(sample_img, "JPEG")

    response = client.post(
        "/api/v1/inference/compare",
        json={
            "image_path": str(sample_img),
            "model_a_id": "yolo11s.pt",
            "model_b_id": "rtdetr-l.pt",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "comparison_id" in data
    assert data["model_a_result"]["model_id"] == "yolo11s.pt"
    assert data["model_b_result"]["model_id"] == "rtdetr-l.pt"


def test_api_inference_benchmark_endpoint():
    """Verify POST /api/v1/inference/benchmark endpoint."""
    response = client.post(
        "/api/v1/inference/benchmark",
        json={
            "model_id": "yolo11s.pt",
            "runs": 3,
            "warmup_runs": 1,
            "batch_size": 1,
            "imgsz": 640,
            "device": "auto",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "benchmark_id" in data
    assert data["runs"] == 3
    assert data["fps"] > 0


def test_api_unload_model_endpoint():
    """Verify POST /api/v1/inference/unload/{model_id} endpoint."""
    response = client.post("/api/v1/inference/unload/yolo11s.pt")
    assert response.status_code == 200
    data = response.json()
    assert "unloaded" in data
