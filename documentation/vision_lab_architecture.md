# VisionForge Vision Lab Architecture & Interactive Inference Engine

## Executive Overview
The **Vision Lab** (Interactive Inference Studio) is VisionForge's flagship interactive computer vision environment. It provides real-time, reproducible, and standardized inference execution against trained computer vision checkpoints (YOLO11, RT-DETR, or registered custom checkpoints).

---

## 1. Model Lifecycle & Memory Warming Architecture

To optimize performance and memory footprint (especially on Apple Silicon M4 / GPU instances), model checkpoints are managed via `ModelLifecycleManager` with explicit lifecycle states:

```
[ NOT_LOADED ] ---> [ LOADING ] ---> [ READY ] ---> [ UNLOADING ] ---> [ NOT_LOADED ]
                          |
                          v
                     [ FAILED ]
```

### Lifecycle States:
- **`NOT_LOADED`**: Model weights reside solely on disk. Memory footprint = 0 MB.
- **`LOADING`**: Weights are actively instantiated into PyTorch / Ultralytics runtime.
- **`READY`**: Model weights are warmed in RAM/VRAM, ready for zero-latency forward passes.
- **`FAILED`**: Model loading encountered corrupt checkpoint weights or hardware incompatibility; clear error diagnostics returned.
- **`UNLOADING`**: Weights purged from RAM/VRAM memory cache.

---

## 2. Standardized Prediction Schema

VisionForge decouples model framework specifics (Ultralytics, TorchScript, ONNX) from downstream UI consumption by enforcing a framework-independent prediction representation:

```json
{
  "prediction_id": "pred_inf_8a2d1f99_0",
  "class_id": 0,
  "class_name": "helmet",
  "confidence": 0.942,
  "bbox": {
    "x_center": 0.5,
    "y_center": 0.4,
    "width": 0.3,
    "height": 0.3,
    "pixel_coords": [224.0, 144.0, 416.0, 336.0]
  },
  "model_id": "yolo11s.pt",
  "model_version": "1.0.0"
}
```

### Coordinate Standardization:
- **Normalized Bounding Box**: `[x_center, y_center, width, height]` scaled strictly to `[0.0, 1.0]`.
- **Pixel Coordinates**: `[x1, y1, x2, y2]` in original image dimension pixel space for UI rendering.

---

## 3. Preprocessing & Postprocessing Pipeline

1. **Preprocessing**:
   - Image uploaded or fetched from Visual Memory.
   - Preserves original image untouched.
   - RGB color-space conversion & resolution scaling (`imgsz`=320, 640, or 1024 px).
2. **Forward Pass**:
   - High-precision latency measurement using `time.perf_counter()`.
3. **Postprocessing**:
   - Confidence score filtering against user threshold (`conf_threshold`=0.01-1.0).
   - Non-Maximum Suppression (NMS) applied natively at model level (`iou_threshold`=0.01-1.0). NMS is never applied twice.

---

## 4. Visual Overlay Generation

Visual prediction overlays are produced as derived image artifacts:
- Original uploaded image remains 100% pristine.
- Overlay artifact saved to `inference/overlays/{inf_id}_overlay.jpg` using PIL ImageDraw with vibrant, class-specific color coding.
- SVG overlays rendered dynamically in the frontend canvas for interactive click-to-inspect capabilities.

---

## 5. Latency Measurement & Benchmarking Protocol

Latency is measured using high-resolution performance counters:

$$\text{FPS} = \frac{1000}{\text{Average Latency (ms)}}$$

### Multi-Pass Latency Benchmarking:
- Executes $N$ warm-up iterations to eliminate cold-start OS disk paging artifacts.
- Measures $M$ timed inference runs.
- Computes **Mean**, **Median (p50)**, **p95 Latency**, **Min/Max**, and **Throughput (FPS)**.

---

## 6. Architectural Note: Video Frame Inference Preparation

> [!NOTE]
> **Future Real-Time Video Architecture Preparation**
>
> Although real-time video stream decoding (RTSP / WebRTC / MP4) is out of scope for the current image inference phase, `InferenceService` has been engineered to seamlessly support video frame streams:
>
> 1. **Framework-Independent Payload**: Video frames extracted by future OpenCV / GStreamer pipelines will pass directly to `InferenceService.run_inference()` as raw RGB memory frames.
> 2. **Contract Preservation**: Predictions for each frame will reuse the exact `StandardPrediction` and `NormalizedBoundingBox` format, enabling zero-change reuse of visualization overlays.
> 3. **Batching**: Multi-frame video batches will leverage `InferenceBenchmarkConfig.batch_size` for GPU tensor parallelism.

---
*VisionForge Computer Vision Architecture Documentation.*
