# VisionForge Video Lab — Temporal Computer Vision Analysis Architecture & Specification

## 1. Purpose & Contract
VisionForge Video Lab is a **temporal computer vision analysis subsystem** built to analyze object movements, maintain persistent spatial trajectories, evaluate region-of-interest (ROI) occupancy, and extract deterministic rule-based temporal events across time from video streams.

Unlike single-image inference in Vision Lab (*"What objects exist in this image?"*), Video Lab answers temporal questions:
> **"What is happening across time?"**
> - Which objects entered, dwelled in, or exited designated spatial regions?
> - What were the continuous motion trajectories, pixel velocities, and travel distances of tracked entities?
> - At what exact timestamp did a specific object appear, disappear, or change state?

---

## 2. Central Processing Pipeline

```
VIDEO ASSET (.mp4, .mov, .avi, .mkv)
  │
  ▼ [1. Ingestion & Validation]
Deterministic Metadata Extraction (Width, Height, FPS, Duration, SHA-256 Fingerprint)
  │
  ▼ [2. Frame Extraction]
OpenCV VideoCapture Strided Frame Sampling (Every Nth frame)
  │
  ▼ [3. Canonical Model Inference]
YOLO11 / Detection Model via InferenceService Lifecycle (Pixel Bounding Boxes, Classes, Confidence)
  │
  ▼ [4. Multi-Object Tracking]
ByteTracker (Kalman Filter State Estimation + Two-Stage Hungarian IoU Association)
  │
  ▼ [5. Trajectory & Velocity Estimation]
Continuous Center-Point Paths, Euclidean Displacement, Velocity (px/s), Dwell Durations
  │
  ▼ [6. Spatial / ROI Analysis]
Ray-Casting Polygon & Rectangle Point Containment (`is_point_in_region`)
  │
  ▼ [7. Temporal Event Generation]
Rule-Based State Transition Engine (`TRACK_STARTED`, `OBJECT_ENTERED_REGION`, `OBJECT_DWELLED`, `OBJECT_LEFT_REGION`, `TRACK_ENDED`)
  │
  ▼ [8. Queryable Evidence & Grounding]
Visual Query DSL (`EVENT_SEARCH`, `OBJECT_COUNT`, `TRACK_SEARCH`) with Timestamp Jump Navigation
  │
  ▼ [9. Verification & Export]
Native HTML5 Synchronized Playback, Trajectory Overlays, CSV & JSON Analysis Artifacts
```

---

## 3. Supported Tasks & Capabilities

1. **Deterministic Video Ingestion**:
   - Validation of container, codec, resolution, and non-empty byte stream.
   - SHA-256 content fingerprinting to prevent duplicate processing.
2. **Object Detection**:
   - Integration with canonical VisionForge `InferenceService` and `ModelManager` models (e.g. YOLO11s, YOLO11n, YOLO11m, YOLO11x).
3. **Multi-Object Tracking (ByteTrack)**:
   - High-performance tracking associating high-confidence and low-confidence detections across consecutive frames to maintain identity through partial occlusions.
4. **Spatial Region of Interest (ROI)**:
   - User-defined polygon and rectangular bounding zones with coordinate normalization (`[0, 1]` relative to frame resolution).
5. **Dwell Time Computation**:
   - Exact temporal difference: $\Delta t = t_{\text{exit}} - t_{\text{enter}}$ based on video timestamps.
6. **Temporal Event Stream**:
   - Real-time generation of explainable event records with frame index, timestamp, track IDs, and geometric trigger rules.
7. **Natural Language & Structured Visual Query**:
   - Query parser mapping questions (e.g. *"What objects entered Zone A?"*) into typed query operations executed over persisted event logs.
8. **Export & Lineage**:
   - Trajectory data export as RFC 4180 CSV and JSON analysis records with full provenance linkage.

---

## 4. Model Source of Truth & Lineage
Every video analysis run records its exact model provenance:
- **Model Identifier**: `model_id` (e.g. `yolo11s.pt`)
- **Framework**: `PyTorch / Ultralytics`
- **Inference Configuration**: Confidence Threshold (e.g. `0.45`), IoU Threshold (`0.80`), Sampling Stride (`2`)
- **Hardware Target**: PyTorch device resolution (`mps`, `cuda`, `cpu`)
- **Video Fingerprint**: SHA-256 hash of raw video bytes

---

## 5. Storage Architecture
Video assets and analysis artifacts are persisted under the configured storage root (`~/.cache/visionforge/video/`):
- `videos_metadata.json`: Registered video assets and technical telemetry.
- `video_runs.json`: Historical tracking runs, full trajectories, and per-track duration/speed metrics.
- `video_sessions.json`: Lifecycle sessions linking video asset, model checkpoint, and processing configuration.
- `regions.json`: User-defined spatial ROIs with vertex coordinates and color metadata.
- `events.json`: Chronologically sorted temporal events with grounding evidence.

---

## 6. Performance & Operational Limits
- **Maximum Recommended Video Duration**: 10 minutes (per interactive run).
- **Sampling Strides**: Configurable stride $k \in [1, 60]$ (default: every 2nd frame).
- **Inference Latency**: Measured in real milliseconds per frame ($\text{ms/frame}$) and processing FPS ($\text{proc\_fps}$).
- **Streaming**: Native HTTP chunked video streaming via `GET /api/v1/video/stream/{video_id}`.

---

## 7. Limitations
- **3D World Geometry**: Coordinates and velocities are measured in 2D image-space pixels ($\text{px/s}$). Real-world metric velocity requires camera calibration / homography estimation.
- **Occlusion Persistence**: ByteTrack maintains tracks across short occlusions (up to `track_buffer=30` frames), but extreme occlusions or camera cuts terminate the track ID honestly.
- **Semantic Understanding**: Events are derived deterministically from bounding box and geometric ROI state transitions.
