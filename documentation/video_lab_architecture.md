# VisionForge Video Lab & Multi-Object Tracking Architecture

## Executive Summary
VisionForge extends static computer vision inference into **Video Intelligence** with multi-object tracking, persistent Track IDs, trajectory analysis, temporal analytics, and interactive visualization.

```
VIDEO
  ↓
FRAME SAMPLING (Stride 1, 2, 5)
  ↓
OBJECT DETECTION (Vision Lab Model Inference)
  ↓
MULTI-OBJECT TRACKING (ByteTrack IoU + Kalman)
  ↓
PERSISTENT TRACK IDENTITIES (Track #4, Track #7)
  ↓
SPATIAL TRAJECTORIES (Pixel Center History & Norm Coordinates)
  ↓
TEMPORAL ANALYTICS & PIXEL-SPEED TELEMETRY
  ↓
INTERACTIVE VIDEO LAB VISUALIZATION
```

---

## 1. Frame Sampling & Compute Safety

> [!TIP]
> **Configurable Frame Sampling**
>
> Processing every single video frame can be computationally expensive on local hardware. VisionForge provides configurable frame sampling modes:
> - `EVERY_FRAME` (Stride 1): Maximum temporal resolution for high-speed object motion.
> - `EVERY_2ND_FRAME` (Stride 2): Recommended default balancing temporal accuracy and compute speed.
> - `EVERY_5TH_FRAME` (Stride 5): Optimized for fast processing of long videos on MacBook M4 Air or free-tier GPUs.

---

## 2. ByteTrack Multi-Object Tracker

VisionForge integrates the **ByteTrack** multi-object tracking algorithm:
1. **Detection Association**: Computes Intersection-over-Union (IoU) matrices between existing active tracks and new frame detections.
2. **Persistent Track IDs**: Objects maintain the same integer Track ID (e.g. `Track #4`) across consecutive frames.
3. **Occlusion & Lost Frames**: Tracks tolerate brief occlusions up to `max_lost_frames = 30` before state termination.

---

## 3. Trajectory & Pixel Speed Formulation

For each tracked object $k$, spatial position is recorded at frame $i$:

$$P_i = (x_{\text{center}}, y_{\text{center}})$$

Total cumulative distance in pixel-space:

$$D_{\text{total}} = \sum_{i=1}^{N-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$

Average pixel speed:

$$V_{\text{pixel}} = \frac{D_{\text{total}}}{\Delta t} \quad (\text{pixels / second})$$

> [!IMPORTANT]
> **Image-Plane Pixel Speed vs Physical Speed**
>
> Speeds are calculated strictly as **pixel speed** ($\text{px/s}$) on the image plane. VisionForge does not claim real-world physical speed ($\text{m/s}$ or $\text{km/h}$) unless camera calibration and homography parameters are provided.

---

## 4. Hardware Safety & Resource Management

- **Memory Bounding**: Videos are processed in chunked streams without loading full video bytes into RAM.
- **Resource Cleanup**: Temporary frame extractions are automatically purged after pipeline completion.
- **Device Support**: Runs seamlessly on MacBook M4 Air (CPU/MPS) and Google Colab GPUs.

---

## 5. Architectural Distinction: Tracking vs Re-Identification

> [!NOTE]
> **Tracking vs Re-Identification**
>
> Multi-object tracking maintains object identity **within a single video tracking run**. If an object leaves the camera frame completely and re-appears much later, ByteTrack will assign a new Track ID unless a deep visual Person/Vehicle Re-Identification (ReID) model is explicitly integrated.

---
*VisionForge Video Intelligence Architecture Documentation.*
