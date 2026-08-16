# Advanced Video Understanding & Temporal Intelligence Architecture

## 1. Overview & Research Motivation

Computer vision systems often collapse temporal video data into isolated per-frame bounding box predictions. In real-world visual perception and surveillance workflows, understanding requires modeling the continuous evolution of visual entities over time:

$$\text{Frames} \longrightarrow \text{Detections} \longrightarrow \text{Tracks} \longrightarrow \text{Trajectories} \longrightarrow \text{Observable Events} \longrightarrow \text{Temporal Evidence}$$

VisionForge's Advanced Video Understanding & Temporal Intelligence Layer introduces deterministic, explainable multi-object tracking, spatial region of interest (ROI) monitoring, and natural language temporal query grounding without hallucinated physics (e.g. fake km/h) or unwarranted semantic assumptions.

---

## 2. Temporal Representation Hierarchy

### A. Frame Observations & Detections
Each frame $f_t$ at timestamp $t$ produces raw bounding box detections $d_i = [x_{\min}, y_{\min}, x_{\max}, y_{\max}, \text{conf}, \text{cls}]$.

### B. Persistent Multi-Object Tracking (ByteTrack)
The system associates detections across time steps using two-stage IoU matching:
- High-confidence detections ($\ge 0.45$) matched with active tracks.
- Low-confidence detections matched with unmatched tracks to handle visual occlusion and motion blur.

### C. Continuous Spatial Trajectories & Motion Metrics
For each active Track $T_k$, trajectory points are recorded at each sampled frame:
- **Instantaneous Velocity**: Image-space pixel displacement over time $\vec{v}(t) = \frac{\Delta \vec{p}}{\Delta t} \text{ (px/s)}$.
- **Cumulative Trajectory Distance**: Total pixel distance traversed across all frames.
- **Debounced Region State**: Track centroid containment inside defined polygon or rectangle ROIs.

---

## 3. Deterministic Temporal Event Detection

Rule-based event detectors evaluate spatial-temporal criteria to emit verifiable event instances:

| Event Type | Trigger Criteria | Reliability |
| :--- | :--- | :--- |
| `TRACK_STARTED` / `OBJECT_APPEARED` | First valid observation of a persistent Track ID | `HIGH` |
| `TRACK_ENDED` / `OBJECT_DISAPPEARED` | Final observation before track termination | `HIGH` |
| `OBJECT_ENTERED_REGION` | Centroid transitions from outside to inside ROI | `HIGH` |
| `OBJECT_LEFT_REGION` | Centroid transitions from inside to outside ROI | `HIGH` |
| `OBJECT_DWELLED` | Continuous duration inside ROI $\ge t_{\text{dwell}}$ (e.g., 3.0s) | `HIGH` |
| `OBJECT_STOPPED` | Image-plane speed drops $< 15\text{ px/s}$ for $\ge 2.0\text{s}$ | `MEDIUM` |
| `OBJECTS_BECAME_CLOSE` | Euclidean distance between tracks $\le d_{\text{prox}}$ for $\ge 1.5\text{s}$ | `MEDIUM` |
| `OBJECT_COUNT_CHANGED` | Net change in active track count across time bins | `HIGH` |

---

## 4. Visual Query Layer & Natural Language Grounding

The Visual Query Layer parses natural language questions into structured query DSL objects without external dependencies:

```json
{
  "query_type": "EVENT_SEARCH",
  "event_type": "OBJECT_ENTERED_REGION",
  "object_class": "person",
  "region_name": "Zone A (Loading Dock)",
  "time_range": [0.0, 10.0]
}
```

Queries return evidence frames with direct jump-to-timestamp links (`/video-lab?seek=4.2&track=101`) enabling instantaneous visual verification.

---

## 5. Lineage & Provenance Tracking

Every `VideoSession` maintains complete cryptographic and operational lineage:
- Model checkpoint hash and architecture (`yolo11s.pt`).
- Tracker configuration (IoU threshold, buffer length, frame sampling mode).
- Video asset SHA-256 fingerprint.
- Full CSV trajectory export for auditability and downstream research.
