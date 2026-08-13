# VisionForge Temporal Event Intelligence Architecture

## Executive Summary
VisionForge **Temporal Event Intelligence** transforms low-level object tracking trajectories into higher-level, explainable, observable temporal events.

$$\text{Tracks \& Trajectories} \rightarrow \text{Region ROI Intersections} \rightarrow \text{Rule-Based Detection} \rightarrow \text{Debouncing \& Merging} \rightarrow \text{Event Timeline Stream} \rightarrow \text{Evidence Verification}$$

```
00:02.1  Track #4 started
00:04.7  Track #4 entered Loading Zone A
00:07.9  Track #4 stopped
00:08.5  Track #4 dwelled for 6.5s in Loading Zone A
00:12.2  Track #7 entered Loading Zone A
00:13.0  Track #4 and Track #7 became close (45.2 px apart)
00:15.3  Track #4 left Loading Zone A
```

---

## 1. Core Principles & Zero-Inference Architecture

> [!IMPORTANT]
> **Observable Visual Evidence Guarantee**
>
> 1. **Rule-Derived**: Events are derived strictly from trajectory geometry, frame timestamps, and spatial bounds. No Large Language Models (LLMs) or hallucinated probabilistic claims are used.
> 2. **Zero Inference Overhead**: Event detection operates directly on pre-computed `VideoInferenceRun` tracks without re-running object detection models.

---

## 2. Event Types & Derivation Rules

VisionForge supports 10 observable event types:

| Event Type | Derivation Rule & Condition | Parameters Recorded |
| :--- | :--- | :--- |
| **`TRACK_STARTED`** | Track status transitions to active ($t = t_{\text{first}}$) | `track_id`, `class_name`, `initial_position` |
| **`TRACK_ENDED`** | Track status terminates ($t = t_{\text{last}}$) | `track_id`, `class_name`, `total_visibility_sec` |
| **`OBJECT_ENTERED_REGION`** | Trajectory point enters defined Region ROI | `track_id`, `region_id`, `entry_position` |
| **`OBJECT_LEFT_REGION`** | Trajectory point exits defined Region ROI | `track_id`, `region_id`, `exit_position`, `total_dwell_sec` |
| **`OBJECT_DWELLED`** | Object remains inside ROI for $\ge t_{\text{dwell}}$ (default 3.0s) | `track_id`, `region_id`, `dwell_duration_sec` |
| **`OBJECT_STOPPED`** | Image-plane speed $< v_{\text{stop}}$ (default 15 px/s) | `track_id`, `avg_speed_px_s`, `duration_sec` |
| **`OBJECT_MOVED`** | Image-plane speed $\ge v_{\text{stop}}$ (default 15 px/s) | `track_id`, `avg_speed_px_s`, `duration_sec` |
| **`OBJECT_COUNT_CHANGED`** | Active object count per second increases/decreases | `previous_count`, `new_count`, `change_delta` |
| **`OBJECTS_BECAME_CLOSE`** | Inter-track center distance $\le d_{\text{prox}}$ (default 100 px) | `track_a`, `track_b`, `distance_px` |
| **`OBJECTS_MOVED_APART`** | Inter-track center distance $\ge d_{\text{sep}}$ (default 180 px) | `track_a`, `track_b`, `distance_px` |

---

## 3. Debouncing & Interval Merging

> [!TIP]
> **Hysteresis Debouncing & Deduplication**
>
> - **Debounce Window**: Uses a 3-frame hysteresis window to eliminate noisy single-frame region boundary flickers.
> - **State Merging**: Consecutive frame states (`STOPPED` or `MOVING`) are merged into a single continuous time interval event rather than emitting hundreds of per-frame records.

---

## 4. Visual Evidence Verification

Each detected temporal event generates an `EventEvidence` record:
- **Frame Before**: $f_{\text{start}} - 2$ (Pre-onset frame).
- **Event Frame**: $f_{\text{start}}$ (Key event occurrence frame).
- **Frame After**: $f_{\text{end}} + 2$ (Post-event frame).
- Allows visual verification of every event on the frontend via `[ Inspect Evidence ]`.

---

## 5. Architectural Distinction: Visual Events vs Action Recognition

> [!NOTE]
> **Rule-Derived Visual Events vs Action Recognition**
>
> VisionForge temporal events describe **observable visual trajectory events** in image-space. They do not claim deep human action semantics (e.g. "person is stealing an item") unless a specialized action-recognition model is integrated.

---
*VisionForge Temporal Event Intelligence Architecture Documentation.*
