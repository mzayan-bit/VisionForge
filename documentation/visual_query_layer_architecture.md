# VisionForge Visual Query Layer Architecture

## Executive Summary
VisionForge **Visual Query Layer** enables users to ask structured questions about existing computer vision facts (detections, tracks, trajectories, regions, temporal events) using natural language.

$$\text{User Question} \rightarrow \text{Interpretation} \rightarrow \text{Structured Query DSL} \rightarrow \text{Validation} \rightarrow \text{Execution} \rightarrow \text{QueryResult} \rightarrow \text{Visual Evidence}$$

---

## 1. Core Principles & Security Protections

> [!IMPORTANT]
> **Read-Only Security Guarantee & No Hallucinated Numerical Facts**
>
> 1. **100% Read-Only**: The Visual Query Layer is strictly read-only. It cannot execute arbitrary code, generate executable SQL, modify datasets, launch training jobs, or perform file operations.
> 2. **No Numerical Hallucinations**: Answers and evidence links are derived strictly from pre-computed detections, tracks, trajectories, regions, and temporal events.

---

## 2. Supported Query Types

VisionForge supports 7 deterministic query types:

| Query Type | Description & Purpose | Natural Language Example |
| :--- | :--- | :--- |
| **`EVENT_SEARCH`** | Filter temporal events by type, region, track, or class | *"Which objects entered Loading Zone A?"* |
| **`TRACK_SEARCH`** | Search track trajectories by class, duration, or confidence | *"Which person tracks stayed longer than 5 seconds?"* |
| **`OBJECT_COUNT`** | Count active objects visible at a specific timestamp | *"How many people were present at 10 seconds?"* |
| **`TRACK_AGGREGATION`** | Compute max/avg metrics over track trajectories | *"Which track stayed longest in Zone B?"* |
| **`EVENT_AGGREGATION`** | Compute aggregations over temporal events | *"Which region had the most events?"* |
| **`TIME_RANGE_SEARCH`** | Search events and tracks within a time window | *"What happened between 5 and 15 seconds?"* |
| **`REGION_SEARCH`** | Search events and tracks intersecting a region ROI | *"Show events in Zone A."* |

---

## 3. Structured Query DSL (`VisualQuery`)

The internal query representation maps natural language text into a deterministic `VisualQuery` schema:

```json
{
  "query_id": "vq_84f91a2c",
  "run_id": "vrun_traffic_01",
  "query_type": "EVENT_SEARCH",
  "event_type": "OBJECT_ENTERED_REGION",
  "object_class": "person",
  "region_name": "Loading Zone A",
  "min_duration_sec": 3.0,
  "sort_by": "timestamp",
  "sort_order": "ASC",
  "limit": 50
}
```

---

## 4. Query Validation & Execution

1. **`QueryValidator`**:
   - Ensures `run_id` and `video_id` exist.
   - Validates `region_name` against defined Region ROIs.
   - Enforces non-negative timestamps and duration thresholds.
2. **`QueryExecutor`**:
   - Executes queries directly against stored `VideoInferenceRun` tracks and `TemporalEvent` records without calling neural networks.
   - Formulates evidence-backed summaries without altering numerical values.

---

## 5. Visual Evidence & Query Reproducibility

- **Visual Evidence (`QueryEvidenceItem`)**: Connects every query result record directly to video player timestamps (`seek=5.2s`) and track bounding boxes.
- **Reproducibility Hash**: SHA256 deterministic hash computed over `(run_id, query_type, records)` ensuring identical execution outputs across research iterations.

---

## 6. Ambiguity & Unsupported Query Handling

- **Ambiguous Queries**: Returned with status `AMBIGUOUS` (e.g., *"Show objects in the zone"* when multiple region ROIs exist).
- **Unsupported Queries**: Questions requiring unavailable models (e.g. *"What was the person doing?"*, *"Is the person happy?"*) return status `UNSUPPORTED` with explanations of what IS queryable.

---
*VisionForge Visual Query Layer Architecture Documentation.*
