# VisionForge Unified Visual Search Architecture

## Executive Summary
VisionForge **Unified Visual Search System** enables multi-modal visual similarity search across images, video frames, track object crops, video moments/events, and dataset samples using the pre-computed dense embedding vectors (SigLIP-base-patch16-224) and Visual Memory store.

$$\text{Query Input (Image / Frame / Crop / Event / Sample)} \rightarrow \text{Dense Embedding} \rightarrow \text{Vector Matrix Match} \rightarrow \text{Metadata Filtering} \rightarrow \text{Traceable Provenance}$$

---

## 1. Unified Visual Asset Abstraction

VisionForge categorizes all searchable visual entities under a unified `VisualAsset` schema:

| Asset Type | Description | Source Traceability |
| :--- | :--- | :--- |
| **`IMAGE`** | Standalone uploaded or memory image | `Visual Memory Record` |
| **`FRAME`** | Sampled video keyframe from video stream | `Video Asset + Timestamp (s) + Frame #` |
| **`OBJECT_CROP`** | Object appearance bounding box crop from track | `Video + Track ID + Bounding Box [x1, y1, x2, y2]` |
| **`EVENT_FRAME`** | Keyframe evidence snapshot from temporal event | `Video + Event ID + Evidence Timestamp` |
| **`DATASET_SAMPLE`** | Image sample from training/eval dataset | `Dataset ID + Sample ID + Ground Truth Class` |

---

## 2. Search Pipelines & Modalities

1. **Image $\rightarrow$ Similar Visual Assets**: Upload an image, extract 768D SigLIP embedding, search top-K matches in Visual Memory.
2. **Video Frame $\rightarrow$ Similar Frames / Moments**: Select video timestamp $t = \tau$, search similar video frames across all processed video runs.
3. **Object Crop $\rightarrow$ Similar Object Appearances**: Select detected track (e.g. `Track #17 (person)`), search visually similar object crops across datasets and video tracks.
4. **Event Moment $\rightarrow$ Similar Moments**: Select temporal event (e.g. `OBJECT_DWELLED in Loading Zone A`), search similar keyframe evidence snapshots across videos.
5. **Dataset Sample $\rightarrow$ Similar Samples**: Search for visually similar samples in dataset clusters.

---

## 3. Near-Duplicate Candidate Discovery

> [!TIP]
> **Controlled Near-Duplicate Candidate Detection**
>
> Computes full pairwise cosine similarity matrix $S = \tilde{M} \cdot \tilde{M}^T$ across indexed visual assets and flags pairs with similarity score $\ge 95\%$.
> - Helps detect redundant video frames, duplicated dataset samples, and class balance biases.

---

## 4. Embedding Model Space Compatibility

> [!IMPORTANT]
> **Vector Space Isolation Guarantee**
>
> The system records `embedding_model` and `embedding_version` for every query vector and indexed candidate. Searches verify compatibility to prevent corrupt comparisons across mismatched vector spaces.

---

## 5. Cross-System Integrations

- **Video Lab**: Every `FRAME`, `OBJECT_CROP`, and `EVENT_FRAME` search result includes a direct `[ Open Source / Video Lab ]` link seeking the video player to the exact timestamp (`/video-lab?video=sample_traffic_01&seek=14.2&track=17`).
- **Embedding Explorer**: Every result includes `[ View in Embedding Explorer ]` navigating to `/explorer` with point highlighting.
- **Active Learning & Failure Analysis**: Researchers can select any failure case or active learning candidate and launch `[ Find Similar ]` to diagnose recurring visual patterns.

---

## 6. Architectural Distinction: Visual Similarity vs Re-Identification

> [!NOTE]
> **Visual Similarity Guarantee**
>
> Visual search returns items with high representation proximity in dense embedding space. It does not claim biometric facial identification, person re-identification, or physical identity tracking.

---
*VisionForge Unified Visual Search Architecture Documentation.*
