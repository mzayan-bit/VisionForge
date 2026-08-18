# VisionForge 5–10 Minute Interactive Showcase & Demo Guide

This guide walks a technical evaluator, recruiter, or collaborator through a complete live walkthrough of VisionForge in under 10 minutes.

---

## Prerequisites (1 minute)

```bash
# 1. Start the workbench (Docker or Local)
make dev
# or: docker compose up -d

# 2. Seed the real COCO8 dataset into the platform
make seed
```

Open your browser to: [http://localhost:3000](http://localhost:3000)

---

## Step 1: Research Workspace Overview (1 minute)
- **URL**: [http://localhost:3000](http://localhost:3000)
- **What to Observe**:
  - Top metric counters (Datasets, Models, Experiments, Workflows).
  - Live system health telemetry indicator.
  - Quick action links into Dataset Lab, Model Registry, and Research Workflows.

---

## Step 2: Dataset Intelligence & Quality Scorecard (1.5 minutes)
- **URL**: [http://localhost:3000/datasets](http://localhost:3000/datasets)
- **Action**: Click on the ingested **COCO8** dataset (`prep_coco8_v1_0_0`).
- **What to Observe**:
  - Class distribution histogram across the 12 categories (person, dog, car, skateboard, etc.).
  - Bounding box geometry scatter (normalized $x, y, w, h$).
  - Data quality scorecard: bounding box boundary compliance, image aspect ratio variance, and train/val/test split ratio validation.

---

## Step 3: Interactive Inference & Vision Lab (1.5 minutes)
- **URL**: [http://localhost:3000/vision-lab](http://localhost:3000/vision-lab)
- **Action**: Select the registered model `visionforge-yolo11n-coco8:1.0.0` and run an inference on a sample image.
- **What to Observe**:
  - Multi-class bounding box overlay canvas.
  - Confidence and IoU threshold sliders with instant interactive re-rendering.
  - Sub-millisecond execution latency readout (e.g. 7.8 ms).

---

## Step 4: Model Evaluation & Diagnostic Error Analysis (1.5 minutes)
- **URL**: [http://localhost:3000/evaluation](http://localhost:3000/evaluation)
- **Action**: Open the evaluation run for `visionforge-yolo11n-coco8:1.0.0` on the test split.
- **What to Observe**:
  - Measured benchmark metrics: **mAP@50: 68.8%**, **Precision: 62.7%**, **Recall: 66.7%**.
  - Precision-Recall curves across IoU thresholds.
  - **Failure Gallery**: Isolated False Positives, False Negatives, and Localization Mismatches.
  - **One-Click Active Learning**: Curation action to promote hard failure cases to candidate datasets.

---

## Step 5: Explainability & Grad-CAM Spatial Attributions (1 minute)
- **URL**: [http://localhost:3000/explainability](http://localhost:3000/explainability)
- **What to Observe**:
  - 2D Grad-CAM attribution heatmaps overlaid on test sample detections.
  - Target bounding box concentration scores ($>75\%$ activation mass inside object boundaries).

---

## Step 6: Visual Vector Memory & Similarity Search (1 minute)
- **URL**: [http://localhost:3000/search](http://localhost:3000/search)
- **What to Observe**:
  - 768-dimensional dense SigLIP visual feature embedding index.
  - Sub-millisecond cosine similarity ranking and near-duplicate asset discovery.

---

## Step 7: Research Experiments & Lineage Graphs (1 minute)
- **URL**: [http://localhost:3000/experiments](http://localhost:3000/experiments)
- **Action**: Open `exp_coco8_transfer_v1`.
- **What to Observe**:
  - Complete DAG lineage tracking: Dataset Version $\rightarrow$ Model Checkpoint $\rightarrow$ Evaluation Run $\rightarrow$ Decision Record.
  - Cryptographic SHA-256 artifact verification and automated markdown research report export.

---

## Step 8: System Observability & Prometheus Metrics (30 seconds)
- **URL**: [http://localhost:3000/settings](http://localhost:3000/settings)
- **What to Observe**:
  - Subsystem and dependency health matrix (`API`, `Storage`, `Job Queue`, `Model Registry`, `Visual Memory`).
  - Real-time API request meters and background job observatory.
  - Direct Prometheus metrics endpoint link: [http://localhost:8000/metrics](http://localhost:8000/metrics).
