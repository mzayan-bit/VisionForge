# VisionForge Unified Computer Vision Research Workspace Architecture

## 1. Objective & Design Philosophy
VisionForge is a unified **Computer Vision Research Workspace** providing a coherent, data-first, professional engineering environment.

### Design Principles:
1. **Unified Information Hierarchy**: Clear division across primary domains:
   - **Overview**: Central executive telemetry and quick research launchers.
   - **Vision & Video**: Real-time image inference (Vision Lab), multi-object tracking and temporal intelligence (Video Lab), and visual vector search.
   - **Data & Active Learning**: Dataset intelligence, class distributions, version diffs, and sample review.
   - **Models & Evaluation**: Checkpoint registry, training lab, PR curves, and failure analysis.
   - **Research Lab**: Controlled multi-seed experiments, component ablations, end-to-end research workflows, and Ask VisionForge multimodal queries.
   - **System**: Operational diagnostics, telemetry metrics, and architecture documentation.
2. **Data-Grounded Accuracy**: No mock progress, fake metrics, or artificial conclusions. Every number traces back to live API records.
3. **Shared Design Primitives**: Standardized `StatusBadge`, `PageHeader`, `EmptyState`, `LoadingState`, `ErrorState`, `CommandPalette`, and `Button` components.

---

## 2. Workspace Navigation Matrix

| Workspace Area | Route | Core Capabilities |
|---|---|---|
| **Overview** | `/` | Executive metric counters, quick actions, recent studies, subsystem health |
| **Vision Lab** | `/vision-lab` | Multi-model image inference, confidence overlays, attribution links |
| **Video Lab** | `/video-lab` | Multi-camera tracking, trajectory heatmaps, temporal event search |
| **Visual Search** | `/search` | Text, image, and crop similarity search via embedding vectors |
| **Datasets** | `/datasets` | Dataset health, quality audits, version diffs, manifest fingerprints |
| **Active Learning** | `/active-learning` | Entropy-diversity sample selection, review queues, label efficiency |
| **Models** | `/models` | Model registry, task metadata, supported devices, disk usage |
| **Training Lab** | `/training` | Training run telemetry, epoch loss curves, checkpoint generation |
| **Evaluation** | `/evaluation` | mAP@50:95, per-class PR curves, confusion matrices, failure gallery |
| **Explainability** | `/explainability` | Grad-CAM, Layer-CAM, and object concentration heatmaps |
| **Experiments** | `/experiments` | Multi-seed runs, Bessel standard deviations, ablation matrices, grounded reports |
| **Workflows** | `/workflow` | 8-stage lifecycle, dataset lock, human decision gates, lineage DAGs |
| **Ask VisionForge** | `/ask` | Multimodal vision-language query execution grounded in active vision data |
| **Diagnostics** | `/settings` | Real-time request rates, P95 latencies, error rates, failure logs |

---

## 3. Global Command Palette & Shortcuts
- **`⌘K` / `Ctrl+K`**: Opens the global [CommandPalette](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/CommandPalette.tsx) to instantly search and navigate across all workspace sections.
- **Contextual Ask VisionForge**: Directly linked from image predictions, video events, dataset anomalies, and research evaluations.
