# VisionForge Active Learning & Human-in-the-Loop Architecture

## Executive Summary
VisionForge **Active Learning & Human-in-the-Loop Workflow** establishes a scientifically disciplined methodology to identify, prioritize, and curate the most informative candidate samples from vast unlabeled image pools.

$$\text{Candidate Pool} \xrightarrow{\text{Uncertainty + Diversity}} \text{Top-K Batch} \xrightarrow{\text{Focus Review Session}} \text{Approved Curation} \xrightarrow{\text{Explicit Confirmation}} \text{New Dataset Version}$$

---

## 1. Core Principles & Non-Destructive Invariant

> [!IMPORTANT]
> **Human in the Loop Guarantee**
> - Active learning is strictly a **prioritization mechanism**, not an autonomous decision maker.
> - The system will **NEVER** silently delete, relabel, retrain, promote models, or modify datasets without explicit user confirmation.
> - Uncertainty $\ne$ Incorrectness; Diversity $\ne$ Importance; Outlier $\ne$ Bad sample.

---

## 2. Selection Strategies

| Strategy | Algorithmic Formulation | Primary Use Case |
| :--- | :--- | :--- |
| **Uncertainty Sampling** | $U = 1.0 - \max(P) + \text{MarginBonus}$ | Ambiguous bounding boxes & low-confidence detections |
| **Diversity Sampling** | Farthest-Point / Greedy $k$-Center in 768D SigLIP Space | Eliminates redundant near-duplicate candidate batches |
| **Hybrid Sampling** | $0.40 \cdot U + 0.40 \cdot D + 0.20 \cdot F$ | Balanced active learning for real-world deployments |
| **Model Disagreement** | $\Delta(M_A, M_B)$ detection & localization deltas | Isolates samples where baseline and candidate clash |
| **Class-Aware Sampling** | Rare class penalty multiplier | Prevents starvation of $< 5\%$ representation classes |

---

## 3. Human Review Taxonomy

Supported human reviewer decisions:
- **`CONFIRMED`**: Prediction and bounding box geometry verified accurate.
- **`INCORRECT_PREDICTION`**: Model predicted false positive or incorrect category.
- **`ANNOTATION_ISSUE`**: Ground truth requires box adjustment or relabeling.
- **`VALID_HARD_EXAMPLE`**: Genuine high-difficulty valid operational scene.
- **`DUPLICATE`**: Redundant identical scene already represented.
- **`NOT_USEFUL`**: Blank, low-information, or corrupted image asset.
- **`NEEDS_MORE_REVIEW`**: Flagged for secondary domain expert review.
- **`SKIP`**: Passed without decision.

---

## 4. Keyboard-Driven Focus Review Mode

To accelerate expert review throughput:
- `C` &rarr; **Confirm Correct**
- `R` &rarr; **Reject / Incorrect Prediction**
- `A` &rarr; **Annotation Issue**
- `S` &rarr; **Skip Sample**
- `F` &rarr; **Flag for Secondary Review**
- `N` / `&rarr;` &rarr; **Next Sample**
- `P` / `&larr;` &rarr; **Previous Sample**
- `Esc` &rarr; **Exit Focus Session**

---

## 5. Dataset Versioning & Retraining Lineage

When a cycle completes and the researcher clicks **[ Commit Dataset Version ]**:
1. Creates immutable snapshot (e.g. `v12` &rarr; `v13`).
2. Records parent version, change log, and SHA-256 fingerprint.
3. Provides one-click handoff to:
   - **Training Lab** (`/training`): Controlled retraining.
   - **Benchmark Lab** (`/benchmarks`): Controlled before/after mAP validation.
   - **Visual Search** (`/search`): Neighborhood exploration.
   - **Embedding Explorer** (`/explorer`): Vector cluster analysis.

---

## 6. Stopping Criteria

Researchers can configure stopping conditions:
- Review budget reached ($K=50$ samples).
- Diminishing returns observed ($\Delta \text{mAP} < 0.5\%$).
- Target performance milestone achieved ($\text{mAP@50} \ge 85.0\%$).
- Candidate pool exhausted.

---
*VisionForge Active Learning & Human-in-the-Loop Architecture.*
