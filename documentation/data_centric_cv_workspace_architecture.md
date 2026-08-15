# VisionForge Data-Centric Computer Vision Workspace Architecture

## Executive Summary
VisionForge **Data-Centric Computer Vision Workspace** turns dataset intelligence into a scientifically disciplined data curation and quality assurance platform. It answers fundamental questions:
1. *"Is my dataset healthy and structurally sound?"*
2. *"What geometric, label, or image defects exist in it?"*
3. *"Is there visual redundancy or cross-split data leakage?"*
4. *"Which classes are rare or underrepresented across partitions?"*
5. *"Which samples are difficult for vision models and why?"*
6. *"Did dataset version curation actually improve downstream model benchmarks?"*

$$\text{Raw Dataset} \rightarrow \text{Deep Quality Audit} \rightarrow \text{Human Review Curation} \rightarrow \text{Immutable Version} \rightarrow \text{Controlled Retraining} \rightarrow \text{Dataset Diff}$$

---

## 1. Core Principles & Scientific Control

> [!IMPORTANT]
> **Zero Silent Mutation Guarantee**
> - VisionForge will **NEVER** silently delete, modify, or re-split user data.
> - Destructive actions (quarantine, removal, relabeling) require explicit confirmation by a human reviewer.
> - Transparent diagnostic flags (`CORRUPTED`, `ZERO_AREA_BOX`, `TINY_BOX`, `LEAKAGE_CANDIDATE`, `OUTLIER_CANDIDATE`) are diagnostic findings, not automated deletions.

---

## 2. Dataset Health Scorecard Dimensions

| Category | Description | Status Triggers |
| :--- | :--- | :--- |
| **Data Integrity** | Image file decoding and format validity | `CRITICAL` if unreadable files; `NEEDS_REVIEW` if $>15\%$ anomalies; `GOOD` otherwise |
| **Annotation Quality** | Bounding box spatial bounds and positive area | `CRITICAL` if zero-area/duplicate boxes; `NEEDS_REVIEW` if $>5$ warnings; `GOOD` otherwise |
| **Class Balance** | Representation parity across categories | `NEEDS_REVIEW` if $\ge 2$ classes under-represented ($<5\%$); `GOOD` otherwise |
| **Visual Diversity** | Feature space dispersion in 768D SigLIP space | Evaluates visual cluster density and coverage |
| **Potential Leakage** | Cross-split duplicate contamination | `CRITICAL` if exact hash match; `NEEDS_REVIEW` if near-duplicate ($\ge 95\%$ cosine similarity); `GOOD` otherwise |
| **Model Difficulty** | Hard sample density and failure concentration | `NEEDS_REVIEW` if $>20\%$ high difficulty samples; `GOOD` otherwise |

---

## 3. Diagnostic Quality Flags

### Image Quality Flags
- **`CORRUPTED`**: Image file fails decoding.
- **`VERY_SMALL`**: Image resolution $< 64\times 64\text{px}$.
- **`EXTREME_ASPECT_RATIO`**: Aspect ratio $> 4:1$ or $< 1:4$.
- **`LOW_INFORMATION`**: Abnormally low entropy or file size.

### Annotation Quality Flags
- **`ZERO_AREA_BOX`**: Bounding box with zero or negative width/height ($w \le 0$ or $h \le 0$).
- **`OUT_OF_BOUNDS_COORDINATES`**: Bounding box coordinates extend beyond image boundary.
- **`TINY_BOX`**: Bounding box $< 0.2\%$ of total image area.
- **`DUPLICATE_BOX`**: Redundant identical bounding box on same object ($\text{IoU} \ge 95\%$).
- **`OVERLAPPING_BOX`**: High overlap between conflicting classes ($\text{IoU} \ge 85\%$).

---

## 4. Cross-Partition Data Leakage Detection

> [!WARNING]
> **Data Leakage Contamination**
> 
> Cross-split duplicates between `Train` and `Test` partitions artificially inflate evaluation benchmarks. VisionForge separates:
> 1. **Exact Hash Match (`EXACT_HASH`)**: Bit-for-bit identical `SHA-256` content hash across partitions.
> 2. **Visual Representation Candidate (`VISUAL_SIMILARITY`)**: Dense cosine similarity $\ge 0.95$ in SigLIP vector space.

---

## 5. Hard Sample Prioritization

Difficult samples are ranked using an interpretable, transparent composite score:

$$\text{Prioritization} = 0.45 \cdot S_{\text{eval\_fail}} + 0.35 \cdot S_{\text{conf\_gap}} + 0.20 \cdot S_{\text{complexity}}$$

Where:
- $S_{\text{eval\_fail}}$: Benchmark evaluation failure frequency ($0.35$ per failure).
- $S_{\text{conf\_gap}}$: Model confidence margin ($1.0 - \text{confidence}$).
- $S_{\text{complexity}}$: Annotation box density ($0.15$ per bounding box).

---

## 6. Human Review Queue & Dataset Versioning

- Reviewers evaluate candidates across categories: `duplicate_review`, `leakage_review`, `annotation_review`, `outlier_review`, `hard_sample_review`.
- Decisions recorded: `ACCEPT`, `REJECT`, `NEEDS_CORRECTION`, `NOT_A_PROBLEM`, `DUPLICATE`, `INVALID`, `UNCERTAIN`.
- Every curation cycle produces an immutable version snapshot with parent version lineage and cryptographic `SHA-256` fingerprint.

---
*VisionForge Data-Centric Computer Vision Workspace Architecture.*
