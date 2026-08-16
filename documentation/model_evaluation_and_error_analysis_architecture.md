# VisionForge Model Evaluation & Deep Error Analysis Architecture

## 1. Overview & Core Mission
The VisionForge **Model Evaluation & Error Analysis Workspace** transitions model validation from superficial scalar metric reporting to deep, actionable diagnostic intelligence:

$$\text{Model Checkpoint} \longrightarrow \text{Strict Benchmark} \longrightarrow \text{Error Taxonomy} \longrightarrow \text{Failure Gallery} \longrightarrow \text{Dataset Intelligence} \longrightarrow \text{Active Learning}$$

---

## 2. Six-Part Diagnostic Error Taxonomy

| Category | Definition & Matching Criteria | Actionable Remediations |
| :--- | :--- | :--- |
| **Missed Object (False Negative)** | Unmatched ground truth object with zero overlapping predictions at threshold $\tau = 0.25$. | Add similar hard examples via Active Learning. |
| **False Positive (Background)** | Candidate prediction on background with zero ground truth object matches. | Introduce hard negative mining in dataset. |
| **Wrong Class (Misclassification)** | Spatial localization matched ($\text{IoU} \ge 0.50$), but category label mismatched (e.g. Helmet $\rightarrow$ Head). | Rebalance training loss and class weights. |
| **Poor Localization** | Overlapping candidate detection with sub-threshold IoU ($0.10 \le \text{IoU} < 0.50$). | Adjust bounding box regression loss ($L_{\text{DFL}}, L_{\text{CIoU}}$). |
| **Duplicate Detection** | Multiple overlapping high-confidence predictions assigned to already matched ground truth. | Tune Non-Maximum Suppression (NMS) IoU threshold. |
| **Small-Object Failure** | Target object area $< 32^2\text{ px}$ suffering from resolution attenuation. | Multi-scale training & high-res tiled inference. |

---

## 3. Transparent Prioritization & Clustering

### A. Review Priority Score
Every failure is assigned a deterministic, explainable priority score:
$$\text{Priority} = 0.40 \cdot (1 - \text{Confidence}) + 0.35 \cdot (1 - \text{IoU}) + 0.25 \cdot \text{ErrorWeight}$$

### B. Unsupervised Visual Clustering
Failures are grouped in 768D SigLIP visual embedding space into **`Cluster 1`**, **`Cluster 2`**, and **`Cluster 3`** to enable systematic error discovery without unsubstantiated semantic labeling.

---

## 4. Controlled Same-Dataset Model Comparison

Rigorous model comparison strictly enforces scientific control:
- **Baseline ($M_0$)** vs **Candidate ($M_1$)** must evaluate on the **exact same dataset snapshot, version, and split** (e.g. `safety_v2:v1.0.0 (test)`).
- Automatically measures **metric deltas** ($\Delta \text{mAP}$, $\Delta \text{Precision}$, $\Delta \text{Recall}$, $\Delta \text{Latency}$).
- Quantifies **failure deltas** (e.g. False Negatives: $-33$).
- Detects and flags **measurable regressions** (e.g. any class recall drop $> 5\%$).

---

## 5. Seamless Active Learning Integration
From any failure detail in the gallery, researchers can click **`[ Add to Active Learning ]`** to directly enqueue the difficult sample into the active learning cycle candidate pool for human review and retraining.

---
*VisionForge Model Evaluation & Deep Error Analysis Specification.*
