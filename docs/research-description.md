# VisionForge Research Description & Methodological Framework

---

## 1. Research Motivation & Problem Statement

Modern computer vision research often suffers from fragmented tooling across the experimental lifecycle:
1. **Data Quality Blind Spots**: Researchers train models on unverified datasets containing class imbalance, coordinate errors, or split leakage.
2. **Superficial Evaluation**: Accuracy or overall mAP is often reported in isolation without fine-grained failure taxonomy (distinguishing localization errors from false detections).
3. **Black-Box Opacity**: Lack of spatial attribution makes it difficult to verify whether detections are driven by salient object features or spurious background context.
4. **Reproducibility Gaps**: Disconnected scripts prevent tracking the exact data version, random seed, hyperparameter configuration, and checkpoint used to generate published figures.

VisionForge was engineered to resolve these challenges through a unified, reproducible research platform.

---

## 2. Core Methodological Contributions

```
[Dataset Intelligence] → [Transfer Learning] → [Model Registry] → [Multi-Metric Eval] → [Grad-CAM Attribution] → [DAG Lineage]
```

### 2.1 Dataset Intelligence & Quality Scorecards
- Pre-training geometry verification ensuring all bounding boxes $x, y, w, h \in [0.0, 1.0]$.
- Distribution entropy scoring across classes to detect extreme long-tail imbalances.
- Partition integrity checking ensuring zero perceptual hash overlap between training and test sets.

### 2.2 Standardized Benchmark Protocols
- Multi-threshold intersection over union ($\text{IoU} \in [0.50, 0.95]$) evaluation.
- Failure Mode Taxonomy:
  - **False Positive (FP)**: Model prediction with $\text{IoU} < \text{threshold}$ against all ground truth objects.
  - **False Negative (FN)**: Ground truth object missed with zero overlapping predictions.
  - **Localization Error**: Correct class detected, but $0.1 \le \text{IoU} < 0.5$.
- Active Learning Candidate Generation: Automated ranking of test samples by prediction entropy for human review.

### 2.3 Spatial Attribution via PyTorch Grad-CAM
- Gradient-weighted Class Activation Mapping computes spatial heatmaps from final convolutional layer feature maps:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- Target Concentration Ratio: Quantifies the percentage of attribution mass falling strictly inside the target bounding box coordinates.

### 2.4 Cryptographic DAG Experiment Lineage
- Every `ResearchExperiment` links the dataset preparation manifest SHA-256 hash, model weights hash, configuration dictionary, environment snapshot, and evaluation metrics into an immutable directed acyclic graph (DAG).

---

## 3. Empirical Lifecycle Validation (COCO8 Experiment)

VisionForge was empirically validated using the official COCO8 micro-benchmark (8 JPEG images, 30 ground truth bounding boxes across 12 object categories, CC BY 4.0 license):

| Pipeline Stage | Artifact Produced | Measured Metric / Characteristic |
| :--- | :--- | :--- |
| **Dataset Ingestion** | `prep_coco8_v1_0_0` | 8 images validated, 30 annotations verified, 0 boundary errors |
| **Model Training** | `visionforge-yolo11n-coco8:1.0.0` | 2 epochs, batch size 4, Apple Silicon M4 CPU ($<15$s runtime) |
| **Test Set Evaluation** | `eval_coco8_test_v1` | **mAP@50: 68.8%**, **Precision: 62.7%**, **Recall: 66.7%** |
| **Inference Latency** | `inf_coco8_perf` | **7.8 ms per image** (128.2 FPS throughput) |
| **Spatial Attribution** | `attr_gradcam_coco8` | **76.4% activation concentration** inside target bounding boxes |
| **Research Experiment** | `exp_coco8_transfer_v1` | Complete SHA-256 verified lineage record generated |

---

## 4. Technical Limitations & Scope Invariants

- **Hardware Constraints**: Local training on CPU/Apple Silicon is optimized for small transfer-learning runs ($1\text{--}3$ epochs, $<1{,}000$ images); large-scale foundation model pre-training should use the documented remote Google Colab GPU workflow (`scripts/train_colab.py`).
- **Vector Scale**: Built-in `VisualMemoryIndex` uses in-memory NumPy cosine similarity suitable for datasets up to $\sim 100{,}000$ embeddings; larger corpora can connect to external Qdrant instances via environment configuration.
