# VisionForge Research Benchmark & Evaluation Lab Architecture

## Executive Summary
VisionForge **Research Benchmark & Evaluation Lab** provides a rigorous, scientifically validated platform to answer critical model evaluation questions:
1. *"How good is this computer vision system?"*
2. *"Did candidate Model B actually improve over baseline Model A?"*
3. *"Where does the model fail and under what error categories?"*
4. *"Are improvements statistically and visually meaningful?"*

$$\text{Dataset Snapshot} + \text{Model Version} + \text{Controlled Protocol} \rightarrow \text{Authentic Metrics} + \text{Diagnostic Failures} + \text{Reproducible Telemetry}$$

---

## 1. Core Principles & Scientific Control

> [!IMPORTANT]
> **Zero Fabrication Policy**
> - No fake or hallucinated metrics.
> - Precision, Recall, F1, and AP are calculated using standard 101-point COCO interpolation across IoU thresholds $[0.50, 0.55, \dots, 0.95]$.
> - Strict Fair Comparison enforcement: Comparing two models validates matching `dataset_id`, `dataset_version`, `split_used`, and `task`. If conditions differ, the system flags `INCOMPARABLE`.

---

## 2. Detection Metrics Mathematical Definitions

| Metric | Mathematical Definition | Standard Baseline |
| :--- | :--- | :--- |
| **IoU** | $\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$ | $\ge 0.50$ (Detection threshold) |
| **Precision** | $P = \frac{TP}{TP + FP}$ | Evaluated at confidence threshold |
| **Recall** | $R = \frac{TP}{TP + FN}$ | Evaluated at confidence threshold |
| **F1 Score** | $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean of $P$ and $R$ |
| **AP@50** | $\int_0^1 P(r) dr \approx \frac{1}{101} \sum_{i=0}^{100} P_{\text{interp}}(r_i)$ at $\text{IoU} = 0.50$ | VOC Standard |
| **mAP@50:95** | $\frac{1}{10} \sum_{\text{IoU}=0.50}^{0.95} \text{AP}_{\text{IoU}}$ | COCO Standard |

---

## 3. Diagnostic Error Taxonomy

VisionForge categorizes all prediction errors into structured failure classes:

1. **`FALSE_POSITIVE`**: High confidence detection with no ground truth object overlap.
2. **`FALSE_NEGATIVE`**: Ground truth annotation missed by all candidate predictions.
3. **`MISCLASSIFICATION`**: High IoU ($\ge 0.50$) spatial match with wrong class category label.
4. **`POOR_LOCALIZATION`**: Correct class match with sub-threshold IoU ($0.10 \le \text{IoU} < 0.50$).
5. **`LOW_CONFIDENCE`**: Accurate bounding box suppressed below operating confidence threshold.
6. **`DUPLICATE_DETECTION`**: Redundant multiple detections on the same ground truth object.
7. **`BACKGROUND_DETECTION`**: Hallucinated detection on empty background imagery.

---

## 4. Controlled Model Comparison & Regression Detection

When comparing candidate model $M_B$ against baseline $M_A$:
- **Absolute Delta**: $\Delta = \text{Metric}(M_B) - \text{Metric}(M_A)$
- **Relative Delta**: $\Delta\% = \frac{\Delta}{\text{Metric}(M_A)} \times 100\%$
- **Regression Trigger**:
  - $\Delta \text{mAP@50} < -2.0\%$ triggers `REGRESSION`.
  - $\Delta \text{Latency} > +10.0\%$ triggers `REGRESSION`.
  - $\Delta \text{mAP@50} > +2.0\%$ triggers `IMPROVED`.
  - Otherwise `NEUTRAL`.

---

## 5. Runtime & Latency Benchmarking Protocol

> [!TIP]
> **Warm-Up Isolation**
>
> Steady-state latency is measured over $N \ge 30$ iterations after discarding initial $W=5$ warm-up iterations. Latency reports mean, median, and 95th percentile (p95) across Preprocessing, Forward Inference, and Postprocessing/NMS stages.

---

## 6. Reproducibility Guarantee

Every `BenchmarkRun` stores:
- **Dataset Cryptographic Fingerprint** (`SHA-256` of dataset manifest)
- **Environment Snapshot** (OS platform, CPU architecture, GPU device, Python version, PyTorch version)
- **Git Commit SHA** and working tree status
- **Random Seeds** (Python, NumPy, PyTorch RNGs)

---
*VisionForge Research Benchmark & Evaluation Lab Architecture.*
