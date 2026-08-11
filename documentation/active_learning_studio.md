# VisionForge Active Learning & Intelligent Sample Selection Architecture

## Executive Summary
VisionForge provides an **Active Learning + Intelligent Sample Selection Engine** to answer the research question: *"Which images should we label or inspect next?"*

Rather than selecting samples randomly or manually sifting through thousands of un-annotated images, VisionForge ranks candidates using a multi-signal scoring model that combines model prediction uncertainty, embedding space novelty, farthest-point visual diversity, historical failure relevance, and image quality metrics.

---

## 1. Multi-Signal Scoring Formulation

For any candidate image $i$, individual normalized signal scores $S \in [0.0, 1.0]$ are calculated:

### 1.1 Model Uncertainty Proxy
Model prediction uncertainty $S_u$ evaluates object detection confidence ambiguity:

$$S_u = 1.0 - \max_{j} (\text{confidence}_j) + \text{Bonus}_{\text{competing}}$$

High uncertainty scores correspond to predictions with low maximum confidence (e.g., confidence 0.35 $\rightarrow$ uncertainty 0.65) or competing overlapping bounding boxes with narrow confidence margins ($< 0.15$).

### 1.2 Embedding Novelty Score
Novelty $S_n$ measures Euclidean/Cosine distance from the candidate's 768-d embedding vector $v_i$ to the nearest neighbors in the indexed dataset matrix:

$$d_{\text{min}} = \min_{j \in \text{Dataset}} (1.0 - v_i \cdot v_j)$$

$$S_n = \frac{1}{1.0 + e^{-6(d_{\text{min}} - 0.4)}}$$

### 1.3 Farthest-Point Visual Diversity Sampling
To prevent selecting 100 visually redundant images, VisionForge employs **Greedy k-Center (Farthest-Point) Sampling**:
1. Select candidate $c_1$ with highest combined uncertainty/novelty as initial seed $S = \{c_1\}$.
2. Iteratively select candidate $c_k$ maximizing minimum pairwise distance to already chosen samples:
   $$c_k = \arg\max_{c \notin S} \left( \min_{s \in S} \| v_c - v_s \| \right)$$
3. Assign $S_d$ based on distance to nearest selected neighbor.

### 1.4 Composite Selection Score
$$S_{\text{composite}} = w_u \cdot S_u + w_n \cdot S_n + w_d \cdot S_d + w_f \cdot S_f + w_q \cdot S_q$$

Normalized weights satisfy $\sum w_i = 1.0$.

---

## 2. Selection Strategies

| Strategy | Primary Objective | Weight Profile ($w_u, w_n, w_d, w_f$) |
| :--- | :--- | :--- |
| `UNCERTAINTY` | Sample ambiguous prediction candidates | $(0.70, 0.10, 0.10, 0.10)$ |
| `DIVERSITY` | Maximize visual space coverage | $(0.10, 0.20, 0.60, 0.10)$ |
| `UNCERTAINTY_DIVERSITY` | Balanced uncertainty & visual coverage | $(0.40, 0.25, 0.25, 0.10)$ |
| `NOVELTY` | Target underrepresented visual regions | $(0.10, 0.70, 0.10, 0.10)$ |

---

## 3. Mandatory Test-Set Protection Guarantee

> [!IMPORTANT]
> **Strict Evaluation Test-Set Protection**
>
> The candidate pool validation engine strictly inspects candidate file paths against dataset split manifests (`split == "test"`). Any candidate path matching test set patterns is automatically blocked, and pure test-set candidate pools raise a `TestSetProtectionError` (HTTP 400). This guarantees that evaluation test sets are **NEVER** exposed to active learning selection or retraining.

---

## 4. Human-in-the-Loop Review Workflow

VisionForge enforces a strict **Human-in-the-Loop** model. Recommended samples do **NOT** automatically enter training:

```
Candidate Pool
   │
   ▼
Active Learning Multi-Signal Ranking
   │
   ▼
Human Review Queue (Accept / Reject / Skip / Mark for Labeling)
   │
   ▼
Labeling Workbench
   │
   ▼
Dataset Preparation (New Dataset Version)
   │
   ▼
Controlled Training Experiment (User Triggered)
```

---

## 5. Limitations

> [!WARNING]
> **Active Learning Performance Non-Guarantee**
>
> Active learning sample selection does **NOT** guarantee immediate model accuracy gains. The actual performance benefit of newly labeled active-learning candidates must be empirically verified by running a controlled training run and evaluating mAP on the untouched test split.

---
*VisionForge Active Learning Documentation.*
