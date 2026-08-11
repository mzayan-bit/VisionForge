# VisionForge Experiment Tracking, Data/Model Lineage, and Reproducibility Architecture

## Executive Summary
VisionForge provides an immutable, end-to-end **Experiment Tracking, Data/Model Lineage, and Reproducibility System**. It answers the core scientific question: *"What exactly produced this result?"* for any model checkpoint, evaluation metric, or inference prediction across the platform lifecycle.

---

## 1. Lineage Graph Architecture

VisionForge models dependency lineage as a directed acyclic graph (DAG):

```
Dataset (v2.0)
   │
   ▼
Preparation (#12)
   │
   ▼
Training Run (#18)
   │
   ▼
Model Checkpoint (v3.0) ───┬───► Evaluation Run (#21)
                           │
                           ├───► Benchmark Run (#7)
                           │
                           └───► Inference Result (#92)
```

### Resource References:
To prevent storage duplication, experiments link entities using unique string keys (`dataset_id`, `preparation_id`, `training_run_ids`, `model_ids`, `evaluation_ids`, `benchmark_ids`, `inference_ids`).

---

## 2. Dataset SHA-256 Fingerprinting Protocol

A dataset version has a stable cryptographic fingerprint computed from its manifest JSON:

$$\text{ManifestHash} = \text{SHA-256}(\text{CanonicalJSON}(\text{Manifest}))$$

$$\text{DatasetFingerprint} = \text{SHA-256}(\text{dataset\_id} \mathbin{\Vert} \text{version} \mathbin{\Vert} \text{num\_samples} \mathbin{\Vert} \text{num\_classes} \mathbin{\Vert} \text{ManifestHash})$$

This fingerprint verifies whether the dataset content or manifest split has changed since the experiment was created.

---

## 3. Cryptographic Artifact Checksums

Important experimental artifacts (PyTorch checkpoints `.pt`, metric JSON logs, overlay JPEG images, benchmark summaries) are verified using streaming SHA-256 checksums:

```python
hasher = hashlib.sha256()
with open(artifact_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        hasher.update(chunk)
sha256_checksum = hasher.hexdigest()
```

---

## 4. Environment Snapshots & Git Tracking

Whenever an experiment is initialized, an immutable `EnvironmentSnapshot` is captured:
- **Python Version**: e.g., `3.11.15`
- **OS Platform**: e.g., `Darwin 24.3.0 (arm64)`
- **Framework Version**: PyTorch `2.2.0`
- **Git Commit SHA**: `git rev-parse HEAD` (e.g. `17aa89269...`)
- **Git Branch**: `git rev-parse --abbrev-ref HEAD`
- **Git Clean Status**: `is_working_tree_clean`

---

## 5. Reproduction Workflow

Reproducing an experiment does **NOT** mutate historical data. Instead:
1. `POST /api/v1/experiments/{exp_id}/reproduce` reads the parent experiment's configuration snapshot, dataset fingerprint, and random seed.
2. Spawns a **new Experiment** in `DRAFT` state pre-populated with parent parameters and sets `parent_experiment_id = exp_id`.
3. The new run receives a unique ID (`exp_...`) and records a new execution environment snapshot.

---

## 6. Scientific Reproducibility Limitations

> [!WARNING]
> **Floating-Point & Hardware Non-Determinism**
>
> While VisionForge guarantees **metadata, configuration, and code reproducibility**, bit-for-bit identical floating-point tensors across different hardware devices (e.g., Apple M4 MPS vs NVIDIA T4 CUDA) cannot be guaranteed due to atomic GPU reduction operations and kernel instruction order differences.

---
*VisionForge Experiment Lineage & Audit Documentation.*
