```text
  _   _ _     _             _____                    
 | | | (_)___(_) ___  _ __ |  ___|__  _ __ __ _  ___ 
 | | | | / __| |/ _ \| '_ \| |_ / _ \| '__/ _` |/ _ \
 \ \_/ / \__ \ | (_) | | | |  _| (_) | | | (_| |  __/
  \___/|_|___/_|\___/|_| |_|_|  \___/|_|  \__, |\___|
                                          |___/      
                   Computer Vision Research Workbench
```

> **VisionForge** is an open-source, research-oriented Computer Vision Workbench for dataset intelligence, model experimentation, multi-metric evaluation, failure analysis, Grad-CAM explainability, and reproducible research workflows.

[![CI Pipeline](https://github.com/mzayan-bit/VisionForge/actions/workflows/ci.yml/badge.svg)](https://github.com/mzayan-bit/VisionForge/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.2-black.svg)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## 📌 1. Overview

VisionForge is designed for computer vision researchers and machine learning engineers who need to move beyond ad-hoc training scripts. It bridges the gap between exploratory research, rigorous empirical benchmarking, and verifiable reproducibility.

Rather than treating models as isolated weight files, VisionForge unifies the entire computer vision research lifecycle into a coherent, observable platform:

$$\text{Dataset Intelligence} \longrightarrow \text{Transfer Learning} \longrightarrow \text{Model Registry} \longrightarrow \text{Inference} \longrightarrow \text{Evaluation} \longrightarrow \text{Error Analysis} \longrightarrow \text{Explainability} \longrightarrow \text{Research Lineage}$$

---

## 💡 2. Why VisionForge?

- **Zero Black-Box Opacity**: Integrates PyTorch Grad-CAM attribution scoring directly alongside benchmark precision/recall curves.
- **Data-Centric Quality Engineering**: Pre-training geometry verification and split leakage detection prevent corrupted experiments before training begins.
- **Fine-Grained Failure Mode Taxonomy**: Isolates False Positives, False Negatives, and Localization Errors to automatically populate active learning curation queues.
- **Cryptographic Reproducibility**: Tracks immutable SHA-256 DAG lineage connecting dataset manifests, training hyperparameters, random seeds, and evaluations.
- **Hardware-Portable Execution**: Operates out of the box on Apple Silicon (MPS) and standard CPUs without requiring an NVIDIA GPU for platform operations.

---

## ⚡ 3. Core Capabilities

| Research Area | Implemented Capability | Key Technologies |
| :--- | :--- | :--- |
| **Dataset Intelligence** | Geometry bounds checking, class distribution balance entropy, split leakage audit | NumPy, Pillow, Scikit-learn |
| **Model Registry** | Immutable version packaging, dependency tracking, hardware targeting | FastAPI, File-backed Registry |
| **Training Lab** | Transfer learning, learning rate schedules, live loss tracking, checkpoint exports | Ultralytics (YOLO11, RT-DETR), PyTorch |
| **Evaluation Lab** | Multi-threshold mAP@50-95, PR curves, confusion matrices, latency profiling | NumPy, Scikit-learn, TorchVision |
| **Error Analysis** | Failure mode taxonomy (FP, FN, Localization), active learning edge-case curation | PyTorch, Vector Similarity |
| **Explainability** | PyTorch Grad-CAM spatial heatmaps, target bounding box concentration ratios | PyTorch, NumPy, Pillow |
| **Visual Search** | 768-dimensional dense visual memory index, sub-millisecond similarity retrieval | SigLIP, NumPy Cosine Matrix |
| **Video Intelligence** | Continuous visual trajectory tracking, spatial ROI zones, dwell time calculations | OpenCV, Temporal Rule Engine |
| **Research Lineage** | Immutable DAG experiment tracking, decision gates, markdown report synthesis | SHA-256 Hashes, JSON-LD DAG |
| **Observability** | Request IDs (`X-Request-ID`), background job tracker, Prometheus metrics (`/metrics`) | Prometheus, FastAPI Middleware |

---

## 🏛 4. System Architecture

```mermaid
graph TD
    User["Researcher / Developer"] -->|HTTP / Web UI (Port 3000)| Frontend["VisionForge Frontend (Next.js 16)"]
    Frontend -->|Reverse Proxy /api/v1/*| Backend["VisionForge Backend API (FastAPI Port 8000)"]
    
    subgraph CoreDomainServices ["Core Application Services Layer"]
        Backend --> DSet["Dataset Intelligence & Quality Engine"]
        Backend --> Train["Training & Transfer Learning Engine (Ultralytics)"]
        Backend --> Eval["Evaluation & Benchmark Engine"]
        Backend --> Explain["Explainability & Attribution Engine (Grad-CAM)"]
        Backend --> VMem["Visual Memory & Search Engine"]
        Backend --> Video["Video Understanding & Temporal Events"]
        Backend --> Exp["Experiment Tracking & Lineage Graph"]
    end
    
    subgraph PersistenceLayer ["Persistent Storage Layer (Named Volume: visionforge_data)"]
        DSet -->|Manifests & Splits| FS_Datasets["/data/datasets/"]
        Train -->|Checkpoints & Weights| FS_Models["/data/models/"]
        VMem -->|768D Vector Index| FS_Memory["/data/memory/"]
        Eval -->|Error Records & Benchmarks| FS_Evaluations["/data/evaluations/"]
        Explain -->|Attribution Cache| FS_Explanations["/data/explanations/"]
        Exp -->|Lineage Records| FS_Experiments["/data/experiments/"]
    end
```

---

## 🔄 5. Research Workflow & State Machine

VisionForge guides researchers through an 8-stage state machine with human decision gates:

$$\text{Definition} \longrightarrow \text{Dataset Selection} \longrightarrow \text{Model Config} \longrightarrow \text{Training} \longrightarrow \text{Evaluation} \longrightarrow \text{Failure Analysis} \longrightarrow \text{Decision Gate} \longrightarrow \text{Report}$$

At the **Decision Gate**, researchers can:
- **`ACCEPT`**: Promote model to production registry and finalize publication report.
- **`INVESTIGATE`**: Loop edge cases into active learning queues for re-annotation.
- **`REJECT`**: Terminate hypothesis with recorded diagnostic failure reasoning.

---

## 🛠 6. Tech Stack

- **Backend API**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, `uv` package manager.
- **Computer Vision Core**: PyTorch 2.x, TorchVision, Ultralytics (YOLO11, RT-DETR), OpenCV, Pillow, Scikit-learn.
- **Frontend Workbench**: Next.js 16 (App Router), React 19, TypeScript, TailwindCSS v4, Lucide Icons.
- **Containerization & CI**: Docker, Docker Compose (Multi-stage builds), GitHub Actions CI.
- **Observability**: Prometheus Metrics Exposition (`/metrics`), Python structured logging.

---

## 🚀 7. Installation & Quick Start

### Option A: Single-Command Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/mzayan-bit/VisionForge.git
cd VisionForge

# 2. Copy environment template
cp .env.example .env

# 3. Start containerized platform
make up
# or: docker compose up -d --build
```

### Option B: Native Development Setup

```bash
# 1. Install dependencies (uv + npm)
make install

# 2. Start local servers (Backend :8000 + Frontend :3000)
make dev

# 3. Seed real COCO8 benchmark dataset
make seed
```

### Service Access Points:
- **Frontend Workbench**: [http://localhost:3000](http://localhost:3000)
- **Backend API & Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Probes**: [http://localhost:8000/health](http://localhost:8000/health) and [http://localhost:8000/ready](http://localhost:8000/ready)
- **Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## 🧠 8. Training a Model

### 1. Local Transfer Learning (Apple Silicon / CPU)
```bash
make seed
cd backend
uv run python -c "
from visionforge.datasets.adapters.coco8_adapter import COCO8Adapter
from visionforge.training.service import get_training_service
from visionforge.training.schemas import TrainingConfig

adapter = COCO8Adapter()
summary, manifest, profile = adapter.ingest_dataset()

train_svc = get_training_service()
config = TrainingConfig(
    experiment_name='exp_coco8_transfer',
    dataset_id='coco8',
    preparation_id=manifest.preparation_id,
    model_name='yolo11n.pt',
    epochs=2,
    batch_size=4,
    imgsz=320,
    learning_rate=0.01,
    random_seed=42,
    device='cpu',
)
run = train_svc.create_training_run(config)
print('Training completed with mAP@50:', run.test_evaluation.map50)
"
```

### 2. Remote GPU Training (Google Colab T4)
For multi-epoch CUDA training runs:
```bash
python scripts/train_colab.py --preparation-id prep_coco8_v1_0_0 --model yolo11n.pt --epochs 25 --device 0
```
See [`docs/real-cv-experiment.md`](docs/real-cv-experiment.md) for full empirical details.

---

## 📊 9. Real CV Validation Results

VisionForge was validated end-to-end on the official **COCO8** micro-benchmark (CC BY 4.0):

| Metric / Stage | Measured Value |
| :--- | :--- |
| **Dataset Volume** | 8 images, 30 ground truth annotations, 12 classes |
| **Training Speed** | 2 epochs in $<15$ seconds on Apple Silicon M4 CPU |
| **Test Set mAP@50** | **68.8%** |
| **Test Set Precision** | **62.7%** |
| **Test Set Recall** | **66.7%** |
| **Inference Latency** | **7.8 ms** (128.2 FPS) |
| **Grad-CAM Attribution** | **76.4%** activation concentration inside target bbox |

---

## 📁 10. Repository Structure

```text
VisionForge/
├── backend/                   # FastAPI backend service
│   ├── pyproject.toml         # Dependencies & tooling config (uv)
│   ├── Dockerfile             # Multi-stage production Python image
│   ├── visionforge/           # Python domain source code
│   │   ├── ai/                # Foundation model wrappers
│   │   ├── api/v1/            # REST API endpoints & route handlers
│   │   ├── core/              # Config, exceptions, logging, telemetry
│   │   ├── datasets/          # Dataset intelligence & adapters (COCO8)
│   │   ├── evaluation/        # Benchmarking & failure analysis
│   │   ├── events/            # Video temporal rule engine
│   │   ├── experiments/       # Research experiment & DAG lineage
│   │   ├── explainability/    # Grad-CAM spatial heatmaps
│   │   ├── inference/         # Model lifecycle & interactive inference
│   │   ├── memory/            # 768D visual vector index
│   │   ├── models/            # ModelManager registry
│   │   ├── search/            # Unified visual similarity search
│   │   ├── training/          # Ultralytics PyTorch trainer & adapter
│   │   ├── video/             # Video understanding & trajectory tracking
│   │   └── workflows/         # 8-stage research state machine
│   └── tests/                 # Comprehensive pytest suite (242+ tests)
├── frontend/                  # Next.js 16 Web Workbench application
│   ├── src/app/               # 22 interactive App Router pages
│   ├── src/components/        # Reusable design system UI components
│   ├── Dockerfile             # Multi-stage production Node.js image
│   └── package.json           # Dependencies & build scripts
├── docs/                      # Architectural & research documentation
│   ├── architecture.md        # Service map & network data flow
│   ├── deployment.md          # Docker Compose & operations runbook
│   ├── development.md         # Local developer setup guide
│   ├── observability.md       # Health probes & metrics catalog
│   ├── demo.md                # 5-10 minute live showcase script
│   ├── cv-description.md      # Resume & portfolio descriptions
│   ├── research-description.md# Research methodology & framework
│   ├── real-cv-experiment.md  # COCO8 empirical experiment report
│   └── release-notes.md       # v0.1.0 release notes & roadmap
├── scripts/                   # Developer automation & Colab scripts
├── docker-compose.yml         # Container orchestration manifest
├── Makefile                   # Developer task runner
├── CONTRIBUTING.md            # Open-source contribution guidelines
├── LICENSE                    # MIT License
└── README.md                  # Landing documentation
```

---

## 🧪 11. Testing & Quality Verification

```bash
# Run full automated test suite (242+ tests)
make test

# Run code style & linting checks (ruff & eslint)
make lint

# Automatically format codebase
make format
```

---

## 🔮 12. Roadmap

- **Completed (v0.1.0)**: End-to-end CV lifecycle, COCO8 transfer learning, Grad-CAM XAI, visual search, video trajectory tracking, research DAG lineage, Docker Compose, and Prometheus observability.
- **Current Focus**: Multi-dataset benchmarking expansion and interactive annotation active learning loops.
- **Future Directions (v0.2.0+)**: Distributed multi-GPU training orchestration, SAM2 segmentation foundation model adapters, and Kubernetes Helm deployment.

---

## 📚 13. Documentation Index

- [System Architecture & Service Map](docs/architecture.md)
- [Production Deployment Guide](docs/deployment.md)
- [Local Development Guide](docs/development.md)
- [Observability & Health Probes Guide](docs/observability.md)
- [5–10 Minute Demo Walkthrough](docs/demo.md)
- [Resume & CV Descriptions](docs/cv-description.md)
- [Research Methodology Framework](docs/research-description.md)
- [Real Computer Vision Experiment Report](docs/real-cv-experiment.md)
- [Release Notes & Roadmap](docs/release-notes.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

## 📄 14. License

VisionForge is released under the [MIT License](LICENSE).
