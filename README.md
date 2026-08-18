```text
  _   _ _     _             _____                    
 | | | (_)___(_) ___  _ __ |  ___|__  _ __ __ _  ___ 
 | | | | / __| |/ _ \| '_ \| |_ / _ \| '__/ _` |/ _ \
 \ \_/ / \__ \ | (_) | | | |  _| (_) | | | (_| |  __/
  \___/|_|___/_|\___/|_| |_|_|  \___/|_|  \__, |\___|
                                          |___/      
                   Computer Vision Research Workbench
```

> An open-source, reproducible Computer Vision Research Platform for dataset intelligence, model training, benchmarking, error analysis, Grad-CAM explainability, video understanding, and research experiment tracking.

---

## 📌 What is VisionForge?

**VisionForge** is a full-stack computer vision workbench designed for AI researchers and engineers. It covers the entire empirical research lifecycle:

$$\text{Dataset Intelligence} \longrightarrow \text{Model Training} \longrightarrow \text{Model Registry} \longrightarrow \text{Inference} \longrightarrow \text{Evaluation} \longrightarrow \text{Error Analysis} \longrightarrow \text{Explainability} \longrightarrow \text{Research Reports}$$

### Core Capabilities:
- **Dataset Intelligence & Versioning**: Multi-dimensional health scorecards, label geometry checks, pre-split validation, and leakage detection.
- **Model Training Lab**: Transfer learning and fine-tuning with YOLO11 and RT-DETR, real-time loss tracking, learning rate schedules, and checkpoint exports.
- **Model Registry**: Immutable version packaging, dependency tracking, metadata persistence, and hardware device targeting.
- **Evaluation & Benchmark Lab**: Multi-metric precision/recall/mAP analysis, IoU threshold curves, confusion matrices, and runtime profiling.
- **Diagnostic Error Analysis & Failure Gallery**: Isolated false positives, false negatives, poor localization, and active learning candidate curation.
- **Explainability & Attribution**: PyTorch Grad-CAM spatial heatmaps with target bounding box concentration scoring.
- **Video & Temporal Intelligence**: Multi-object trajectory tracking, spatial ROI zones, dwell time calculation, and temporal rule engine.
- **Multimodal Visual Search & Query**: 768-dimensional dense visual memory index for sub-millisecond similarity retrieval.
- **Research Experiment Tracking**: Immutable lineage graphs, hypothesis validation, and structured research report synthesis.

---

## 🚀 Quick Start (Docker Compose)

The fastest way to launch VisionForge is using Docker Compose:

```bash
# 1. Clone the repository
git clone https://github.com/mzayan-bit/VisionForge.git
cd VisionForge

# 2. Copy environment configuration
cp .env.example .env

# 3. Start containerized services
make up
# or: docker compose up -d --build
```

### Accessing VisionForge:
- **Frontend Workbench UI**: [http://localhost:3000](http://localhost:3000)
- **Backend REST API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

To stop the containers:
```bash
make down
# or: docker compose down
```

---

## 💻 Local Development Setup (Native)

### Prerequisites:
- Python `3.11+` and [`uv`](https://astral.sh/uv) installer
- Node.js `20+` and `npm`

```bash
# 1. Install dependencies
make install

# 2. Start local development servers (Backend :8000 + Frontend :3000)
make dev

# 3. Seed real COCO8 benchmark dataset into the workbench
make seed
```

---

## ⚙️ Environment Configuration

VisionForge reads environment variables from `.env`. Copy `.env.example` to start:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Active runtime environment (`development`, `production`). |
| `DEBUG` | `false` | Enable or disable verbose stack traces. |
| `LOG_LEVEL` | `INFO` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `BACKEND_PORT` | `8000` | Published backend port. |
| `FRONTEND_PORT` | `3000` | Published frontend port. |
| `DATA_DIR` | `~/.cache/visionforge` | Persistent storage directory for datasets and models. |
| `DEFAULT_DEVICE` | `auto` | Target compute engine (`auto`, `cpu`, `cuda`, `mps`). |
| `DATABASE_URL` | *None* | Optional PostgreSQL connection string. |
| `REDIS_URL` | *None* | Optional Redis cache/broker URL. |
| `QDRANT_URL` | *None* | Optional Qdrant vector database URL. |
| `NEO4J_URL` | *None* | Optional Neo4j graph database URL. |
| `MLFLOW_TRACKING_URI` | *None* | Optional MLflow tracking server URI. |

---

## 🧪 Running Tests & Quality Verification

```bash
# Run full automated test suite (233+ unit and lifecycle tests)
make test

# Run code style & lint checks (ruff & eslint)
make lint

# Automatically format code
make format
```

---

## 🧠 Training a Model

### 1. Local Transfer Learning (Apple Silicon M4 / CPU):
Execute transfer learning on the ingested COCO8 dataset:
```bash
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

### 2. Remote GPU Training (Google Colab):
For multi-GPU / CUDA acceleration, run the dedicated Colab script:
```bash
python scripts/train_colab.py
```
See [`docs/real-cv-experiment.md`](docs/real-cv-experiment.md) for full reproduction results.

---

## 📚 Detailed Documentation

- [System Architecture & Service Map](docs/architecture.md)
- [Production Deployment Guide](docs/deployment.md)
- [Local Development Guide](docs/development.md)
- [Real Computer Vision Lifecycle Experiment Report](docs/real-cv-experiment.md)

---

## 📄 License

VisionForge is released under the [MIT License](LICENSE).
