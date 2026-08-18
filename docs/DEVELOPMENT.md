# VisionForge Local Development Guide

---

## 1. Prerequisites & Tooling

To run VisionForge natively on your development workstation (macOS / Linux / WSL2), ensure the following tools are installed:

- **Python**: Version `3.11` or higher
- **uv**: Fast Python package installer & dependency manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Node.js**: Version `20.x` or `22.x` (LTS)
- **npm**: Version `10.x` or higher

---

## 2. Local Environment Setup

### 2.1 Clone & Configure Environment

```bash
git clone https://github.com/mzayan-bit/VisionForge.git
cd VisionForge

# Copy environment configuration
cp .env.example .env
```

### 2.2 Install Dependencies

```bash
# Using Makefile:
make install

# Or manually:
cd backend && uv sync --dev
cd ../frontend && npm ci
```

---

## 3. Running Local Development Servers

Start both the backend FastAPI server (with auto-reload on port `8000`) and the Next.js frontend development server (on port `3000`):

```bash
make dev
# or:
./scripts/dev.sh
```

- **Frontend Application**: [http://localhost:3000](http://localhost:3000)
- **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Direct Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 4. Seeding Initial Real Dataset

To populate the local Visual Memory Index, Dataset Intelligence profiles, and manifests with real computer vision data:

```bash
make seed
```

This ingests the official **COCO8** micro benchmark dataset (8 images, 30 annotations, 12 classes, CC BY 4.0) into `~/.cache/visionforge/datasets/` and indexes feature vectors in memory.

---

## 5. Running Tests & Quality Verification

```bash
# Run full automated test suite (backend pytest + frontend type check):
make test

# Run code style & linting checks:
make lint

# Automatically format code:
make format
```

---

## 6. Remote GPU Training vs Local Execution

VisionForge cleanly decouples platform operation from compute-intensive model training:

- **Local Machine (Apple Silicon M4 / CPU)**: Suitable for dataset exploration, visual memory search, active learning review, transfer learning smoke tests (1–3 epochs), model registration, and inference evaluations.
- **Remote Cloud / Google Colab (NVIDIA CUDA GPU)**: Suitable for large batch sizes ($>32$) and extended multi-epoch training ($>50$ epochs).

### Running Remote Colab Training:

1. Open the [Google Colab Notebook](https://colab.research.google.com/).
2. Run the automated headless script from the repository:
   ```bash
   python scripts/train_colab.py
   ```
3. Export the resulting `best.pt` checkpoint to your local or containerized VisionForge model registry via `ModelManager` or the Model Registry Web UI.

---

## 7. Resetting Development Environment

To wipe local cache files, training history, and reset the environment to a clean state:

> [!WARNING]
> This command is destructive and removes cached datasets, trained weights, and experiment records from `~/.cache/visionforge/` and Docker volumes.

```bash
make reset-dev
```
