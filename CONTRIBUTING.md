# Contributing to VisionForge

Thank you for your interest in contributing to VisionForge! VisionForge is an open-source, research-oriented computer vision platform built with Python (FastAPI + PyTorch/Ultralytics) and Next.js 16.

---

## 1. Code of Conduct & Core Principles

- **Technical Integrity**: Never commit fabricated metrics, fake benchmarks, or hardcoded evaluation scores.
- **Hardware Portability**: Ensure core services remain executable without mandatory NVIDIA GPU hardware (CPU/MPS default).
- **Clean Architecture**: Decouple domain services from REST controllers and keep database/broker integrations strictly optional.

---

## 2. Development Setup

### Prerequisites
- **Python**: `>= 3.11`
- **uv**: Fast Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Node.js**: `20.x` or `22.x` (LTS) & `npm`
- **Docker**: Optional, for containerized execution

### Initial Setup
```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/VisionForge.git
cd VisionForge

# 2. Copy environment template
cp .env.example .env

# 3. Install dependencies
make install
```

---

## 3. Development Workflow

```bash
# Start local development servers (Backend :8000 + Frontend :3000)
make dev

# Ingest test dataset
make seed
```

---

## 4. Quality Verification & Testing

Before submitting a Pull Request, verify that all linters, type checks, and automated test suites pass:

```bash
# Run full automated test suite (242+ tests)
make test

# Run code style & linting checks (ruff & eslint)
make lint

# Automatically format code
make format
```

---

## 5. Pull Request Guidelines

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Do **not** commit datasets, video files, trained weights (`*.pt`), or `.env` files.
4. Open a Pull Request against `main` using the provided PR template.
