# VisionForge Developer Guide

Welcome to the VisionForge development guide. This document outlines how to set up your environment, follow code style standards, and submit pull requests.

## Development Environment Setup

### Prerequisites

- Python 3.11 or later
- [uv package manager](https://astral.sh/uv)
- Node.js 20+ and npm

### Backend Setup

Initialize the virtual environment and install development dependencies:

```bash
cd backend
uv venv
uv pip install -e ".[dev]"
```

Run tests and linter:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

### Frontend Setup

Install Node packages and run local server:

```bash
cd frontend
npm install
npm run dev
```

Run frontend build verification:

```bash
npm run build
```

## Unified Workspace Verification

Before committing changes, execute the workspace quality script:

```bash
./scripts/lint.sh
```

## Commit Guidelines

Keep commits logical, focused, and descriptive. Follow imperatively framed commit messages (e.g., `added documentation`, `set up backend`).
