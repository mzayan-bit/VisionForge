# VisionForge Backend API Service

The backend layer for VisionForge — a high-performance Python application built with FastAPI, Pydantic v2, and `uv`.

## Architecture Overview

- `visionforge/main.py`: ASGI application lifecycle and router registration.
- `visionforge/config.py`: Strongly typed environment settings.
- `visionforge/api/v1/`: Versioned REST API endpoints.
- `visionforge/core/`: Application cross-cutting utilities (logging, core abstractions).

## Setup & Running

Install dependencies and start development server using `uv`:

```bash
uv venv
uv pip install -e ".[dev]"
uv run uvicorn visionforge.main:app --reload --port 8000
```
