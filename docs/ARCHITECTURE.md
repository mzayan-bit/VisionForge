# VisionForge Architecture Specification

## Overview

VisionForge is designed around three fundamental core principles: **Decoupled Architecture**, **Standardized Adapter Interfaces**, and **High DX Tooling**.

```text
               +-----------------------------+
               |     Next.js Frontend        |
               +--------------+--------------+
                              | REST / WS
               +--------------v--------------+
               |      FastAPI Core Backend    |
               +--------------+--------------+
                              |
       +----------------------+----------------------+
       |                      |                      |
+------v-------+      +-------v------+      +--------v-------+
|    Model     |      |  Benchmark   |      |  Visualization |
|   Registry   |      |    Engine    |      |    Pipeline    |
+--------------+      +--------------+      +----------------+
```

## System Components

### 1. API Core & Lifecycle (`backend/visionforge`)

- `visionforge.main`: Entry point for the FastAPI ASGI service. Handles lifespan hooks, middleware registration (CORS), and router attachments.
- `visionforge.config`: Central settings manager powered by Pydantic `BaseSettings`. Loads configurations from environment variables or `.env` files safely.
- `visionforge.api.v1`: Versioned REST API layer exposing system status, health checks, and future model management routes.

### 2. Workbench Frontend (`frontend/src`)

- Built on Next.js App Router using React 19 and TailwindCSS.
- `src/components/`: Atomic, reusable UI components (headers, status indicators, metrics cards).
- `src/app/`: Server-rendered page layouts and route handlers.

### 3. Developer Tooling & Verification (`scripts/`)

- `dev.sh`: Spawns backend and frontend servers concurrently for local development.
- `lint.sh`: Runs Ruff check/formatting, Pytest suites, and Next.js TypeScript build validation.
