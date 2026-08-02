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

### 1. API Core & Platform Abstractions (`backend/visionforge`)

- `visionforge.main`: Application factory (`create_app`) mounting routers, middleware, exception handlers, and root metadata.
- `visionforge.core.config`: Centralized Pydantic Settings manager (`VisionForgeSettings`) with environment variable validation, `.env` support, and `@lru_cache` singleton accessor.
- `visionforge.core.logging`: ANSI colored console logging with ISO timestamps, logger scoping (`visionforge.*`), and request timing logs.
- `visionforge.core.responses`: Unified generic response envelope (`APIResponse[T]`) for consistent JSON contracts across success, error, and validation outputs.
- `visionforge.core.exceptions`: Centralized exception handlers translating custom domain exceptions (`VisionForgeException`), HTTP errors, and validation failures.
- `visionforge.core.middleware`: Request tracing (`X-Request-ID`), process timing (`X-Process-Time`), and CORS middleware.
- `visionforge.core.lifecycle`: Async lifespan context manager tracking application boot time and registered route tree.
- `visionforge.core.dependencies`: FastAPI dependency injection layer providing accessors for settings, system diagnostics, and app state.
- `visionforge.api.v1`: Versioned REST API layer (`/api/v1/health`, `/api/v1/system/info`).

### 2. Workbench Frontend (`frontend/src`)

- Built on Next.js App Router using React 19 and TailwindCSS.
- `src/components/`: Atomic, reusable UI components (headers, status indicators, metrics cards).
- `src/app/`: Server-rendered page layouts and route handlers.

### 3. Developer Tooling & Verification (`scripts/`)

- `dev.sh`: Spawns backend and frontend servers concurrently for local development.
- `lint.sh`: Runs Ruff check/formatting, Pytest suites, and Next.js TypeScript build validation.
