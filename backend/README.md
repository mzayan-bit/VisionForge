# VisionForge Backend Core Architecture

The backend foundation for VisionForge — a modular, high-performance Python platform built with FastAPI, Pydantic v2, and `uv`.

---

## 🏛️ Architectural Principles

1. **Modular Domain Structure**: Decoupled core infrastructure supporting seamless extension for future engines (datasets, models, benchmarking, visualization, plugins).
2. **Unified API Envelopes**: Consistent JSON response contracts (`APIResponse[T]`) across success, error, and validation outcomes.
3. **Pydantic Settings Layer**: Strongly-typed configuration system with environment variable overriding, `.env` file loading, and cached singleton access.
4. **Structured ANSI Logging**: Colored terminal output with ISO timestamps, logger hierarchy, request performance metrics, and stack trace formatting.
5. **Centralized Exception Handling**: Global error translation for domain exceptions, HTTP errors, and request validation failures into predictable JSON payloads.
6. **Dependency Injection**: First-class FastAPI dependency accessors for settings, system diagnostics, and application state.

---

## 📂 Folder Layout

```text
backend/
├── visionforge/
│   ├── main.py              # Application factory (create_app) & entrypoint
│   ├── config.py            # Backward-compatible settings export bridge
│   ├── api/
│   │   └── v1/              # Versioned API routes
│   │       ├── router.py    # API v1 router registry
│   │       ├── health.py    # Health diagnostics endpoint (/api/v1/health)
│   │       └── system.py    # System runtime info endpoint (/api/v1/system/info)
│   └── core/                # Core platform abstractions
│       ├── config.py        # Pydantic BaseSettings & Environment validation
│       ├── logging.py       # ANSI color formatter & structured logger setup
│       ├── responses.py     # Generic APIResponse[T] envelope & error schemas
│       ├── exceptions.py    # Centralized exception classes & FastAPI handlers
│       ├── middleware.py    # X-Request-ID tracing, X-Process-Time, & CORS
│       ├── lifecycle.py     # Async lifespan context manager & uptime tracker
│       └── dependencies.py  # Dependency injection functions & type aliases
└── tests/                   # Pytest test suite (config, health, system, middleware, exceptions)
```

---

## ⚙️ Configuration System

Configuration is defined in `visionforge.core.config.VisionForgeSettings` using `pydantic_settings.BaseSettings`:

- **Environment Support**: Loads settings from `.env` file or environment variables automatically.
- **Environments**: `development`, `staging`, `production`, `testing`.
- **Singleton Access**: Use `get_settings()` (cached with `@lru_cache`).

### Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | `str` | `development` | Target execution environment |
| `DEBUG` | `bool` | `False` | Enable detailed debug logging & stack traces |
| `HOST` | `str` | `0.0.0.0` | Bind host IP address |
| `PORT` | `int` | `8000` | Bind HTTP server port |
| `LOG_LEVEL` | `str` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CORS_ORIGINS` | `List[str]` | `["http://localhost:3000"]` | Allowed HTTP origin list for CORS |

---

## 📬 Unified Response Models

All API responses follow the standard `APIResponse[T]` model:

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": { ... },
  "meta": {
    "timestamp": "2026-08-03T00:05:00+00:00"
  },
  "error": null
}
```

If an error occurs:

```json
{
  "success": false,
  "message": "Resource not found",
  "data": null,
  "meta": {
    "timestamp": "2026-08-03T00:05:00+00:00",
    "request_id": "85960415-e7b8-4829-9db9-5307296a943d"
  },
  "error": {
    "code": "NOT_FOUND",
    "message": "Resource not found",
    "details": null
  }
}
```

---

## 🛡️ Middleware & Tracing

Every incoming request passes through `RequestTracingMiddleware`:
- `X-Request-ID`: Generated UUID4 or preserved incoming request trace ID.
- `X-Process-Time`: Recorded execution latency (e.g. `1.23ms`).
- Structured logging: `[REQ] GET /api/v1/health -> 200 OK (1.23ms) [req_id=...]`.

---

## 🧪 Local Setup & Testing

### Virtual Environment Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run Tests
```bash
pytest -v
```

### Start Development Server
```bash
uv run uvicorn visionforge.main:app --reload --port 8000
```
