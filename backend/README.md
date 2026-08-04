# VisionForge Backend Core Architecture

The backend foundation for VisionForge — a modular, high-performance Python platform built with FastAPI, Pydantic v2, and `uv`.

---

## 🏛️ Architectural Principles

1. **Modular Domain Structure**: Decoupled core infrastructure supporting seamless extension for future engines (datasets, models, benchmarking, visualization, plugins).
2. **Vision Engine Execution Layer**: Central execution operating system (`VisionEngine`, `TaskManager`, `EnginePipeline`, `ExtensionRegistry`, `MetricsCollector`).
3. **AI Core Orchestration**: Unified `BaseVisionModel` abstract contract, `ModelRegistry`, `DeviceManager`, and `CacheManager` without model-specific code sprawl.
4. **Unified API Envelopes**: Consistent JSON response contracts (`APIResponse[T]`) across success, error, and validation outcomes.
5. **Pydantic Settings Layer**: Strongly-typed configuration system with environment variable overriding, `.env` file loading, and cached singleton access.
6. **Structured ANSI Logging**: Colored terminal output with ISO timestamps, logger hierarchy, request performance metrics, and stack trace formatting.
7. **Centralized Exception Handling**: Global error translation for domain exceptions, HTTP errors, and request validation failures into predictable JSON payloads.
8. **Dependency Injection**: First-class FastAPI dependency accessors for settings, system diagnostics, Vision Engine, AI Core registry, and device managers.

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
│   ├── engine/              # Vision Engine Execution Layer
│   │   ├── runner.py        # VisionEngine orchestrator facade & get_vision_engine singleton
│   │   ├── context.py       # ExecutionContext model
│   │   ├── task.py          # BaseVisionTask abstract class & TaskState enum
│   │   ├── manager.py       # TaskManager & get_task_manager singleton
│   │   ├── pipeline.py      # EnginePipeline & standard PipelineStage implementations
│   │   ├── extensions.py    # ExtensionRegistry for plugin hooks
│   │   ├── metrics.py       # MetricsCollector, ExecutionMetrics, & StageMetrics
│   │   └── exceptions.py    # EngineException hierarchy
│   ├── ai/                  # AI Core Package (Orchestration & Abstractions)
│   │   ├── base.py          # BaseVisionModel Abstract Base Class & ModelStatus
│   │   ├── types.py         # ModelMetadata, TaskType, InputType, OutputType, MemoryRequirements
│   │   ├── registry.py      # ModelRegistry & get_model_registry singleton
│   │   ├── schemas.py       # Generic InferenceResult[T], ExecutionMetadata, & InferenceError
│   │   ├── device.py        # DeviceManager, DeviceType, & HardwareCapabilities
│   │   └── cache.py         # CacheManager, CacheStats, & disk space management
│   └── core/                # Core platform abstractions
│       ├── config.py        # Pydantic BaseSettings & Environment validation
│       ├── logging.py       # ANSI color formatter & structured logger setup
│       ├── responses.py     # Generic APIResponse[T] envelope & error schemas
│       ├── exceptions.py    # Centralized exception classes & FastAPI handlers
│       ├── middleware.py    # X-Request-ID tracing, X-Process-Time, & CORS
│       ├── lifecycle.py     # Async lifespan context manager & uptime tracker
│       └── dependencies.py  # Dependency injection functions & type aliases
└── tests/                   # Pytest test suite (config, health, system, middleware, exceptions, ai_core, vision_engine)
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
| `MODEL_CACHE_DIR` | `str` | `~/.cache/visionforge/models` | Root path for cached checkpoints and artifacts |
| `DEFAULT_DEVICE` | `str` | `auto` | Default compute hardware target (`auto`, `cpu`, `cuda`, `mps`) |
| `MAX_CACHED_MODELS` | `int` | `3` | Maximum concurrent loaded models permitted in memory |

---

## ⚙️ Vision Engine Execution Layer

The Vision Engine coordinates execution lifecycle across validation, task creation, model resolution, pipeline stages, metrics, and error recovery:

```python
from visionforge.ai.types import TaskType
from visionforge.engine import get_vision_engine

engine = get_vision_engine()

# Execute task through full engine pipeline
result = await engine.run_task(
    task_type=TaskType.DETECTION,
    payload={"image": "frame_001.png"},
    model_name="yolo-v8-detector",
    device="auto",
)
```

---

## 🧪 Local Setup & Testing

### Run Tests
```bash
pytest -v
```

### Start Development Server
```bash
uv run uvicorn visionforge.main:app --reload --port 8000
```
