# VisionForge Model Management Architecture

VisionForge implements a production-grade model management subsystem designed to handle the discovery, installation, validation, and lifecycle of computer vision models. It operates conceptually similar to package managers or advanced ML model hubs (e.g., Hugging Face, Ollama), ensuring users never manually interact with the filesystem.

## Core Philosophy

- **Abstracted Filesystem:** Users and developers interact with models via the API or UI. Direct filesystem manipulation is not required.
- **Strong Typing:** All models follow strict metadata schemas ensuring programmatic guarantees during inference.
- **Deterministic Storage:** Model directories are strictly versioned (`models/{name}/{version}/`) to prevent collisions and support rollback.

## Architecture Components

### 1. Model Manager (`visionforge.models.manager.ModelManager`)
The central orchestrator for all model operations. It acts as the gatekeeper for:
- Discovering installed models.
- Managing the installation state machine.
- Validating models on disk.
- Exposing hardware requirements and disk usage stats.

### 2. Storage System (`visionforge.models.storage.ModelStorage`)
Manages the physical filesystem layout, ensuring thread-safe, atomic operations.

```text
~/.cache/visionforge/
├── models/         # Final resting place for installed models
│   └── yolov8/
│       └── 1.0.0/  # Versioned model directories
├── downloads/      # In-flight model downloads
├── temp/           # Extraction and validation scratch space
└── metadata/       # Index of installed models
```

The storage system utilizes the application's configuration system to dynamically resolve the `model_cache_dir`, avoiding hardcoded paths.

### 3. Metadata System (`visionforge.models.metadata`)
Every installed model tracks extensive typed metadata via `InstalledModelMetadata`:
- **Identity:** Name, Version, Source
- **Requirements:** Task, Framework, Supported Devices, Memory Requirements
- **Lifecycle:** Install Date, Last Used, Status (Installing, Installed, Error)
- **Telemetry:** Disk Size Bytes

### 4. Validation Engine (`visionforge.models.validation`)
Models are validated before, during, and after installation:
- **Name/Version Validation:** Enforces clean, URL-safe naming conventions.
- **Integrity Validation:** Ensures model weights exist and metadata is uncorrupted.
- Graceful error reporting prevents corrupted models from crashing the `VisionEngine`.

## Installation Lifecycle

Future model downloads follow a strict, transactional pipeline:

1. **Validate Request:** Verify model name, version, and hardware capability.
2. **Resolve Destination:** Create temporary download directories.
3. **Download:** Stream weights into the temp directory.
4. **Verify Integrity:** Calculate SHA-256 hashes against the manifest.
5. **Register Model:** Persist `InstalledModelMetadata`.
6. **Finalize:** Atomically move from `downloads/` to `models/`.

*Note: The current implementation provides the architectural pipeline; actual network downloading of model weights will be implemented in subsequent phases.*

## API Integration

Clean REST endpoints provide the frontend with necessary telemetry:
- `GET /api/v1/models` - List installed models
- `GET /api/v1/models/{name}` - Retrieve detailed metadata
- `GET /api/v1/models/status` - Manager health and count
- `GET /api/v1/models/storage` - Disk usage metrics
- `POST /api/v1/models/{name}/validate` - Trigger deep integrity check

## Future Extension Strategy

The Model Manager is designed for massive scalability:
- **Remote Registries:** The `ModelSource` schema can easily be extended to support custom enterprise registries or S3 buckets.
- **Model Quantization:** Future iterations can store quantization parameters (e.g., INT8, FP16) in the metadata schema, allowing the `VisionEngine` to dynamically select the optimal weights.
- **Background Workers:** The installation pipeline is designed to be offloaded to background tasks (e.g., Celery or FastAPI BackgroundTasks) for massive LLM/VLM downloads.
