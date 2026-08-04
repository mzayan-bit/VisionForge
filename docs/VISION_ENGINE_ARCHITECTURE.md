# VisionForge Vision Engine Architecture Specification

## ⚙️ Philosophy: The Operating System of VisionForge

The **Vision Engine** acts as the central execution layer and operating system of VisionForge.

Every computer vision operation—whether object detection, instance segmentation, depth estimation, OCR, vision-language model prompting, image retrieval, video understanding, or 3D reconstruction—is executed through this unified engine rather than through model-specific or endpoint-specific code sprawl.

---

## 🏛️ Vision Engine Package Structure

```text
backend/visionforge/engine/
├── __init__.py       # Package exports
├── runner.py         # VisionEngine orchestrator facade & get_vision_engine singleton
├── context.py        # ExecutionContext model (request_id, task_id, device, options, settings)
├── task.py           # BaseVisionTask abstract class & TaskState lifecycle enum
├── manager.py        # TaskManager & get_task_manager singleton
├── pipeline.py       # EnginePipeline & standard PipelineStage implementations
├── extensions.py     # ExtensionRegistry for plugin stage and task factory hooks
├── metrics.py        # MetricsCollector, ExecutionMetrics, & StageMetrics
└── exceptions.py     # EngineException hierarchy & error recovery descriptors
```

---

## 🔄 Execution Lifecycle Flow

```text
Incoming Request
      │
      ▼
1. Validation (Request context, task parameters, option specs)
      │
      ▼
2. Task Creation (ExecutionContext & TaskManager registration)
      │
      ▼
3. Model Resolution (ModelRegistry lookup & compute target resolution)
      │
      ▼
4. Execution (Pipeline stages: PreProcessing -> Model.predict -> PostProcessing)
      │
      ▼
5. Result Formatting (Formatting into standardized InferenceResult envelope)
      │
      ▼
Structured Response
```

---

## 📌 Task Lifecycle States (`TaskState`)

Tasks transition through explicit lifecycle states:

```text
CREATED ➔ QUEUED ➔ VALIDATING ➔ PREPROCESSING ➔ EXECUTING ➔ POSTPROCESSING ➔ FORMATTING ➔ COMPLETED
                                                                                          └──> FAILED
                                                                                          └──> CANCELLED
```

---

## 🧩 Pipeline Architecture (`EnginePipeline`)

The execution pipeline runs tasks through ordered `PipelineStage` abstractions:

1. **`ValidationStage`**: Validates request inputs and context.
2. **`PreProcessingStage`**: Contract stage for input normalization.
3. **`ModelExecutionStage`**: Resolves target model from `ModelRegistry` and invokes `model.predict()`.
4. **`PostProcessingStage`**: Contract stage for post-processing model tensor outputs.
5. **`ResultFormattingStage`**: Envelopes output into standard `InferenceResult[T]`.

---

## 🔌 Extension Hooks (`ExtensionRegistry`)

Future plugins and custom tasks extend engine behavior cleanly without modifying core source code:

- **Custom Pipeline Stages**: `extension_registry.register_stage(custom_stage, position=2)`
- **Task Factories**: `extension_registry.register_task_factory(TaskType.DETECTION, factory_fn)`
- **Post-Processors**: `extension_registry.register_post_processor(TaskType.SEGMENTATION, post_proc_fn)`

---

## 📊 Metrics Collection & Error Recovery

### Metrics (`MetricsCollector`)
Records per-stage latency (`StageMetrics`), device backend used, memory metrics, warnings, and errors.

### Centralized Error Recovery
Engine execution failures are intercepted and wrapped cleanly into `EngineException` (e.g. `TaskValidationError`, `ModelResolutionError`, `PipelineExecutionError`), ensuring raw internal stack traces are never exposed to API consumers.

---

## 🛠️ Usage Example

```python
from visionforge.ai.types import TaskType
from visionforge.engine import get_vision_engine

engine = get_vision_engine()

# Execute a computer vision operation through the engine
result = await engine.run_task(
    task_type=TaskType.DETECTION,
    payload={"image": "frame_001.png"},
    model_name="yolo-v8-detector",
    device="auto",
    options={"confidence": 0.85},
)

print("Success:", result.success)
print("Execution Time:", result.metadata.execution_time_ms)
```
