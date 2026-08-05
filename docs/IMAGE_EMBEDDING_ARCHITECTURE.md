# VisionForge Image Embedding Subsystem Architecture

VisionForge implements an industrial-grade **Image Embedding Pipeline** to generate dense, 768-dimensional L2-normalized vector representations for input images. The architecture is decoupled, supporting model swapping, hardware acceleration (CPU, Apple Silicon `mps`, CUDA), memory management, and future vector index/database integrations.

---

## 1. Selected Model: Google SigLIP (`google/siglip-base-patch16-224`)

### Why SigLIP was Chosen
After evaluating current open-source vision encoders (CLIP, SigLIP, EVA-CLIP, OpenCLIP):

1. **Pairwise Sigmoid Loss vs. Softmax:** SigLIP (Sigmoid Loss for Language Image Pre-Training) replaces the global softmax loss used in classical OpenAI CLIP with a pairwise sigmoid loss. This optimizes memory during pre-training and produces significantly higher zero-shot image-text retrieval quality and classification performance at the exact same parameter size (~300MB weights).
2. **Compact Hardware Footprint:** Base ViT-B/16 architecture consumes under 512MB VRAM / 1GB RAM, making it fast and lightweight for CPU, Apple Silicon (`mps`), and CUDA backends.
3. **Open License:** Released under the Apache-2.0 open-source license.
4. **Normalized 768-Dimensional Output:** Provides high-capacity 768D embeddings that map cleanly into modern vector indices (e.g., FAISS, Qdrant, Milvus, pgvector).

---

## 2. Multi-Stage Execution Pipeline

Every image processed through VisionForge runs through an ordered, multi-stage `EmbeddingPipeline`:

```text
Image Input (File Bytes / PIL)
        │
        ▼
Stage 1: ImageValidationStage   ──► Check format (JPEG, PNG, WebP), dimensions, corruption
        │
        ▼
Stage 2: ImagePreprocessingStage ──► Convert to RGB, transform to 224x224 tensor
        │
        ▼
Stage 3: EmbeddingGenerationStage──► Vision transformer feature extraction (torch.no_grad())
        │
        ▼
Stage 4: L2 Normalization       ──► Unit vector norm: v = v / ||v||_2
        │
        ▼
Stage 5: Metadata & Statistics  ──► Record timings, vector stats (min/max/mean/std), image meta
        │
        ▼
Standardized ImageEmbeddingResult Payload
```

---

## 3. Memory & Lifecycle Management

`SigLIPEmbeddingModel` implements `BaseVisionModel` with lifecycle execution controls:

- **Lazy Loading:** Model metadata initializes without loading PyTorch weights into memory. Weights are loaded on-demand upon first inference request or explicit `/api/v1/embeddings/model/load` call.
- **Explicit Memory Cleanup:** Invoking `/api/v1/embeddings/model/unload` or `model.unload()` releases PyTorch tensors, invokes `gc.collect()`, and flushes `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`.
- **Reloading:** `model.reload(device)` allows zero-downtime device target swapping (e.g., CPU $\rightarrow$ MPS).

---

## 4. Reusable Embedding Object Schema (`ImageEmbeddingResult`)

```json
{
  "embedding": [0.0341, -0.0128, "...", 0.0892],
  "dimension": 768,
  "model": "siglip-base-patch16-224",
  "version": "1.0.0",
  "timestamp": "2026-08-05T16:30:00Z",
  "execution_time_ms": 18.4,
  "loading_time_ms": 150.2,
  "device_used": "cpu",
  "l2_norm": 1.0000,
  "image_metadata": {
    "width": 224,
    "height": 224,
    "format": "JPEG",
    "mode": "RGB",
    "aspect_ratio": 1.0,
    "file_size_bytes": 46080
  },
  "vector_stats": {
    "min": -0.1245,
    "max": 0.1892,
    "mean": 0.0012,
    "std": 0.0361,
    "non_zero_count": 768
  }
}
```

---

## 5. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/embeddings/generate` | Accepts multipart image file, executes embedding pipeline, returns 768D vector & stats |
| `GET` | `/api/v1/embeddings/model-info` | Returns operational status, dimension, device, and memory metadata |
| `POST` | `/api/v1/embeddings/model/load` | Explicitly loads model weights onto target device |
| `POST` | `/api/v1/embeddings/model/unload` | Unloads model weights from RAM/VRAM |

---

## 6. Future Extension Strategy

- **Model Swapping:** Developers can register alternative vision backends (e.g., `openai/clip-vit-large-patch14`, `EVA-02`, `DINOv2`) into `ModelRegistry` by implementing `BaseVisionModel`.
- **Vector Database Integration:** Output `ImageEmbeddingResult` objects are pre-normalized (`||v||_2 = 1.0`), making cosine similarity equivalent to dot product $u \cdot v$ for high-speed indexing in HNSW or IVF vector tables.
