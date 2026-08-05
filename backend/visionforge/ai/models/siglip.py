"""VisionForge SigLIP Image Embedding Model Implementation."""

import gc
import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
from PIL import Image

from visionforge.ai.base import BaseVisionModel
from visionforge.ai.schemas import ExecutionMetadata, InferenceResult
from visionforge.ai.schemas_embedding import ImageEmbeddingResult, ImageMetadata, VectorStats
from visionforge.ai.types import (
    InputType,
    MemoryRequirements,
    ModelMetadata,
    ModelStatus,
    OutputType,
    TaskType,
)
from visionforge.core.exceptions import VisionForgeException

logger = logging.getLogger("visionforge.ai.models.siglip")

# Default Hugging Face repo for SigLIP base model
SIGLIP_HF_MODEL_ID = "google/siglip-base-patch16-224"


class ModelLoadError(VisionForgeException):
    """Raised when failing to load vision model weights into compute memory."""

    def __init__(self, message: str):
        super().__init__(message=message, code="MODEL_LOAD_ERROR", status_code=500)


class SigLIPEmbeddingModel(BaseVisionModel):
    """Production-grade SigLIP (Sigmoid Loss for Language Image Pre-Training) Embedding Model.

    Encapsulates lazy loading, device binding (CPU, MPS, CUDA), memory cleanup,
    image feature extraction, and 768-dimensional L2-normalized vector output generation.
    """

    def __init__(self, model_id: str = SIGLIP_HF_MODEL_ID):
        super().__init__()
        self._model_id = model_id
        self._model: Any = None
        self._processor: Any = None
        self._loading_time_ms: float = 0.0

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="siglip-base-patch16-224",
            version="1.0.0",
            author="Google / VisionForge",
            task=TaskType.RETRIEVAL,
            license="Apache-2.0",
            supported_input_types=[InputType.IMAGE],
            supported_output_types=[OutputType.EMBEDDINGS],
            memory_requirements=MemoryRequirements(
                vram_mb=512,
                ram_mb=1024,
                disk_space_mb=350,
            ),
            device_support=["cpu", "cuda", "mps"],
            description=(
                "Google SigLIP Base Patch16 224px Vision Transformer. Replaces Softmax "
                "with pairwise Sigmoid loss for state-of-the-art image embedding quality."
            ),
            status=self.status,
        )

    async def initialize(self) -> None:
        """Initialize configuration state. Lightweight operation without loading weights."""
        if self._status == ModelStatus.UNINITIALIZED:
            self._status = ModelStatus.INITIALIZED
            logger.info("Initialized SigLIP model metadata specification '%s'", self._model_id)

    async def load(self, device: str | None = None) -> None:
        """Load model weights and processor into target compute memory."""
        start_time = time.perf_counter()
        self._status = ModelStatus.LOADING

        target_device = device or self._device
        if target_device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS device requested but not available. Falling back to CPU.")
            target_device = "cpu"
        elif target_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            target_device = "cpu"

        self._device = target_device

        try:
            from transformers import AutoImageProcessor, SiglipModel

            logger.info(
                "Loading SigLIP model '%s' onto device '%s'...", self._model_id, self._device
            )
            self._processor = AutoImageProcessor.from_pretrained(self._model_id)
            self._model = SiglipModel.from_pretrained(self._model_id).to(self._device)
            self._model.eval()

            self._loading_time_ms = (time.perf_counter() - start_time) * 1000
            self._status = ModelStatus.READY
            logger.info(
                "Successfully loaded SigLIP model onto %s in %.2fms",
                self._device,
                self._loading_time_ms,
            )

        except Exception as exc:
            self._status = ModelStatus.FAILED
            logger.error("Failed to load SigLIP model '%s': %s", self._model_id, str(exc))
            raise ModelLoadError(f"Failed to load SigLIP model weights: {str(exc)}") from exc

    async def unload(self) -> None:
        """Unload model weights and release memory resources."""
        if self._status in (ModelStatus.READY, ModelStatus.LOADING):
            logger.info("Unloading SigLIP model from device '%s'...", self._device)
            self._model = None
            self._processor = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            self._status = ModelStatus.UNLOADED
            logger.info("Successfully unloaded SigLIP model and cleared VRAM/RAM.")

    async def reload(self, device: str | None = None) -> None:
        """Safely reload model weights into memory."""
        await self.unload()
        await self.load(device=device)

    async def predict(self, inputs: Any, **kwargs: Any) -> InferenceResult[ImageEmbeddingResult]:
        """Execute image embedding generation pipeline.

        Inputs:
            inputs: PIL.Image.Image or bytes or PyTorch Tensor or NumPy array.
        """
        import io

        start_time = time.perf_counter()

        if self._status != ModelStatus.READY:
            await self.load(device=self._device)

        # 1. Image Modality Parsing & Metadata Extraction
        if isinstance(inputs, bytes):
            pil_image = Image.open(io.BytesIO(inputs)).convert("RGB")
        elif isinstance(inputs, Image.Image):
            pil_image = inputs.convert("RGB")
        elif isinstance(inputs, np.ndarray):
            pil_image = Image.fromarray(inputs).convert("RGB")
        else:
            raise ValueError(f"Unsupported input image payload type: {type(inputs)}")

        width, height = pil_image.size
        aspect_ratio = round(width / height, 4) if height > 0 else 1.0

        image_meta = ImageMetadata(
            width=width,
            height=height,
            format=pil_image.format or "RGB",
            mode=pil_image.mode,
            aspect_ratio=aspect_ratio,
            file_size_bytes=len(pil_image.tobytes()),
        )

        # 2. Preprocessing & Tensor Encoding
        inputs_processed = self._processor(images=pil_image, return_tensors="pt").to(self._device)

        # 3. Model Inference Execution
        with torch.no_grad():
            if hasattr(self._model, "get_image_features"):
                features = self._model.get_image_features(
                    pixel_values=inputs_processed["pixel_values"]
                )
                if isinstance(features, torch.Tensor):
                    embedding_tensor = features.flatten()
                elif hasattr(features, "image_embeds") and features.image_embeds is not None:
                    embedding_tensor = features.image_embeds.flatten()
                elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                    embedding_tensor = features.pooler_output.flatten()
                else:
                    embedding_tensor = features[0].flatten()
            else:
                outputs = self._model(**inputs_processed)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embedding_tensor = outputs.pooler_output.flatten()
                else:
                    embedding_tensor = outputs.last_hidden_state[0].mean(dim=0).flatten()

            # 4. L2 Normalization (v / ||v||_2)
            norm = torch.linalg.norm(embedding_tensor, ord=2)
            if norm > 0:
                normalized_tensor = embedding_tensor / norm
            else:
                normalized_tensor = embedding_tensor

            vector = normalized_tensor.cpu().numpy().astype(float).tolist()
            l2_norm_val = float(torch.linalg.norm(normalized_tensor, ord=2).item())

        exec_time_ms = (time.perf_counter() - start_time) * 1000

        # 5. Vector Statistics
        vec_np = np.array(vector)
        vec_stats = VectorStats(
            min=float(np.min(vec_np)),
            max=float(np.max(vec_np)),
            mean=float(np.mean(vec_np)),
            std=float(np.std(vec_np)),
            non_zero_count=int(np.count_nonzero(vec_np)),
        )

        embedding_res = ImageEmbeddingResult(
            embedding=vector,
            dimension=len(vector),
            model=self.metadata.name,
            version=self.metadata.version,
            timestamp=datetime.now(UTC).isoformat(),
            execution_time_ms=round(exec_time_ms, 2),
            loading_time_ms=round(self._loading_time_ms, 2),
            device_used=self._device,
            l2_norm=round(l2_norm_val, 4),
            image_metadata=image_meta,
            vector_stats=vec_stats,
            extra_metadata={"model_id": self._model_id, "architecture": "ViT-Base/16"},
        )

        exec_meta = ExecutionMetadata(
            model_name=self.metadata.name,
            model_version=self.metadata.version,
            device_used=self._device,
            execution_time_ms=round(exec_time_ms, 2),
        )

        return InferenceResult(
            success=True,
            message="Image embedding generated successfully",
            data=embedding_res,
            metadata=exec_meta,
        )

    async def cleanup(self) -> None:
        """Perform final resource cleanup on shutdown."""
        await self.unload()
