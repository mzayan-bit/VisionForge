"""VisionEngine Dedicated Image Embedding Pipeline Implementation."""

import io
import logging
from typing import Any

from PIL import Image, UnidentifiedImageError

from visionforge.ai.registry import get_model_registry
from visionforge.ai.schemas_embedding import ImageEmbeddingResult
from visionforge.engine.context import ExecutionContext
from visionforge.engine.exceptions import PipelineExecutionError, TaskValidationError
from visionforge.engine.pipeline import EnginePipeline, PipelineStage
from visionforge.engine.task import TaskState

logger = logging.getLogger("visionforge.engine.pipelines.embedding_pipeline")


class ImageValidationStage(PipelineStage):
    """Stage 1: Validate input image bytes or PIL Image instance for integrity and bounds."""

    @property
    def name(self) -> str:
        return "image_validation"

    @property
    def state_on_enter(self) -> TaskState:
        return TaskState.VALIDATING

    async def process(self, context: ExecutionContext, payload: Any) -> Image.Image:
        if payload is None:
            raise TaskValidationError("Input image payload cannot be None")

        if isinstance(payload, bytes):
            try:
                img = Image.open(io.BytesIO(payload))
                img.verify()  # Check for header/file corruption
                # Re-open after verify() since verify() alters file offset
                img = Image.open(io.BytesIO(payload))
            except (UnidentifiedImageError, ValueError) as exc:
                raise TaskValidationError(f"Invalid or corrupted image data: {str(exc)}") from exc
        elif isinstance(payload, Image.Image):
            img = payload
        else:
            raise TaskValidationError(
                f"Unsupported image input type '{type(payload)}'. Expected bytes or PIL Image."
            )

        if img.width <= 0 or img.height <= 0:
            raise TaskValidationError(f"Invalid image dimensions: {img.width}x{img.height}")

        return img


class ImagePreprocessingStage(PipelineStage):
    """Stage 2: Convert image to standardized RGB color space."""

    @property
    def name(self) -> str:
        return "image_preprocessing"

    @property
    def state_on_enter(self) -> TaskState:
        return TaskState.PREPROCESSING

    async def process(self, context: ExecutionContext, payload: Any) -> Image.Image:
        if not isinstance(payload, Image.Image):
            raise PipelineExecutionError("image_preprocessing", "Expected PIL.Image.Image payload")

        return payload.convert("RGB")


class EmbeddingGenerationStage(PipelineStage):
    """Stage 3: Model resolution and feature vector extraction."""

    @property
    def name(self) -> str:
        return "embedding_generation"

    @property
    def state_on_enter(self) -> TaskState:
        return TaskState.EXECUTING

    async def process(self, context: ExecutionContext, payload: Any) -> ImageEmbeddingResult:
        registry = get_model_registry()
        model_name = context.model_name or "siglip-base-patch16-224"

        model = registry.get(model_name)
        context.model_instance = model

        inference_res = await model.predict(payload, device=context.device, **context.options)
        if not inference_res.success or inference_res.data is None:
            err = inference_res.error.message if inference_res.error else "Prediction failed"
            raise PipelineExecutionError("embedding_generation", err)

        return inference_res.data


class EmbeddingPipeline(EnginePipeline):
    """Industrial-grade, multi-stage Image Embedding Pipeline."""

    def __init__(self):
        super().__init__(
            stages=[
                ImageValidationStage(),
                ImagePreprocessingStage(),
                EmbeddingGenerationStage(),
            ]
        )
