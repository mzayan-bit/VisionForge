"""Image Embeddings REST API Endpoints."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from visionforge.ai.registry import get_model_registry
from visionforge.ai.schemas_embedding import ImageEmbeddingResult
from visionforge.ai.types import ModelMetadata, TaskType
from visionforge.core.responses import APIResponse, success_response
from visionforge.engine.runner import get_vision_engine

logger = logging.getLogger("visionforge.api.v1.embeddings")

router = APIRouter(tags=["Embeddings"])


class EmbeddingModelInfoData(BaseModel):
    """Response payload for embedding model metadata and lifecycle status."""

    name: str = Field(description="Model identifier name")
    version: str = Field(description="Semantic version")
    task: str = Field(description="Primary vision task type")
    status: str = Field(
        description="Lifecycle execution state (e.g. ready, uninitialized, unloaded)"
    )
    device: str = Field(description="Active compute device target")
    dimension: int = Field(default=768, description="Output embedding vector dimension")
    metadata: ModelMetadata = Field(description="Full model metadata record")


class ModelLifecycleActionData(BaseModel):
    """Response payload for model load/unload lifecycle actions."""

    name: str = Field(description="Model name")
    status: str = Field(description="New model operational status")
    message: str = Field(description="Action summary message")


@router.post(
    "/embeddings/generate",
    response_model=APIResponse[ImageEmbeddingResult],
    summary="Generate Image Embedding Vector",
    description=(
        "Processes an uploaded image through the Image Embedding Pipeline and returns "
        "a 768-dim normalized L2 vector with metadata."
    ),
)
async def generate_embedding(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
) -> APIResponse[ImageEmbeddingResult]:
    """Generate normalized L2 image embedding vector for provided image file."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file content-type '{file.content_type}'. Must be an image file.",
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        engine = get_vision_engine()
        result = await engine.run_task(
            task_type=TaskType.RETRIEVAL,
            payload=image_bytes,
            model_name="siglip-base-patch16-224",
        )

        if not result.success or result.data is None:
            err_msg = result.error.message if result.error else "Embedding generation failed"
            raise HTTPException(status_code=500, detail=err_msg)

        embedding_data: ImageEmbeddingResult = result.data
        return success_response(
            data=embedding_data,
            message="Image embedding vector generated successfully",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error generating embedding: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Embedding pipeline failure: {str(exc)}"
        ) from exc


@router.get(
    "/embeddings/model-info",
    response_model=APIResponse[EmbeddingModelInfoData],
    summary="Get Embedding Model Information",
    description="Returns operational state, dimension, memory, and metadata for embedding model.",
)
async def get_embedding_model_info() -> APIResponse[EmbeddingModelInfoData]:
    """Return operational metadata and status for the primary embedding model."""
    registry = get_model_registry()
    model = registry.get("siglip-base-patch16-224")

    data = EmbeddingModelInfoData(
        name=model.metadata.name,
        version=model.metadata.version,
        task=model.metadata.task.value,
        status=model.status.value,
        device=model.device,
        dimension=768,
        metadata=model.metadata,
    )

    return success_response(
        data=data,
        message="Embedding model information retrieved",
    )


@router.post(
    "/embeddings/model/load",
    response_model=APIResponse[ModelLifecycleActionData],
    summary="Load Embedding Model Weights",
    description="Loads model weights into target compute memory (CPU/MPS/CUDA).",
)
async def load_embedding_model(device: str = "auto") -> APIResponse[ModelLifecycleActionData]:
    """Load model weights into memory."""
    registry = get_model_registry()
    model = registry.get("siglip-base-patch16-224")

    await model.load(device=device)

    data = ModelLifecycleActionData(
        name=model.metadata.name,
        status=model.status.value,
        message=f"Model loaded successfully onto device '{model.device}'",
    )

    return success_response(data=data, message="Model load completed")


@router.post(
    "/embeddings/model/unload",
    response_model=APIResponse[ModelLifecycleActionData],
    summary="Unload Embedding Model Weights",
    description="Unloads model weights from compute device memory and releases RAM/VRAM.",
)
async def unload_embedding_model() -> APIResponse[ModelLifecycleActionData]:
    """Unload model weights from memory."""
    registry = get_model_registry()
    model = registry.get("siglip-base-patch16-224")

    await model.unload()

    data = ModelLifecycleActionData(
        name=model.metadata.name,
        status=model.status.value,
        message="Model unloaded successfully from compute memory",
    )

    return success_response(data=data, message="Model unload completed")
