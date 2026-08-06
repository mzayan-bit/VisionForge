"""Visual Memory REST API Endpoints."""

import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from visionforge.ai.schemas_embedding import ImageEmbeddingResult
from visionforge.ai.types import TaskType
from visionforge.core.responses import APIResponse, success_response
from visionforge.engine.runner import get_vision_engine
from visionforge.memory.index import (
    VisualMemoryIndex,
    VisualMemoryRecord,
    VisualMemoryStats,
    get_visual_memory_index,
)

router = APIRouter(tags=["Visual Memory"])


def _get_memory_index() -> VisualMemoryIndex:
    return get_visual_memory_index()


@router.post(
    "/memory/index",
    response_model=APIResponse[VisualMemoryRecord],
    summary="Index Image into Visual Memory",
    description=(
        "Processes an uploaded image through SigLIP embedding pipeline and indexes "
        "the resulting vector and metadata into Visual Memory."
    ),
)
async def index_image_into_memory(
    file: UploadFile = File(..., description="Image file to index"),
    record_id: str | None = Form(None, description="Optional custom record ID"),
    tags: str | None = Form(None, description="Comma-separated tags (e.g. 'landscape,nature')"),
) -> APIResponse[VisualMemoryRecord]:
    """Generate embedding for uploaded image and store in Visual Memory Index."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Generate image embedding vector
    engine = get_vision_engine()
    res = await engine.run_task(
        task_type=TaskType.RETRIEVAL,
        payload=image_bytes,
        model_name="siglip-base-patch16-224",
    )

    if not res.success or res.data is None:
        err = res.error.message if res.error else "Failed to generate image embedding"
        raise HTTPException(status_code=500, detail=err)

    embedding_data: ImageEmbeddingResult = res.data

    # 2. Construct record
    uid = record_id or f"mem_{uuid.uuid4().hex[:10]}"
    parsed_tags = [t.strip() for t in tags.split(",")] if tags else []

    record = VisualMemoryRecord(
        id=uid,
        embedding=embedding_data.embedding,
        dimension=embedding_data.dimension,
        image_metadata=embedding_data.image_metadata.model_dump(),
        tags=parsed_tags,
    )

    # 3. Store record & save to disk
    mem = _get_memory_index()
    mem.add_record(record)
    mem.save_to_disk()

    return success_response(data=record, message=f"Image successfully indexed as '{uid}'")


@router.get(
    "/memory/stats",
    response_model=APIResponse[VisualMemoryStats],
    summary="Get Visual Memory Statistics",
    description="Returns total indexed vectors, memory footprint, and disk persistence telemetry.",
)
async def get_memory_stats() -> APIResponse[VisualMemoryStats]:
    """Return visual memory telemetry and statistics."""
    mem = _get_memory_index()
    stats = mem.get_stats()
    return success_response(data=stats, message="Visual memory statistics retrieved")


@router.get(
    "/memory/records",
    response_model=APIResponse[list[VisualMemoryRecord]],
    summary="List Visual Memory Records",
    description="Returns paginated list of indexed visual memory records.",
)
async def list_memory_records(
    limit: int = 50, offset: int = 0
) -> APIResponse[list[VisualMemoryRecord]]:
    """Return paginated list of indexed records."""
    mem = _get_memory_index()
    records = mem.list_records(limit=limit, offset=offset)
    return success_response(
        data=records, message=f"Retrieved {len(records)} record(s) from Visual Memory"
    )


@router.delete(
    "/memory/clear",
    response_model=APIResponse[dict[str, Any]],
    summary="Clear Visual Memory Store",
    description="Purges all indexed vectors and clears disk index.",
)
async def clear_visual_memory() -> APIResponse[dict[str, Any]]:
    """Purge all records from visual memory index."""
    mem = _get_memory_index()
    count = mem.clear()
    mem.save_to_disk()
    return success_response(
        data={"purged_count": count},
        message=f"Cleared {count} record(s) from Visual Memory",
    )
