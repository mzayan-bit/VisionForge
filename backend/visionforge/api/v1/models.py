"""Model Management REST API Endpoints."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.models.manager import ModelManager, get_model_manager
from visionforge.models.metadata import InstalledModelMetadata
from visionforge.models.storage import StorageStats

router = APIRouter(tags=["Models"])


def _get_manager() -> ModelManager:
    return get_model_manager()


# ─── Response Schemas ────────────────────────────────────────────────


class ModelListData(BaseModel):
    """Response payload for listing installed models."""

    models: list[InstalledModelMetadata] = Field(
        description="List of installed model metadata records"
    )
    total: int = Field(description="Total number of installed models")


class ModelDetailData(BaseModel):
    """Response payload for a single model detail."""

    model: InstalledModelMetadata = Field(
        description="Full model metadata record"
    )


class ModelValidationData(BaseModel):
    """Response payload for model validation results."""

    name: str = Field(description="Validated model name")
    valid: bool = Field(description="Whether the model passed validation")
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal validation warnings"
    )
    errors: list[str] = Field(
        default_factory=list, description="Fatal validation errors"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Model metadata if validation succeeded"
    )


class ManagerStatusData(BaseModel):
    """Response payload for model manager health status."""

    status: str = Field(description="Manager operational status")
    installed_models: int = Field(description="Total installed models")
    storage: StorageStats = Field(description="Storage usage statistics")


# ─── Endpoints ───────────────────────────────────────────────────────


@router.get(
    "/models",
    response_model=APIResponse[ModelListData],
    summary="List installed models",
    description="Returns metadata for all installed VisionForge models.",
)
async def list_models() -> APIResponse[ModelListData]:
    """Return all installed model metadata."""
    mgr = _get_manager()
    installed = mgr.list_installed()
    data = ModelListData(models=installed, total=len(installed))
    return success_response(
        data=data,
        message=f"Retrieved {len(installed)} installed model(s)",
    )


@router.get(
    "/models/status",
    response_model=APIResponse[ManagerStatusData],
    summary="Get model manager status",
    description="Returns model manager health, storage stats, and model count.",
)
async def get_manager_status() -> APIResponse[ManagerStatusData]:
    """Return model manager operational status and storage statistics."""
    mgr = _get_manager()
    status = mgr.get_manager_status()
    data = ManagerStatusData(**status)
    return success_response(
        data=data,
        message="Model manager status retrieved",
    )


@router.get(
    "/models/storage",
    response_model=APIResponse[StorageStats],
    summary="Get storage usage",
    description="Returns detailed storage usage statistics for the model directory.",
)
async def get_storage_usage() -> APIResponse[StorageStats]:
    """Return model storage disk usage statistics."""
    mgr = _get_manager()
    stats = mgr.storage.get_storage_stats()
    return success_response(
        data=stats,
        message="Storage usage statistics retrieved",
    )


@router.get(
    "/models/{model_name}",
    response_model=APIResponse[ModelDetailData],
    summary="Get model details",
    description="Returns full metadata for a specific installed model.",
)
async def get_model_detail(model_name: str) -> APIResponse[ModelDetailData]:
    """Return detailed metadata for a specific installed model."""
    mgr = _get_manager()
    meta = mgr.get_model(model_name)
    data = ModelDetailData(model=meta)
    return success_response(
        data=data,
        message=f"Model '{model_name}' details retrieved",
    )


@router.post(
    "/models/{model_name}/validate",
    response_model=APIResponse[ModelValidationData],
    summary="Validate installed model",
    description="Validates model metadata integrity and directory structure.",
)
async def validate_model(
    model_name: str,
) -> APIResponse[ModelValidationData]:
    """Validate an installed model's metadata and directory integrity."""
    mgr = _get_manager()
    report = mgr.validate_model(model_name)
    data = ModelValidationData(**report)
    return success_response(
        data=data,
        message=f"Validation completed for model '{model_name}'",
    )
