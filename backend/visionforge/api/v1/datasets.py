"""Dataset Preparation Pipeline REST API Endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.datasets.schemas import PreparationRun, SplitConfig, SplitStrategy
from visionforge.datasets.service import (
    DatasetPreparationService,
    get_dataset_preparation_service,
)

router = APIRouter(tags=["Dataset Preparation"])


class CreatePreparationRequest(BaseModel):
    """Request payload to initiate a dataset preparation run."""

    dataset_id: str = Field(default="default_dataset", description="Target dataset identifier")
    dataset_version: str = Field(default="v1.0", description="Target dataset version string")
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0, description="Training set fraction")
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0, description="Validation set fraction")
    test_ratio: float = Field(default=0.15, ge=0.0, lt=1.0, description="Test set fraction")
    random_seed: int = Field(default=42, ge=0, description="Random seed for 100% reproducibility")
    strategy: SplitStrategy = Field(default=SplitStrategy.RANDOM, description="Split strategy")
    group_by_field: str | None = Field(default=None, description="Metadata field for group-aware split")
    stratify_by_field: str | None = Field(default=None, description="Metadata field for stratified split")


def _get_service() -> DatasetPreparationService:
    return get_dataset_preparation_service()


@router.post(
    "/datasets/prepare",
    response_model=APIResponse[PreparationRun],
    summary="Create & Execute Dataset Preparation Run",
    description="Validates dataset, detects data leakage (exact and near-duplicates), and generates reproducible splits.",
)
async def create_preparation_run(
    req: CreatePreparationRequest,
) -> APIResponse[PreparationRun]:
    """Execute dataset preparation run."""
    svc = _get_service()
    config = SplitConfig(
        train_ratio=req.train_ratio,
        val_ratio=req.val_ratio,
        test_ratio=req.test_ratio,
        random_seed=req.random_seed,
        strategy=req.strategy,
        group_by_field=req.group_by_field,
        stratify_by_field=req.stratify_by_field,
    )

    try:
        run = svc.create_preparation_run(
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
            split_config=config,
        )
        msg = f"Dataset preparation '{run.preparation_id}' completed with status {run.status.value}"
        return success_response(data=run, message=msg)
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.get(
    "/datasets/prepare/history",
    response_model=APIResponse[list[PreparationRun]],
    summary="Get Dataset Preparation History",
    description="Returns paginated list of historical dataset preparation runs.",
)
async def list_preparation_history(
    limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
) -> APIResponse[list[PreparationRun]]:
    """Return preparation history runs."""
    svc = _get_service()
    runs = svc.list_history(limit=limit, offset=offset)
    return success_response(data=runs, message=f"Retrieved {len(runs)} preparation run(s)")


@router.get(
    "/datasets/prepare/{prep_id}",
    response_model=APIResponse[PreparationRun],
    summary="Get Dataset Preparation Run Details",
    description="Returns detailed preparation status, validation report, leakage findings, and split stats.",
)
async def get_preparation_run(prep_id: str) -> APIResponse[PreparationRun]:
    """Get preparation run by ID."""
    svc = _get_service()
    try:
        run = svc.get_run(prep_id)
        return success_response(data=run, message=f"Preparation run '{prep_id}' retrieved")
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.get(
    "/datasets/prepare/{prep_id}/manifest",
    summary="Export Prepared Dataset Manifest",
    description="Exports the prepared dataset manifest in machine-readable JSON or CSV format.",
)
async def export_dataset_manifest(
    prep_id: str, format: str = Query("json", enum=["json", "csv"])
) -> Any:
    """Export prepared dataset manifest."""
    svc = _get_service()
    try:
        data = svc.export_manifest(prep_id, fmt=format)
        if format == "csv":
            return PlainTextResponse(content=data, media_type="text/csv")
        return success_response(data=data, message=f"Manifest '{prep_id}' exported successfully")
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc
