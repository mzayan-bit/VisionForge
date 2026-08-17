"""VisionForge Model Lifecycle & End-to-End Pipeline API Routes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status

from visionforge.core.responses import APIResponse, success_response
from visionforge.models.lifecycle_schemas import (
    AdvancePipelineRequest,
    CreatePipelineRequest,
    DeployModelRequest,
    ModelLifecyclePipeline,
    PipelineLineageNode,
)
from visionforge.models.lifecycle_service import (
    PipelineNotFoundError,
    get_model_lifecycle_service,
)

logger = logging.getLogger("visionforge.api.v1.lifecycle")

router = APIRouter(prefix="/lifecycle", tags=["Model Lifecycle & Pipeline"])


def _get_service():
    return get_model_lifecycle_service()


@router.post(
    "/pipelines",
    response_model=APIResponse[ModelLifecyclePipeline],
    status_code=status.HTTP_201_CREATED,
    summary="Create & Initiate Model Lifecycle Pipeline",
)
def create_model_lifecycle_pipeline(
    payload: CreatePipelineRequest,
) -> APIResponse[ModelLifecyclePipeline]:
    """Create and initiate a full 9-step experiment-to-deployment pipeline."""
    service = _get_service()
    pipeline = service.create_pipeline(payload)
    return success_response(
        data=pipeline,
        message=f"Created lifecycle pipeline '{pipeline.pipeline_id}' (Status: {pipeline.status.value})",
    )


@router.get(
    "/pipelines",
    response_model=APIResponse[list[ModelLifecyclePipeline]],
    summary="List Model Lifecycle Pipelines",
)
def list_model_lifecycle_pipelines(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[ModelLifecyclePipeline]]:
    """Retrieve paginated list of historical model lifecycle pipelines."""
    service = _get_service()
    pipelines = service.list_pipelines(limit=limit, offset=offset)
    return success_response(
        data=pipelines, message=f"Retrieved {len(pipelines)} model lifecycle pipeline(s)"
    )


@router.get(
    "/pipelines/{pipeline_id}",
    response_model=APIResponse[ModelLifecyclePipeline],
    summary="Get Single Model Lifecycle Pipeline",
)
def get_model_lifecycle_pipeline(pipeline_id: str) -> APIResponse[ModelLifecyclePipeline]:
    """Retrieve detailed stage-by-stage execution state and artifacts for a pipeline."""
    service = _get_service()
    try:
        pipeline = service.get_pipeline(pipeline_id)
        return success_response(data=pipeline, message=f"Retrieved pipeline '{pipeline_id}'")
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/pipelines/{pipeline_id}/advance",
    response_model=APIResponse[ModelLifecyclePipeline],
    summary="Advance Pipeline to Next Stage",
)
def advance_pipeline_stage(
    pipeline_id: str, payload: AdvancePipelineRequest | None = None
) -> APIResponse[ModelLifecyclePipeline]:
    """Advance a model lifecycle pipeline to the next stage."""
    service = _get_service()
    try:
        target_st = payload.target_stage if payload else None
        pipeline = service.advance_pipeline(pipeline_id, target_st)
        return success_response(
            data=pipeline, message=f"Pipeline advanced to stage '{pipeline.current_stage.value}'"
        )
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/pipelines/{pipeline_id}/deploy",
    response_model=APIResponse[ModelLifecyclePipeline],
    summary="Deploy Verified Model to Inference Runtime",
)
def deploy_pipeline_model(
    pipeline_id: str, payload: DeployModelRequest
) -> APIResponse[ModelLifecyclePipeline]:
    """Deploy model checkpoint from a completed pipeline into active Vision Engine runtime."""
    service = _get_service()
    try:
        pipeline = service.deploy_pipeline_model(pipeline_id, payload)
        return success_response(
            data=pipeline,
            message=f"Model '{pipeline.target_model_name}' successfully deployed to runtime!",
        )
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/pipelines/{pipeline_id}/lineage",
    response_model=APIResponse[list[PipelineLineageNode]],
    summary="Get 9-Stage Lineage & Provenance DAG",
)
def get_pipeline_lineage(pipeline_id: str) -> APIResponse[list[PipelineLineageNode]]:
    """Retrieve full artifact and stage provenance DAG for auditing and reproducibility."""
    service = _get_service()
    try:
        nodes = service.get_pipeline_lineage(pipeline_id)
        return success_response(
            data=nodes,
            message=f"Retrieved {len(nodes)} lineage node(s) for pipeline '{pipeline_id}'",
        )
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
