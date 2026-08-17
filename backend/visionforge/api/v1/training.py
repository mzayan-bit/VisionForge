"""Training Pipeline REST API Endpoints."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.models.metadata import InstalledModelMetadata
from visionforge.training.schemas import (
    EvaluationResult,
    SmokeTestResult,
    TrainingConfig,
    TrainingRun,
)
from visionforge.training.service import (
    TrainingService,
    get_training_service,
)

router = APIRouter(tags=["Training Pipeline"])


class RegisterModelRequest(BaseModel):
    """Payload to register trained model artifact into ModelManager."""

    version_tag: str = Field(default="v1.0.0", description="Target model version string")


class PredictSmokeTestRequest(BaseModel):
    """Payload to trigger inference smoke test."""

    sample_image_paths: list[str] = Field(
        default_factory=list, description="Optional paths to test images"
    )


def _get_service() -> TrainingService:
    return get_training_service()


@router.post(
    "/training/runs",
    response_model=APIResponse[TrainingRun],
    summary="Create & Execute Training Run",
    description="Validates training configuration, prepares dataset using adapter, executes PyTorch YOLO training, and runs test evaluation.",
)
async def create_training_run(config: TrainingConfig) -> APIResponse[TrainingRun]:
    """Execute new model training run."""
    svc = _get_service()
    try:
        run = svc.create_training_run(config)
        msg = f"Training run '{run.run_id}' completed with status {run.status.value}"
        return success_response(data=run, message=msg)
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.get(
    "/training/runs",
    response_model=APIResponse[list[TrainingRun]],
    summary="List Training Runs",
    description="Returns paginated list of historical training runs.",
)
async def list_training_runs(
    limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
) -> APIResponse[list[TrainingRun]]:
    """Return historical training runs."""
    svc = _get_service()
    runs = svc.list_runs(limit=limit, offset=offset)
    return success_response(data=runs, message=f"Retrieved {len(runs)} training run(s)")


@router.get(
    "/training/runs/{run_id}",
    response_model=APIResponse[TrainingRun],
    summary="Get Training Run Details",
    description="Returns detailed training status, metric history, best checkpoint path, and evaluation results.",
)
async def get_training_run(run_id: str) -> APIResponse[TrainingRun]:
    """Get training run by ID."""
    svc = _get_service()
    try:
        run = svc.get_run(run_id)
        return success_response(data=run, message=f"Training run '{run_id}' retrieved")
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post(
    "/training/runs/{run_id}/evaluate",
    response_model=APIResponse[EvaluationResult],
    summary="Execute Separate Test Set Evaluation",
    description="Evaluates the trained checkpoint on the isolated test set partition.",
)
async def evaluate_training_run(run_id: str) -> APIResponse[EvaluationResult]:
    """Execute test evaluation."""
    svc = _get_service()
    try:
        eval_res = svc.evaluate_test_set(run_id)
        return success_response(
            data=eval_res, message=f"Test set evaluation completed for '{run_id}'"
        )
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post(
    "/training/runs/{run_id}/register",
    response_model=APIResponse[InstalledModelMetadata],
    summary="Register Trained Checkpoint in ModelManager",
    description="Registers the trained model checkpoint into ModelManager as a versioned artifact.",
)
async def register_training_model(
    run_id: str, req: RegisterModelRequest
) -> APIResponse[InstalledModelMetadata]:
    """Register model artifact."""
    svc = _get_service()
    try:
        meta = svc.register_model_artifact(run_id, version_tag=req.version_tag)
        return success_response(
            data=meta, message=f"Model registered as '{meta.name}' ({meta.version})"
        )
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post(
    "/training/runs/{run_id}/predict",
    response_model=APIResponse[SmokeTestResult],
    summary="Inference Smoke Test",
    description="Runs lightweight inference using the trained checkpoint on sample test images.",
)
async def predict_smoke_test(
    run_id: str, req: PredictSmokeTestRequest
) -> APIResponse[SmokeTestResult]:
    """Run inference smoke test."""
    svc = _get_service()
    try:
        result = svc.run_inference_smoke_test(run_id, sample_image_paths=req.sample_image_paths)
        return success_response(
            data=result, message=f"Inference smoke test completed for '{run_id}'"
        )
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc
