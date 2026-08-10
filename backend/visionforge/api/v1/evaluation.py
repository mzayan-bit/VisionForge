"""API Endpoints for Model Evaluation and Benchmarking."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visionforge.evaluation.schemas import (
    BenchmarkRun,
    ErrorPrediction,
    EvaluationConfig,
    EvaluationRun,
)
from visionforge.evaluation.service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
eval_service = EvaluationService()


class CreateEvaluationRequest(BaseModel):
    model_name: str
    checkpoint_path: str
    dataset_id: str
    dataset_version: str
    dataset_yaml: str
    config: EvaluationConfig = EvaluationConfig()
    preparation_id: str | None = None
    training_run_id: str | None = None
    split_used: str = "test"


class CreateBenchmarkRequest(BaseModel):
    eval_ids: list[str]


@router.post("/runs", response_model=EvaluationRun)
async def create_evaluation(req: CreateEvaluationRequest):
    """Trigger a new model evaluation on a dataset."""
    try:
        return eval_service.create_evaluation(
            model_name=req.model_name,
            checkpoint_path=Path(req.checkpoint_path),
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
            dataset_yaml=Path(req.dataset_yaml),
            config=req.config,
            preparation_id=req.preparation_id,
            training_run_id=req.training_run_id,
            split_used=req.split_used,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs", response_model=list[EvaluationRun])
async def list_evaluations():
    """List all evaluation runs."""
    return eval_service.list_evaluations()


@router.get("/runs/{eval_id}", response_model=EvaluationRun)
async def get_evaluation(eval_id: str):
    """Get details of a specific evaluation run."""
    run = eval_service.get_evaluation(eval_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run


@router.get("/runs/{eval_id}/failures", response_model=list[ErrorPrediction])
async def get_evaluation_failures(eval_id: str):
    """Get diagnostic error predictions (failures) for an evaluation run."""
    run = eval_service.get_evaluation(eval_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return eval_service.get_errors(eval_id)


@router.post("/benchmarks", response_model=BenchmarkRun)
async def create_benchmark(req: CreateBenchmarkRequest):
    """Compare multiple models in a benchmark run."""
    try:
        return eval_service.create_benchmark(req.eval_ids)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
