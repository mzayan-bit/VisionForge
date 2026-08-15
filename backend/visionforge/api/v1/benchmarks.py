"""REST API Endpoints for VisionForge Research Benchmark Lab."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.evaluation.schemas import (
    BenchmarkHistoryItem,
    BenchmarkRun,
    ErrorCategory,
    ErrorPrediction,
    EvaluationConfig,
    ModelComparisonResult,
)
from visionforge.evaluation.service import get_evaluation_service

router = APIRouter(prefix="/benchmarks", tags=["Research Benchmarks"])


class CreateBenchmarkRunRequest(BaseModel):
    """Payload to create and execute a research benchmark."""

    name: str = Field(description="Display name of the benchmark")
    model_name: str = Field(description="Model identifier to evaluate")
    model_version: str = Field(default="1.0.0", description="Model version string")
    dataset_id: str = Field(description="Dataset identifier")
    dataset_version: str = Field(default="v1.0.0", description="Dataset version string")
    dataset_fingerprint: str = Field(
        default="sha256_mock_fingerprint", description="Cryptographic dataset hash"
    )
    split_used: str = Field(default="test", description="Dataset split evaluated")
    task: str = Field(default="OBJECT_DETECTION", description="Computer vision task")
    is_baseline: bool = Field(default=False, description="Whether this run serves as baseline")
    baseline_benchmark_id: str | None = Field(default=None, description="Optional baseline reference ID")
    description: str = Field(default="", description="Benchmark objective notes")
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)
    class_names: list[str] | None = None
    ground_truths: dict[str, list[dict[str, Any]]] | None = None
    predictions: dict[str, list[dict[str, Any]]] | None = None


class CompareBenchmarksRequest(BaseModel):
    """Payload to compare candidate benchmark against baseline."""

    baseline_id: str = Field(description="Baseline BenchmarkRun ID")
    candidate_id: str = Field(description="Candidate BenchmarkRun ID")
    regression_threshold_map50: float = Field(
        default=0.02, ge=0.001, le=0.20, description="Tolerance threshold for mAP regression"
    )
    regression_threshold_latency: float = Field(
        default=0.10, ge=0.01, le=0.50, description="Tolerance threshold for latency increase"
    )


def _get_service():
    return get_evaluation_service()


@router.get(
    "/runs",
    response_model=APIResponse[list[BenchmarkRun]],
    summary="List Research Benchmarks",
    description="Lists all persisted benchmark runs with optional filtering.",
)
def list_benchmarks(
    dataset_id: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    is_baseline: bool | None = Query(default=None),
    task: str | None = Query(default=None),
) -> APIResponse[list[BenchmarkRun]]:
    """List benchmark runs."""
    svc = _get_service()
    runs = svc.list_benchmarks(
        dataset_id=dataset_id,
        model_name=model_name,
        is_baseline=is_baseline,
        task=task,
    )
    return success_response(data=runs, message=f"Retrieved {len(runs)} benchmark run(s)")


@router.post(
    "/runs",
    response_model=APIResponse[BenchmarkRun],
    summary="Create Research Benchmark Run",
    description="Executes accuracy evaluation, steady-state runtime profiling, and error analysis.",
)
def create_benchmark_run(
    req: CreateBenchmarkRunRequest,
) -> APIResponse[BenchmarkRun]:
    """Execute new benchmark run."""
    svc = _get_service()
    try:
        run = svc.create_benchmark_run(
            name=req.name,
            model_name=req.model_name,
            model_version=req.model_version,
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
            dataset_fingerprint=req.dataset_fingerprint,
            split_used=req.split_used,
            task=req.task,
            config=req.config,
            is_baseline=req.is_baseline,
            baseline_benchmark_id=req.baseline_benchmark_id,
            description=req.description,
            ground_truths_by_image=req.ground_truths,
            predictions_by_image=req.predictions,
            class_names=req.class_names,
        )
        return success_response(
            data=run, message=f"Benchmark '{run.benchmark_id}' completed with mAP@50:95 of {run.metrics.map50_95:.2%}"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/runs/{benchmark_id}",
    response_model=APIResponse[BenchmarkRun],
    summary="Get Benchmark Details",
    description="Returns complete metrics, per-class breakdown, threshold points, runtime profile, and reproducibility metadata.",
)
def get_benchmark(benchmark_id: str) -> APIResponse[BenchmarkRun]:
    """Retrieve benchmark run details."""
    svc = _get_service()
    run = svc.get_benchmark(benchmark_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_id}' not found")
    return success_response(data=run)


@router.get(
    "/runs/{benchmark_id}/report",
    summary="Get Benchmark Markdown Report",
    description="Generates a structured scientific research report in Markdown.",
)
def get_benchmark_report(benchmark_id: str):
    """Generate Markdown research report."""
    svc = _get_service()
    try:
        report_md = svc.generate_benchmark_report(benchmark_id)
        return Response(content=report_md, media_type="text/markdown")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/runs/{benchmark_id}/failures",
    response_model=APIResponse[list[ErrorPrediction]],
    summary="Get Benchmark Diagnostic Failures",
    description="Returns diagnostic error predictions categorized into failure taxonomy.",
)
def get_benchmark_failures(
    benchmark_id: str,
    error_type: ErrorCategory | None = Query(default=None),
    class_name: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[ErrorPrediction]]:
    """Retrieve categorized failure predictions."""
    svc = _get_service()
    failures = svc.get_errors(
        id_ref=benchmark_id,
        error_type=error_type,
        class_name=class_name,
        limit=limit,
        offset=offset,
    )
    return success_response(data=failures, message=f"Retrieved {len(failures)} failure prediction(s)")


@router.post(
    "/compare",
    response_model=APIResponse[ModelComparisonResult],
    summary="Compare Baseline vs Candidate Models",
    description="Controlled comparison verifying matching dataset/split/protocol, computing deltas, and checking for regressions.",
)
def compare_benchmarks(
    req: CompareBenchmarksRequest,
) -> APIResponse[ModelComparisonResult]:
    """Execute controlled model comparison."""
    svc = _get_service()
    try:
        cmp_res = svc.compare_benchmarks(
            baseline_id=req.baseline_id,
            candidate_id=req.candidate_id,
            regression_threshold_map50=req.regression_threshold_map50,
            regression_threshold_latency=req.regression_threshold_latency,
        )
        return success_response(
            data=cmp_res, message=f"Model comparison status: {cmp_res.regression_status.value}"
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/history",
    response_model=APIResponse[list[BenchmarkHistoryItem]],
    summary="Get Benchmark Progression History",
    description="Returns longitudinal timeline of model metric progression.",
)
def get_benchmark_history(
    dataset_id: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
) -> APIResponse[list[BenchmarkHistoryItem]]:
    """Retrieve historical progression timeline."""
    svc = _get_service()
    history = svc.get_benchmark_history(dataset_id=dataset_id, model_name=model_name)
    return success_response(data=history, message=f"Retrieved {len(history)} progression point(s)")
