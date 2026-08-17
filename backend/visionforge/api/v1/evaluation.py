"""API Endpoints for Model Evaluation, Error Analysis, Failure Gallery, & Model Comparison."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.evaluation.schemas import (
    ConfidenceDistributions,
    ConfusionMatrixData,
    ErrorCategory,
    EvaluationConfig,
    EvaluationRun,
    FailureSampleDetail,
    ModelComparisonResult,
    PatternAnalysisReport,
    PRCurveData,
    ThresholdPoint,
    VisualFailureCluster,
)
from visionforge.evaluation.service import get_evaluation_service

router = APIRouter(prefix="/evaluation", tags=["Model Evaluation & Error Analysis"])


def _get_service():
    return get_evaluation_service()


class CreateEvaluationRequest(BaseModel):
    model_name: str
    checkpoint_path: str = "yolo11s.pt"
    dataset_id: str = "safety_v2"
    dataset_version: str = "v1.0.0"
    dataset_yaml: str = "dataset.yaml"
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)
    preparation_id: str | None = None
    training_run_id: str | None = None
    split_used: str = "test"


class CompareModelsRequest(BaseModel):
    baseline_eval_id: str = Field(description="Baseline evaluation ID (M0)")
    candidate_eval_id: str = Field(description="Candidate evaluation ID (M1)")
    regression_threshold_map50: float = Field(default=0.02, ge=0.001, le=0.20)
    regression_threshold_latency: float = Field(default=0.10, ge=0.01, le=0.50)


# ─── Evaluation Run Routes ─────────────────────────────────────────────


@router.post(
    "/runs", response_model=APIResponse[EvaluationRun], status_code=status.HTTP_201_CREATED
)
def create_evaluation(req: CreateEvaluationRequest) -> APIResponse[EvaluationRun]:
    """Trigger a new model evaluation run on an exact dataset version."""
    svc = _get_service()
    try:
        run = svc.create_evaluation(
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
        return success_response(data=run, message=f"Created evaluation run '{run.eval_id}'")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/runs", response_model=APIResponse[list[EvaluationRun]])
def list_evaluations() -> APIResponse[list[EvaluationRun]]:
    """List all completed and running evaluations."""
    svc = _get_service()
    runs = svc.list_evaluations()
    return success_response(data=runs, message=f"Retrieved {len(runs)} evaluation run(s)")


@router.get("/runs/{eval_id}", response_model=APIResponse[EvaluationRun])
def get_evaluation(eval_id: str) -> APIResponse[EvaluationRun]:
    """Get details of a specific evaluation run."""
    svc = _get_service()
    run = svc.get_evaluation(eval_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Evaluation run '{eval_id}' not found"
        )
    return success_response(data=run, message=f"Retrieved evaluation '{eval_id}'")


@router.get("/runs/{eval_id}/metrics", response_model=APIResponse[dict[str, Any]])
def get_evaluation_metrics(eval_id: str) -> APIResponse[dict[str, Any]]:
    """Retrieve global and per-class metrics for an evaluation."""
    svc = _get_service()
    run = svc.get_evaluation(eval_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Evaluation run '{eval_id}' not found"
        )
    return success_response(
        data={
            "eval_id": run.eval_id,
            "precision": run.precision,
            "recall": run.recall,
            "f1": run.f1,
            "map50": run.map50,
            "map75": run.map75,
            "map50_95": run.map50_95,
            "per_class_metrics": run.per_class_metrics,
        },
        message="Retrieved evaluation metrics",
    )


# ─── Deep Error Analysis & Telemetry Routes ────────────────────────────


@router.get("/runs/{eval_id}/thresholds", response_model=APIResponse[list[ThresholdPoint]])
def get_threshold_analysis(eval_id: str) -> APIResponse[list[ThresholdPoint]]:
    """Retrieve performance metrics evaluated across confidence thresholds [0.20..0.80]."""
    svc = _get_service()
    pts = svc.get_threshold_analysis(eval_id)
    return success_response(
        data=pts, message=f"Retrieved {len(pts)} threshold analysis operating points"
    )


@router.get("/runs/{eval_id}/confusion", response_model=APIResponse[ConfusionMatrixData])
def get_confusion_data(eval_id: str) -> APIResponse[ConfusionMatrixData]:
    """Retrieve confusion matrix and top measured classification confusion pairs."""
    svc = _get_service()
    data = svc.get_confusion_data(eval_id)
    return success_response(data=data, message="Retrieved confusion matrix data")


@router.get("/runs/{eval_id}/pr-curves", response_model=APIResponse[PRCurveData])
def get_pr_curves(eval_id: str) -> APIResponse[PRCurveData]:
    """Retrieve overall and per-class Precision-Recall curve coordinate points."""
    svc = _get_service()
    curves = svc.get_pr_curve_data(eval_id)
    return success_response(data=curves, message="Retrieved PR curve coordinates")


@router.get("/runs/{eval_id}/confidence-dist", response_model=APIResponse[ConfidenceDistributions])
def get_confidence_distributions(eval_id: str) -> APIResponse[ConfidenceDistributions]:
    """Retrieve empirical confidence distributions for TP, FP, and FN."""
    svc = _get_service()
    dist = svc.get_confidence_distributions(eval_id)
    return success_response(data=dist, message="Retrieved confidence distributions")


# ─── Failure Gallery & Clustering Routes ───────────────────────────────


@router.get("/runs/{eval_id}/failures", response_model=APIResponse[list[FailureSampleDetail]])
def get_failure_gallery(
    eval_id: str,
    error_type: ErrorCategory | None = Query(default=None),
    class_name: str | None = Query(default=None),
    confidence_min: float | None = Query(default=None),
    confidence_max: float | None = Query(default=None),
    iou_min: float | None = Query(default=None),
    iou_max: float | None = Query(default=None),
    split: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    object_size: str | None = Query(default=None),
    review_status: str | None = Query(default=None),
    sort_by: str = Query(default="priority"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[FailureSampleDetail]]:
    """Retrieve filtered and prioritized failure gallery."""
    svc = _get_service()
    items = svc.get_failure_gallery(
        bench_or_eval_id=eval_id,
        error_type=error_type,
        class_name=class_name,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
        iou_min=iou_min,
        iou_max=iou_max,
        split=split,
        model_version=model_version,
        object_size=object_size,
        review_status=review_status,
        sort_by=sort_by,
        limit=limit,
        offset=offset,
    )
    return success_response(data=items, message=f"Retrieved {len(items)} failure sample(s)")


@router.get("/runs/{eval_id}/failures/{sample_id}", response_model=APIResponse[FailureSampleDetail])
def get_failure_detail(eval_id: str, sample_id: str) -> APIResponse[FailureSampleDetail]:
    """Retrieve detailed failure sample telemetry and visual memory neighborhood."""
    svc = _get_service()
    detail = svc.get_failure_detail(eval_id, sample_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Failure sample '{sample_id}' not found"
        )
    return success_response(data=detail, message=f"Retrieved failure sample '{sample_id}'")


@router.get(
    "/runs/{eval_id}/failure-clusters", response_model=APIResponse[list[VisualFailureCluster]]
)
def get_failure_clusters(eval_id: str) -> APIResponse[list[VisualFailureCluster]]:
    """Retrieve unsupervised visual clusters (Cluster 1, Cluster 2, Cluster 3) of failure samples."""
    svc = _get_service()
    clusters = svc.get_failure_clusters(eval_id)
    return success_response(
        data=clusters, message=f"Retrieved {len(clusters)} visual failure cluster(s)"
    )


@router.get("/runs/{eval_id}/pattern-analysis", response_model=APIResponse[PatternAnalysisReport])
def get_pattern_analysis(eval_id: str) -> APIResponse[PatternAnalysisReport]:
    """Retrieve failure correlation report across object sizes and image resolutions."""
    svc = _get_service()
    report = svc.get_pattern_analysis(eval_id)
    return success_response(data=report, message="Retrieved failure pattern analysis report")


@router.post(
    "/runs/{eval_id}/failures/{sample_id}/active-learning",
    response_model=APIResponse[dict[str, Any]],
)
def add_failure_to_active_learning(eval_id: str, sample_id: str) -> APIResponse[dict[str, Any]]:
    """Send a verified failure sample directly to the Active Learning candidate queue."""
    svc = _get_service()
    result = svc.send_failure_to_active_learning(eval_id, sample_id)
    return success_response(data=result, message=result.get("message", "Sent to active learning"))


# ─── Model Comparison & Controlled Regression Routes ───────────────────


@router.post("/compare", response_model=APIResponse[ModelComparisonResult])
def compare_models(req: CompareModelsRequest) -> APIResponse[ModelComparisonResult]:
    """Compare two models evaluated on the same dataset split and detect performance regressions."""
    svc = _get_service()
    try:
        cmp_result = svc.compare_benchmarks(
            baseline_id=req.baseline_eval_id,
            candidate_id=req.candidate_eval_id,
            regression_threshold_map50=req.regression_threshold_map50,
            regression_threshold_latency=req.regression_threshold_latency,
        )
        return success_response(
            data=cmp_result,
            message=f"Comparison completed: {cmp_result.regression_status.value}",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
