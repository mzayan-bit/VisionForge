"""REST API Router for VisionForge Data-Centric Computer Vision Workspace."""

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from visionforge.core.responses import APIResponse, success_response
from visionforge.datasets.intelligence_schemas import (
    CurationDecision,
    DatasetDiffResult,
    DatasetHealthSummary,
    DatasetProfile,
    DatasetVersionRecord,
    HardSampleItem,
    LeakageCandidatePair,
    QualityIssueItem,
)
from visionforge.datasets.intelligence_service import get_dataset_intelligence_service

router = APIRouter(prefix="/datasets/intelligence", tags=["Dataset Intelligence & Curation"])


class CreateVersionRequest(BaseModel):
    """Payload to commit a new dataset version."""

    dataset_id: str = Field(description="Target dataset identifier")
    version_id: str = Field(description="New version string (e.g. 'v2.0.0')")
    parent_version_id: str | None = Field(default=None, description="Parent version ID if iterated")
    changes_summary: str = Field(description="Summary of curation decisions applied")
    total_samples: int = Field(description="Total sample count")
    total_annotations: int = Field(description="Total annotation count")


def _get_service():
    return get_dataset_intelligence_service()


@router.get(
    "/profile",
    response_model=APIResponse[DatasetProfile],
    summary="Get Dataset Profile Snapshot",
    description="Returns detailed dataset profile including class distributions, image statistics, annotation geometry, and health status.",
)
def get_dataset_profile(
    dataset_id: str = Query(default="safety_v2"),
    version: str = Query(default="v2.0.0"),
) -> APIResponse[DatasetProfile]:
    """Retrieve comprehensive dataset profile."""
    svc = _get_service()
    try:
        profile = svc.get_or_compute_profile(dataset_id=dataset_id, dataset_version=version)
        return success_response(
            data=profile,
            message=f"Retrieved profile for '{dataset_id}' ({version}) containing {profile.total_samples} samples",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/health",
    response_model=APIResponse[DatasetHealthSummary],
    summary="Get Dataset Health Scorecard",
    description="Returns categorized health status for Data Integrity, Annotation Quality, Class Balance, Visual Diversity, Leakage, and Model Difficulty.",
)
def get_dataset_health(
    dataset_id: str = Query(default="safety_v2"),
    version: str = Query(default="v2.0.0"),
) -> APIResponse[DatasetHealthSummary]:
    """Retrieve dataset health scorecard."""
    svc = _get_service()
    profile = svc.get_or_compute_profile(dataset_id=dataset_id, dataset_version=version)
    return success_response(
        data=profile.health_summary, message="Dataset health scorecard evaluated"
    )


@router.get(
    "/issues",
    response_model=APIResponse[list[QualityIssueItem]],
    summary="List Flagged Quality Issues",
    description="Returns diagnostic quality issues filtered by type, severity, or split.",
)
def get_quality_issues(
    dataset_id: str = Query(default="safety_v2"),
    issue_type: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    split: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[QualityIssueItem]]:
    """List diagnostic quality issues."""
    svc = _get_service()
    issues = svc.get_quality_issues(
        dataset_id=dataset_id,
        issue_type=issue_type,
        severity=severity,
        split=split,
        limit=limit,
        offset=offset,
    )
    return success_response(
        data=issues, message=f"Retrieved {len(issues)} flagged quality issue(s)"
    )


@router.get(
    "/leakage",
    response_model=APIResponse[list[LeakageCandidatePair]],
    summary="List Cross-Split Leakage Candidates",
    description="Returns detected exact hash duplicates and visual similarity candidates across split boundaries.",
)
def get_cross_split_leakage(
    dataset_id: str = Query(default="safety_v2"),
) -> APIResponse[list[LeakageCandidatePair]]:
    """Retrieve cross-split leakage pairs."""
    svc = _get_service()
    pairs = svc.get_cross_split_leakage(dataset_id=dataset_id)
    return success_response(
        data=pairs, message=f"Retrieved {len(pairs)} cross-split leakage candidate pair(s)"
    )


@router.get(
    "/hard-samples",
    response_model=APIResponse[list[HardSampleItem]],
    summary="List Prioritized Hard Samples",
    description="Returns difficult dataset samples ranked by composite prioritization score.",
)
def get_hard_samples(
    dataset_id: str = Query(default="safety_v2"),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    split: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[HardSampleItem]]:
    """Retrieve prioritized hard samples."""
    svc = _get_service()
    items = svc.get_hard_samples(
        dataset_id=dataset_id,
        min_score=min_score,
        split=split,
        limit=limit,
        offset=offset,
    )
    return success_response(
        data=items, message=f"Retrieved {len(items)} prioritized hard sample(s)"
    )


@router.post(
    "/review",
    response_model=APIResponse[dict[str, str]],
    summary="Submit Curation Decision",
    description="Submits a human review decision (accept, reject, correct, duplicate, etc.) to Human Review Queue.",
)
def submit_curation_decision(
    decision: CurationDecision,
) -> APIResponse[dict[str, str]]:
    """Submit reviewer decision."""
    svc = _get_service()
    svc.record_curation_decision(decision)
    return success_response(
        data={
            "review_id": decision.review_id,
            "sample_id": decision.sample_id,
            "decision": decision.decision,
        },
        message=f"Recorded review decision '{decision.decision}' for sample '{decision.sample_id}'",
    )


@router.get(
    "/review",
    response_model=APIResponse[list[CurationDecision]],
    summary="List Curation Decisions",
    description="Returns past curation review decisions recorded in Human Review Queue.",
)
def list_curation_decisions(
    category: str | None = Query(default=None),
    sample_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[CurationDecision]]:
    """List review decisions."""
    svc = _get_service()
    decisions = svc.list_curation_decisions(
        category=category,
        sample_id=sample_id,
        limit=limit,
        offset=offset,
    )
    return success_response(
        data=decisions, message=f"Retrieved {len(decisions)} curation review decision(s)"
    )


@router.post(
    "/versions",
    response_model=APIResponse[DatasetVersionRecord],
    summary="Commit Dataset Version Snapshot",
    description="Commits an immutable dataset version snapshot with manifest fingerprint.",
)
def create_dataset_version(
    req: CreateVersionRequest,
) -> APIResponse[DatasetVersionRecord]:
    """Create dataset version snapshot."""
    svc = _get_service()
    rec = svc.create_dataset_version(
        dataset_id=req.dataset_id,
        version_id=req.version_id,
        parent_version_id=req.parent_version_id,
        changes_summary=req.changes_summary,
        total_samples=req.total_samples,
        total_annotations=req.total_annotations,
    )
    return success_response(data=rec, message=f"Created dataset version '{rec.version_id}'")


@router.get(
    "/versions",
    response_model=APIResponse[list[DatasetVersionRecord]],
    summary="List Dataset Versions",
    description="Lists all committed version snapshots for a dataset.",
)
def list_dataset_versions(
    dataset_id: str = Query(default="safety_v2"),
) -> APIResponse[list[DatasetVersionRecord]]:
    """List dataset versions."""
    svc = _get_service()
    versions = svc.list_dataset_versions(dataset_id=dataset_id)
    return success_response(data=versions, message=f"Retrieved {len(versions)} dataset version(s)")


@router.get(
    "/diff",
    response_model=APIResponse[DatasetDiffResult],
    summary="Compute Dataset Version Diff",
    description="Granular comparison between two dataset versions showing sample additions/removals, class shifts, and leakage diffs.",
)
def compute_dataset_diff(
    dataset_id: str = Query(default="safety_v2"),
    version_a: str = Query(default="v1.0.0"),
    version_b: str = Query(default="v2.0.0"),
) -> APIResponse[DatasetDiffResult]:
    """Compute dataset diff."""
    svc = _get_service()
    diff = svc.compute_dataset_diff(
        dataset_id=dataset_id,
        version_a=version_a,
        version_b=version_b,
    )
    return success_response(data=diff, message=f"Computed diff between {version_a} and {version_b}")


@router.get(
    "/report",
    summary="Generate Dataset Markdown Report",
    description="Produces a structured Markdown report of dataset health, statistics, and curation findings.",
)
def get_dataset_report(
    dataset_id: str = Query(default="safety_v2"),
    version: str = Query(default="v2.0.0"),
):
    """Generate Markdown dataset report."""
    svc = _get_service()
    report_md = svc.generate_dataset_report(dataset_id=dataset_id, version_id=version)
    return Response(content=report_md, media_type="text/markdown")
