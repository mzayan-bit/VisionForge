"""System diagnostics, Prometheus metrics, and Job Observability API endpoints."""

from typing import Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from visionforge.core.dependencies import SystemRuntimeDep
from visionforge.core.exceptions import JobNotFoundException
from visionforge.core.responses import APIResponse, success_response
from visionforge.core.telemetry import (
    FailureRecord,
    JobRecord,
    SystemDiagnosticsSnapshot,
    get_metrics_collector,
)

router = APIRouter(tags=["System"])


class SystemInfoData(BaseModel):
    """System information response payload."""

    project: str = Field(description="Project display name")
    version: str = Field(description="Backend application version")
    environment: str = Field(description="Active execution environment")
    debug: bool = Field(description="Debug mode state")
    uptime_seconds: float = Field(description="Total uptime in seconds")
    python_version: str = Field(description="Python interpreter version")
    platform: str = Field(description="Host platform architecture")
    status: str = Field(default="ready", description="System operational readiness status")
    total_routes: int = Field(description="Number of registered API routes")
    registered_endpoints: list[str] = Field(
        description="List of all registered endpoint HTTP paths"
    )
    ai_core: dict[str, Any] = Field(description="AI Core subsystem diagnostics")
    vision_engine: dict[str, Any] = Field(description="Vision Engine execution layer diagnostics")
    model_manager: dict[str, Any] = Field(description="Model Manager subsystem diagnostics")


@router.get(
    "/system/info",
    response_model=APIResponse[SystemInfoData],
    summary="Get system diagnostics and runtime metadata",
    description="Returns platform, runtime, uptime, AI Core, Vision Engine, & Model Manager info.",
)
async def get_system_info(
    request: Request,
    runtime_info: SystemRuntimeDep,
) -> APIResponse[SystemInfoData]:
    """Return backend runtime diagnostics information."""
    app = request.app

    # Extract OpenAPI path keys for clean registered endpoint discovery
    openapi_schema = app.openapi()
    endpoints = sorted(list(openapi_schema.get("paths", {}).keys()))

    info_data = SystemInfoData(
        project=runtime_info["project"],
        version=runtime_info["version"],
        environment=runtime_info["environment"],
        debug=runtime_info["debug"],
        uptime_seconds=runtime_info["uptime_seconds"],
        python_version=runtime_info["python_version"],
        platform=runtime_info["platform"],
        status="ready",
        total_routes=len(endpoints),
        registered_endpoints=endpoints,
        ai_core=runtime_info["ai_core"],
        vision_engine=runtime_info["vision_engine"],
        model_manager=runtime_info["model_manager"],
    )

    return success_response(
        data=info_data,
        message="System info diagnostics retrieved successfully",
    )


@router.get(
    "/system/diagnostics",
    response_model=APIResponse[SystemDiagnosticsSnapshot],
    summary="Get live system operational telemetry and diagnostics",
    description="Returns real request rates, error counts, latency percentiles, queue depth, and recent failures.",
)
async def get_system_diagnostics() -> APIResponse[SystemDiagnosticsSnapshot]:
    """Return real-time operational telemetry snapshot."""
    collector = get_metrics_collector()
    snapshot = collector.get_snapshot()
    return success_response(
        data=snapshot,
        message="System operational telemetry snapshot retrieved successfully",
    )


@router.get(
    "/system/metrics",
    summary="Get Prometheus formatted metrics",
    description="Exposes system telemetry, request counts, latencies, and CV operational metrics for Prometheus scrapers.",
)
async def get_prometheus_metrics() -> Response:
    """Return Prometheus text exposition format metrics."""
    collector = get_metrics_collector()
    return PlainTextResponse(
        content=collector.export_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get(
    "/system/jobs",
    response_model=APIResponse[list[JobRecord]],
    summary="List background jobs and workloads",
    description="Returns list of recent and active background jobs with lifecycle timestamps, progress, and error details.",
)
async def get_system_jobs(
    limit: int = Query(default=50, ge=1, le=200, description="Max jobs to return"),
) -> APIResponse[list[JobRecord]]:
    """Return list of observed background jobs."""
    collector = get_metrics_collector()
    jobs = collector.list_jobs(limit=limit)
    return success_response(
        data=jobs,
        message=f"Retrieved {len(jobs)} background jobs",
    )


@router.get(
    "/system/jobs/{job_id}",
    response_model=APIResponse[JobRecord],
    summary="Get job details by ID",
    description="Returns deep execution metadata, duration, and error summary for a specific job.",
)
async def get_system_job(job_id: str) -> APIResponse[JobRecord]:
    """Retrieve details for a specific background job."""
    collector = get_metrics_collector()
    job = collector.get_job(job_id)
    if not job:
        raise JobNotFoundException(job_id=job_id)
    return success_response(
        data=job,
        message=f"Job '{job_id}' details retrieved",
    )


@router.get(
    "/system/errors",
    response_model=APIResponse[list[FailureRecord]],
    summary="List recent subsystem failures",
    description="Returns ring-buffer of recent operational failures with request IDs and diagnostic context.",
)
async def get_recent_errors() -> APIResponse[list[FailureRecord]]:
    """Return list of recent subsystem failures."""
    collector = get_metrics_collector()
    snapshot = collector.get_snapshot()
    return success_response(
        data=snapshot.recent_failures,
        message=f"Retrieved {len(snapshot.recent_failures)} failure records",
    )
