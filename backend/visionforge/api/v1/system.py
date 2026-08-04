"""System diagnostics API endpoints."""

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from visionforge.core.dependencies import SystemRuntimeDep
from visionforge.core.responses import APIResponse, success_response

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
