"""Health check API endpoints."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from visionforge.core.dependencies import (
    CacheManagerDep,
    DeviceManagerDep,
    ModelManagerDep,
    ModelRegistryDep,
    SettingsDep,
    UptimeDep,
)
from visionforge.core.responses import APIResponse, success_response

router = APIRouter(tags=["Health"])


class HealthData(BaseModel):
    """Health check diagnostic data payload."""

    status: str = Field(default="ok", description="Service health state")
    version: str = Field(description="Backend application version")
    service: str = Field(default="visionforge-backend", description="Service name identifier")
    environment: str = Field(description="Active execution environment")
    uptime_seconds: float = Field(description="Application uptime in seconds")
    ai_core: dict[str, Any] = Field(description="AI Core telemetry status")


@router.get(
    "/health",
    response_model=APIResponse[HealthData],
    summary="Get backend health status",
    description="Returns service status, version, active environment, uptime, and AI Core health.",
)
async def get_health(
    settings: SettingsDep,
    uptime: UptimeDep,
    registry: ModelRegistryDep,
    device_mgr: DeviceManagerDep,
    cache_mgr: CacheManagerDep,
    model_mgr: ModelManagerDep,
) -> APIResponse[HealthData]:
    """Return backend health status diagnostics wrapped in standard response envelope."""
    cache_stats = cache_mgr.get_cache_stats()
    manager_status = model_mgr.get_manager_status()

    health_info = HealthData(
        status="ok",
        version=settings.version,
        service="visionforge-backend",
        environment=settings.environment.value,
        uptime_seconds=uptime,
        ai_core={
            "status": "ready",
            "registered_models": registry.count(),
            "installed_models": manager_status["installed_models"],
            "optimal_device": device_mgr.get_optimal_device().value,
            "cache_size_mb": cache_stats.total_size_mb,
            "available_storage": manager_status["storage"],
            "configuration_status": "ok",
            "model_manager_status": manager_status["status"],
        },
    )
    return success_response(
        data=health_info,
        message="Backend health check passed",
    )
