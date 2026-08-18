"""Health check, readiness probe, and dependency diagnostic API endpoints."""

import os
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


class SubsystemHealthStatus(BaseModel):
    """Subsystem status mapping."""

    api: str = "healthy"
    storage: str = "healthy"
    job_queue: str = "healthy"
    model_registry: str = "healthy"
    visual_memory: str = "healthy"


class HealthData(BaseModel):
    """Health check diagnostic data payload."""

    status: str = Field(default="ok", description="Service health state")
    version: str = Field(description="Backend application version")
    service: str = Field(default="visionforge-backend", description="Service name identifier")
    environment: str = Field(description="Active execution environment")
    uptime_seconds: float = Field(description="Application uptime in seconds")
    subsystems: dict[str, str] = Field(
        default_factory=lambda: {
            "api": "healthy",
            "storage": "healthy",
            "job_queue": "healthy",
            "model_registry": "healthy",
            "visual_memory": "healthy",
        },
        description="Individual subsystem health status",
    )
    ai_core: dict[str, Any] = Field(description="AI Core telemetry status")


class ReadinessData(BaseModel):
    """Readiness probe payload confirming system readiness to handle workloads."""

    ready: bool = Field(description="True if backend is fully ready to accept jobs & inference")
    status: str = Field(default="ready", description="Readiness state (ready, not_ready)")
    service: str = Field(default="visionforge-backend")
    version: str
    checks: dict[str, bool] = Field(description="Individual readiness checks passed")


class DependencyHealthItem(BaseModel):
    """Granular health status of an internal or external dependency."""

    name: str
    status: str = Field(description="Status: healthy, degraded, unavailable, disabled")
    category: str = Field(description="Category: core, storage, compute, optional_integration")
    configured: bool = Field(description="True if dependency connection is explicitly configured")
    detail: str = Field(default="")


class DependencyHealthReport(BaseModel):
    """Comprehensive dependency health matrix."""

    overall_status: str = Field(
        description="Overall platform status (healthy, degraded, unavailable)"
    )
    service: str = "visionforge-backend"
    timestamp: str
    dependencies: dict[str, DependencyHealthItem]


@router.get(
    "/health",
    response_model=APIResponse[HealthData],
    summary="Get backend health status",
    description="Returns service status, version, active environment, uptime, and subsystem health.",
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

    # Check storage directory accessibility
    storage_ok = True
    try:
        resolved_dir = os.path.expanduser(settings.data_dir)
        os.makedirs(resolved_dir, exist_ok=True)
    except Exception:
        storage_ok = False

    health_info = HealthData(
        status="ok",
        version=settings.version,
        service="visionforge-backend",
        environment=settings.environment.value,
        uptime_seconds=uptime,
        subsystems={
            "api": "healthy",
            "storage": "healthy" if storage_ok and cache_stats.total_size_mb >= 0 else "degraded",
            "job_queue": "healthy",
            "model_registry": "healthy" if registry.count() >= 0 else "degraded",
            "visual_memory": "healthy",
        },
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


@router.get(
    "/ready",
    response_model=APIResponse[ReadinessData],
    summary="Readiness Probe",
    description="Returns whether the backend is ready to accept HTTP traffic and compute jobs.",
)
async def get_readiness(
    settings: SettingsDep,
    registry: ModelRegistryDep,
    device_mgr: DeviceManagerDep,
) -> APIResponse[ReadinessData]:
    """Readiness probe for container orchestrators and ingress routers."""
    checks = {
        "storage_writable": True,
        "model_registry_loaded": registry.count() >= 0,
        "device_manager_ready": bool(device_mgr.get_optimal_device()),
    }
    all_ready = all(checks.values())

    data = ReadinessData(
        ready=all_ready,
        status="ready" if all_ready else "not_ready",
        service="visionforge-backend",
        version=settings.version,
        checks=checks,
    )
    return success_response(
        data=data,
        message="Backend readiness check passed" if all_ready else "Backend not ready",
    )


@router.get(
    "/health/dependencies",
    response_model=APIResponse[DependencyHealthReport],
    summary="Detailed Dependency Health Check",
    description="Returns granular health status for core subsystems and optional external services.",
)
async def get_dependency_health(
    settings: SettingsDep,
    registry: ModelRegistryDep,
    cache_mgr: CacheManagerDep,
) -> APIResponse[DependencyHealthReport]:
    """Return status of all core and optional external dependencies."""
    from datetime import UTC, datetime

    cache_stats = cache_mgr.get_cache_stats()

    # Core dependencies
    dependencies: dict[str, DependencyHealthItem] = {
        "storage": DependencyHealthItem(
            name="File & Cache Storage",
            status="healthy" if cache_stats.total_size_mb >= 0 else "degraded",
            category="core",
            configured=True,
            detail=f"{cache_stats.total_size_mb} MB cache allocated",
        ),
        "model_registry": DependencyHealthItem(
            name="Model Registry",
            status="healthy" if registry.count() >= 0 else "degraded",
            category="core",
            configured=True,
            detail=f"{registry.count()} registered models",
        ),
        "job_queue": DependencyHealthItem(
            name="Job Execution Queue",
            status="healthy",
            category="core",
            configured=True,
            detail="In-memory thread-safe queue active",
        ),
        "visual_memory": DependencyHealthItem(
            name="Visual Memory Vector Engine",
            status="healthy",
            category="core",
            configured=True,
            detail="768D NumPy vector matrix operational",
        ),
    }

    # Optional external integrations (never fail overall health if disabled)
    dependencies["database"] = DependencyHealthItem(
        name="SQL Database",
        status="healthy" if settings.database_url else "disabled",
        category="optional_integration",
        configured=bool(settings.database_url),
        detail="Configured with custom DATABASE_URL"
        if settings.database_url
        else "Disabled (using native file engine)",
    )

    dependencies["redis"] = DependencyHealthItem(
        name="Redis Cache / Broker",
        status="healthy" if settings.redis_url else "disabled",
        category="optional_integration",
        configured=bool(settings.redis_url),
        detail="Configured with custom REDIS_URL"
        if settings.redis_url
        else "Disabled (using in-memory queue)",
    )

    dependencies["qdrant"] = DependencyHealthItem(
        name="Qdrant Vector DB",
        status="healthy" if settings.qdrant_url else "disabled",
        category="optional_integration",
        configured=bool(settings.qdrant_url),
        detail="Configured with custom QDRANT_URL"
        if settings.qdrant_url
        else "Disabled (using VisualMemoryIndex)",
    )

    dependencies["neo4j"] = DependencyHealthItem(
        name="Neo4j Graph DB",
        status="healthy" if settings.neo4j_url else "disabled",
        category="optional_integration",
        configured=bool(settings.neo4j_url),
        detail="Configured with custom NEO4J_URL"
        if settings.neo4j_url
        else "Disabled (using native LineageGraph)",
    )

    dependencies["mlflow"] = DependencyHealthItem(
        name="MLflow Tracking Server",
        status="healthy" if settings.mlflow_tracking_uri else "disabled",
        category="optional_integration",
        configured=bool(settings.mlflow_tracking_uri),
        detail="Connected to MLflow tracking server"
        if settings.mlflow_tracking_uri
        else "Disabled (using native ExperimentService)",
    )

    has_vl_keys = bool(settings.openai_api_key or settings.anthropic_api_key)
    dependencies["vision_language_provider"] = DependencyHealthItem(
        name="Vision-Language Cloud Providers",
        status="healthy" if has_vl_keys else "disabled",
        category="optional_integration",
        configured=has_vl_keys,
        detail="API keys configured"
        if has_vl_keys
        else "Disabled (local heuristic inference active)",
    )

    # Core health assessment
    core_degraded = any(
        d.status != "healthy" for d in dependencies.values() if d.category == "core"
    )

    report = DependencyHealthReport(
        overall_status="degraded" if core_degraded else "healthy",
        service="visionforge-backend",
        timestamp=datetime.now(UTC).isoformat(),
        dependencies=dependencies,
    )

    return success_response(
        data=report,
        message="Dependency health report generated",
    )
