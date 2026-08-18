"""VisionForge Main Application Entrypoint."""

from fastapi import FastAPI, Response

from visionforge.api.v1.router import router as api_v1_router
from visionforge.core.config import get_settings
from visionforge.core.exceptions import register_exception_handlers
from visionforge.core.lifecycle import lifespan
from visionforge.core.logging import setup_logging
from visionforge.core.middleware import register_middleware
from visionforge.core.responses import APIResponse, success_response


def create_app() -> FastAPI:
    """Application factory for VisionForge Workbench FastAPI service."""
    settings = get_settings()

    # 1. Initialize Logging System
    setup_logging(settings.log_level)

    # 2. Construct FastAPI application instance
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description="VisionForge Computer Vision Workbench Backend API",
        lifespan=lifespan,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
    )

    # 3. Register Middleware (Tracing & CORS)
    register_middleware(app, settings)

    # 4. Register Centralized Exception Handlers
    register_exception_handlers(app)

    # 5. Include API Version Routers
    app.include_router(api_v1_router, prefix="/api")

    # 6. Root Metadata, Probes & Metrics Endpoints
    @app.get(
        "/health",
        summary="Direct Health Check",
        description="Returns lightweight health check status for Docker/Kubernetes container orchestrators.",
    )
    async def health_check() -> dict:
        """Lightweight container health probe endpoint."""
        return {
            "status": "ok",
            "service": "visionforge-backend",
            "version": settings.version,
            "environment": settings.environment.value,
        }

    @app.get(
        "/ready",
        summary="Direct Readiness Check",
        description="Returns readiness state for load balancers and orchestrators.",
    )
    async def ready_check() -> dict:
        """Direct readiness probe endpoint."""
        return {
            "ready": True,
            "status": "ready",
            "service": "visionforge-backend",
            "version": settings.version,
        }

    @app.get(
        "/metrics",
        summary="Prometheus Metrics Exposition",
        description="Emits real operational metrics in Prometheus text exposition format.",
    )
    async def prometheus_metrics() -> Response:
        """Prometheus metrics endpoint."""
        from fastapi.responses import PlainTextResponse

        from visionforge.core.telemetry import get_metrics_collector

        collector = get_metrics_collector()
        return PlainTextResponse(
            content=collector.export_prometheus_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get(
        "/",
        response_model=APIResponse[dict],
        summary="Root API Metadata",
        description="Returns core API metadata and health status links.",
    )
    async def root() -> APIResponse[dict]:
        """Root endpoint returning basic metadata and API version routes."""
        return success_response(
            data={
                "name": settings.project_name,
                "version": settings.version,
                "environment": settings.environment.value,
                "docs": settings.docs_url,
                "health": "/api/v1/health",
                "system": "/api/v1/system/info",
            },
            message="Welcome to VisionForge Workbench API",
        )

    return app


app = create_app()
