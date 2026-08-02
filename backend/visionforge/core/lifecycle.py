"""Application Lifecycle and Event Handlers."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from visionforge.core.config import get_settings
from visionforge.core.logging import get_logger

logger = get_logger("lifecycle")

_boot_time: float = 0.0


def get_boot_time() -> float:
    """Return the timestamp when application started."""
    return _boot_time


def get_uptime_seconds() -> float:
    """Return total application uptime in seconds."""
    if _boot_time == 0.0:
        return 0.0
    return round(time.time() - _boot_time, 2)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager handling application startup and shutdown events."""
    global _boot_time
    _boot_time = time.time()

    settings = get_settings()

    logger.info("==================================================")
    logger.info(" Starting VisionForge Workbench Backend v%s", settings.version)
    logger.info(" Environment : %s", settings.environment.value)
    logger.info(" Debug Mode  : %s", settings.debug)
    logger.info(" Host & Port : %s:%d", settings.host, settings.port)
    logger.info(" Docs URL    : %s", settings.docs_url or "Disabled")
    logger.info("==================================================")

    # Log registered routes
    routes = [route.path for route in app.routes if hasattr(route, "path")]
    logger.info("Registered %d routes on startup.", len(routes))

    yield

    logger.info("Shutting down VisionForge Workbench Backend gracefully...")
