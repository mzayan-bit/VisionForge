"""FastAPI Dependency Injection Layer."""

import platform
import sys
from typing import Annotated, Any

from fastapi import Depends

from visionforge.core.config import VisionForgeSettings, get_settings
from visionforge.core.lifecycle import get_uptime_seconds


def get_settings_dep() -> VisionForgeSettings:
    """Inject application settings instance."""
    return get_settings()


def get_uptime_dep() -> float:
    """Inject current application uptime in seconds."""
    return get_uptime_seconds()


def get_system_runtime_dep(
    settings: Annotated[VisionForgeSettings, Depends(get_settings_dep)],
    uptime: Annotated[float, Depends(get_uptime_dep)],
) -> dict[str, Any]:
    """Inject diagnostic runtime metadata."""
    return {
        "project": settings.project_name,
        "version": settings.version,
        "environment": settings.environment.value,
        "debug": settings.debug,
        "uptime_seconds": uptime,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


# Type Aliases for Dependency Injection in Routers
SettingsDep = Annotated[VisionForgeSettings, Depends(get_settings_dep)]
UptimeDep = Annotated[float, Depends(get_uptime_dep)]
SystemRuntimeDep = Annotated[dict[str, Any], Depends(get_system_runtime_dep)]
