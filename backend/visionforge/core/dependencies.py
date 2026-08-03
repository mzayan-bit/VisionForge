"""FastAPI Dependency Injection Layer."""

import platform
import sys
from typing import Annotated, Any

from fastapi import Depends

from visionforge.ai.cache import CacheManager, get_cache_manager
from visionforge.ai.device import DeviceManager, get_device_manager
from visionforge.ai.registry import ModelRegistry, get_model_registry
from visionforge.core.config import VisionForgeSettings, get_settings
from visionforge.core.lifecycle import get_uptime_seconds


def get_settings_dep() -> VisionForgeSettings:
    """Inject application settings instance."""
    return get_settings()


def get_uptime_dep() -> float:
    """Inject current application uptime in seconds."""
    return get_uptime_seconds()


def get_model_registry_dep() -> ModelRegistry:
    """Inject singleton ModelRegistry instance."""
    return get_model_registry()


def get_device_manager_dep() -> DeviceManager:
    """Inject singleton DeviceManager instance."""
    return get_device_manager()


def get_cache_manager_dep() -> CacheManager:
    """Inject singleton CacheManager instance."""
    return get_cache_manager()


def get_system_runtime_dep(
    settings: Annotated[VisionForgeSettings, Depends(get_settings_dep)],
    uptime: Annotated[float, Depends(get_uptime_dep)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry_dep)],
    device_mgr: Annotated[DeviceManager, Depends(get_device_manager_dep)],
    cache_mgr: Annotated[CacheManager, Depends(get_cache_manager_dep)],
) -> dict[str, Any]:
    """Inject diagnostic runtime metadata including AI Core telemetry."""
    hw_caps = device_mgr.get_hardware_capabilities()
    cache_stats = cache_mgr.get_cache_stats()

    return {
        "project": settings.project_name,
        "version": settings.version,
        "environment": settings.environment.value,
        "debug": settings.debug,
        "uptime_seconds": uptime,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ai_core": {
            "registered_models": registry.count(),
            "optimal_device": hw_caps.optimal_device.value,
            "available_devices": [d.value for d in hw_caps.available_devices],
            "cache_dir": cache_stats.cache_directory,
            "cache_size_mb": cache_stats.total_size_mb,
        },
    }


# Type Aliases for Dependency Injection in Routers
SettingsDep = Annotated[VisionForgeSettings, Depends(get_settings_dep)]
UptimeDep = Annotated[float, Depends(get_uptime_dep)]
SystemRuntimeDep = Annotated[dict[str, Any], Depends(get_system_runtime_dep)]
ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry_dep)]
DeviceManagerDep = Annotated[DeviceManager, Depends(get_device_manager_dep)]
CacheManagerDep = Annotated[CacheManager, Depends(get_cache_manager_dep)]
