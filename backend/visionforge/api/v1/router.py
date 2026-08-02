"""API v1 Router registry."""

from fastapi import APIRouter

from visionforge.api.v1.health import router as health_router
from visionforge.api.v1.system import router as system_router

router = APIRouter(prefix="/v1")
router.include_router(health_router)
router.include_router(system_router)

__all__ = ["router"]
