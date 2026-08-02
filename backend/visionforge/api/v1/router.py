"""API v1 Router registry."""

from fastapi import APIRouter

from visionforge.api.v1.health import router as health_router
from visionforge.api.v1.system import router as system_router

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(system_router)
