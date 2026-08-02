"""System diagnostics endpoint."""

import platform
import sys

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["System"])


class SystemInfoResponse(BaseModel):
    """System information response schema."""

    project: str
    version: str
    python_version: str
    platform: str
    status: str


@router.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info() -> SystemInfoResponse:
    """Return backend runtime diagnostics information."""
    return SystemInfoResponse(
        project="VisionForge Workbench",
        version="0.1.0",
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        status="ready",
    )
