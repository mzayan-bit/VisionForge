"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check status schema."""

    status: str
    version: str
    service: str


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return backend health status."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        service="visionforge-backend",
    )
