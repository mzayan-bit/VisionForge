"""VisionForge Main Application Entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from visionforge.api.v1.router import api_v1_router
from visionforge.config import settings
from visionforge.core.logging import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for backend startup and shutdown."""
    logger.info("Initializing VisionForge Workbench Backend v%s", settings.version)
    yield
    logger.info("Shutting down VisionForge Workbench Backend")


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="VisionForge Computer Vision Workbench Backend API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint returning basic metadata."""
    return {
        "name": settings.project_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
