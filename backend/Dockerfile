# ==============================================================================
# VisionForge Backend Multi-Stage Production Dockerfile
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Dependencies via uv
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install system compilation packages and curl for uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.5.24 /uv /bin/uv

# Copy dependency specifications
COPY pyproject.toml uv.lock* ./

# Create virtualenv and install dependencies deterministically
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN uv sync --frozen --no-dev --no-install-project || uv sync --no-dev --no-install-project

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_CACHE_DIR="/data/models" \
    DATA_DIR="/data" \
    HOST="0.0.0.0" \
    PORT=8000 \
    ENVIRONMENT="production"

WORKDIR /app

# Install minimal runtime system libraries for computer vision (OpenCV, PIL, TorchVision) & curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user
RUN groupadd -g 10001 visionforge && \
    useradd -u 10001 -g visionforge -s /bin/bash -m appuser && \
    mkdir -p /data/models /data/datasets /data/memory /data/training /data/evaluations /data/experiments && \
    chown -R appuser:visionforge /app /data

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy backend source code
COPY --chown=appuser:visionforge visionforge/ /app/visionforge/
COPY --chown=appuser:visionforge pyproject.toml /app/

# Switch to non-root user
USER appuser

# Expose backend API port
EXPOSE 8000

# Container Healthcheck probe
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application with uvicorn
CMD ["uvicorn", "visionforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
