# ==============================================================================
# VisionForge Developer & Operational Task Runner
# ==============================================================================

.PHONY: help install dev up down build logs test lint format seed health reset-dev clean

SHELL := /bin/bash

# Default target
help:
	@echo "VisionForge Task Runner Commands:"
	@echo "  make install     Install local Python (uv) and Node (npm) dependencies"
	@echo "  make dev         Start local development servers (Backend & Frontend)"
	@echo "  make up          Start containerized services with Docker Compose"
	@echo "  make down        Gracefully stop Docker Compose services"
	@echo "  make build       Build Docker images for Backend and Frontend"
	@echo "  make logs        Tail real-time logs from Docker Compose"
	@echo "  make test        Execute full automated test suite"
	@echo "  make lint        Run static analysis (ruff & eslint)"
	@echo "  make format      Format code (ruff & prettier)"
	@echo "  make seed        Ingest real COCO8 benchmark dataset into VisionForge"
	@echo "  make health      Query service health endpoint"
	@echo "  make reset-dev   [DESTRUCTIVE] Reset development cache and container volumes"
	@echo "  make clean       Remove temporary build and test cache artifacts"

# ------------------------------------------------------------------------------
# Dependency Installation
# ------------------------------------------------------------------------------
install:
	@echo "Installing backend dependencies with uv..."
	@cd backend && uv sync
	@echo "Installing frontend dependencies with npm..."
	@cd frontend && npm ci

# ------------------------------------------------------------------------------
# Local Development Execution
# ------------------------------------------------------------------------------
dev:
	@echo "Starting VisionForge local development servers..."
	@./scripts/dev.sh

# ------------------------------------------------------------------------------
# Container Deployment (Docker Compose)
# ------------------------------------------------------------------------------
up:
	@echo "Starting VisionForge services with Docker Compose..."
	@docker compose up -d
	@echo "VisionForge started successfully!"
	@echo "  Frontend UI : http://localhost:3000"
	@echo "  Backend API : http://localhost:8000"
	@echo "  API Docs    : http://localhost:8000/docs"

down:
	@echo "Stopping VisionForge Docker Compose services..."
	@docker compose down

build:
	@echo "Building VisionForge Docker container images..."
	@docker compose build

logs:
	@docker compose logs -f

health:
	@echo "Checking VisionForge Backend Health..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "Backend unreachable"

# ------------------------------------------------------------------------------
# Quality Assurance & Testing
# ------------------------------------------------------------------------------
test:
	@echo "Running backend test suite..."
	@cd backend && uv run pytest -v
	@echo "Running frontend type check..."
	@cd frontend && npx tsc --noEmit

lint:
	@echo "Linting backend with ruff..."
	@cd backend && uv run ruff check .
	@echo "Checking frontend..."
	@cd frontend && npm run lint

format:
	@echo "Formatting backend code..."
	@cd backend && uv run ruff format . && uv run ruff check --fix .

# ------------------------------------------------------------------------------
# Dataset Seeding
# ------------------------------------------------------------------------------
seed:
	@echo "Seeding real COCO8 benchmark dataset into VisionForge..."
	@cd backend && uv run python -c "from visionforge.datasets.adapters.coco8_adapter import COCO8Adapter; summary, manifest, profile = COCO8Adapter().ingest_dataset(); print('Seeded dataset:', manifest.preparation_id, 'with', summary.total_images, 'images and', summary.total_annotations, 'annotations.')"

# ------------------------------------------------------------------------------
# Environment Reset & Cleanup
# ------------------------------------------------------------------------------
reset-dev:
	@echo "=================================================================="
	@echo "WARNING: This command is DESTRUCTIVE and will reset development"
	@echo "cache, database state, and local Docker volumes."
	@echo "=================================================================="
	@docker compose down -v || true
	@rm -rf ~/.cache/visionforge/datasets ~/.cache/visionforge/memory ~/.cache/visionforge/models ~/.cache/visionforge/training ~/.cache/visionforge/evaluations ~/.cache/visionforge/experiments
	@echo "Development environment has been completely reset."

clean:
	@echo "Cleaning temporary build files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf frontend/.next frontend/out backend/~ 2>/dev/null || true
	@echo "Clean completed."
