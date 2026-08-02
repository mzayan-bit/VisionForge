#!/usr/bin/env bash
set -e

echo "=== Starting VisionForge Development Environment ==="

# Check backend
if [ -d "backend" ]; then
    echo "Starting Backend API Server..."
    (cd backend && uv run uvicorn visionforge.main:app --reload --port 8000) &
fi

# Check frontend
if [ -d "frontend" ]; then
    echo "Starting Frontend Development Server..."
    (cd frontend && npm run dev) &
fi

wait
