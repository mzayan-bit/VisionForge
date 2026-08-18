#!/usr/bin/env bash
set -e

trap 'kill $(jobs -p) 2>/dev/null || true; exit' SIGINT SIGTERM EXIT

echo "=== Starting VisionForge Development Environment ==="

# Check backend
if [ -d "backend" ]; then
    echo "Starting Backend API Server on http://localhost:8000..."
    (cd backend && uv run uvicorn visionforge.main:app --reload --host 0.0.0.0 --port 8000) &
fi

# Check frontend
if [ -d "frontend" ]; then
    echo "Starting Frontend Development Server on http://localhost:3000..."
    (cd frontend && npm run dev) &
fi

wait
