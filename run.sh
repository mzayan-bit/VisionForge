#!/usr/bin/env bash
set -e

# Change to the root directory where the script is located
cd "$(dirname "$0")"

echo "============================================================"
echo "          🚀 Starting VisionForge Research Workbench         "
echo "============================================================"

# 1. Ensure .env exists
if [ ! -f ".env" ]; then
    echo "📋 Copying .env.example -> .env..."
    cp .env.example .env
fi

# 2. Seed COCO8 Dataset if not already seeded
echo "🌱 Initializing / verifying real COCO8 benchmark dataset..."
(cd backend && uv run python -c "
from visionforge.datasets.adapters.coco8_adapter import COCO8Adapter
try:
    summary, manifest, profile = COCO8Adapter().ingest_dataset()
    print(f'   ✔ Dataset ready: {manifest.preparation_id} ({summary.total_images} images, {summary.total_annotations} annotations)')
except Exception as e:
    print(f'   ℹ Dataset check notice: {e}')
") || true

# 3. Graceful shutdown handler
cleanup() {
    echo ""
    echo "🛑 Shutting down VisionForge servers..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 4. Start Backend API Server
echo "⚡ Starting Backend API Server (FastAPI on http://localhost:8000)..."
(cd backend && uv run uvicorn visionforge.main:app --reload --host 0.0.0.0 --port 8000) &

# 5. Start Frontend Development Server
echo "💻 Starting Frontend Workbench (Next.js on http://localhost:3000)..."
(cd frontend && npm run dev) &

# 6. Wait 3 seconds and attempt to open browser automatically
(
    sleep 3
    echo ""
    echo "============================================================"
    echo "  🎉 VisionForge is running!"
    echo "  👉 Workbench UI : http://localhost:3000"
    echo "  👉 API Docs     : http://localhost:8000/docs"
    echo "  👉 Metrics      : http://localhost:8000/metrics"
    echo "  👉 Press Ctrl+C to stop all servers anytime"
    echo "============================================================"
    echo ""
    if command -v open >/dev/null 2>&1; then
        open "http://localhost:3000" 2>/dev/null || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://localhost:3000" 2>/dev/null || true
    fi
) &

# Wait for background processes to keep running until user presses Ctrl+C
wait
