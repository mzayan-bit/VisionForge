#!/usr/bin/env bash
set -e

echo "=== Running VisionForge Code Quality Checks ==="

echo "[1/4] Checking Python backend with Ruff..."
(cd backend && uv run ruff check .)

echo "[2/4] Formatting Python backend check..."
(cd backend && uv run ruff format --check .)

echo "[3/4] Running Backend Tests..."
(cd backend && uv run pytest)

echo "[4/4] Checking Frontend TypeScript & ESLint..."
(cd frontend && npm run lint && npx tsc --noEmit)

echo "=== All Quality Checks Passed! ==="
