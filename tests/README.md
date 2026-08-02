# VisionForge Tests Architecture

This directory houses cross-cutting system integration and benchmark test suites for VisionForge.

## Directory Layout

- `unit/`: Component-level isolated unit tests.
- `integration/`: End-to-end API and workflow integration tests.
- `benchmarks/`: Performance and foundational latency benchmarking routines.

## Running Tests

Backend unit tests can be executed directly using `uv`:

```bash
cd backend
uv run pytest
```
