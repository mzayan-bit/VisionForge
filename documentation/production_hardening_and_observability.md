# VisionForge Production Hardening & System Observability Architecture

## 1. Objective & Core Principle
The **VisionForge Production Hardening & Observability Layer** ensures reliability, performance, resilience, and traceability across the entire computer vision workbench.

---

## 2. Request Tracing & Correlation
- **`X-Request-ID`**: Every incoming HTTP request is assigned a unique UUID or preserves the client's `X-Request-ID` header.
- **Trace Propagation**: The `request_id` is propagated through:
  - FastAPI middleware
  - Domain services
  - Background jobs
  - Centralized exception handlers
  - Structured log outputs
  - Frontend error modals

---

## 3. Standardized Error Handling & Envelopes
All API endpoints return consistent error responses adhering to the standard envelope:
```json
{
  "success": false,
  "message": "Human-readable error description",
  "data": null,
  "meta": {
    "timestamp": "2026-08-17T08:38:36.747465+00:00",
    "request_id": "33e50b74-e7a0-4cfc-9849-930c5f6cd23f"
  },
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Dataset with ID 'safety_v2' was not found",
    "details": []
  }
}
```

---

## 4. Generic Pagination & Query Filtering
- Standardized `PaginatedResponse[T]` envelope supporting `items`, `total`, `page`, `page_size`, `total_pages`, `has_next`, and `has_prev`.
- Bounds protection: Page size clamped between 1 and 100 items to prevent accidental out-of-memory overhead on large collections.

---

## 5. Live Operational Telemetry (`MetricsCollector`)
The thread-safe [`MetricsCollector`](../backend/visionforge/core/telemetry.py) tracks:
- Total requests and error rates.
- Rolling average latency and P95 latency percentiles (500-request window).
- Active vs. queued background jobs.
- Recent failure records containing service name, error code, request ID, job ID, and timestamp.

---

## 6. Subsystem Health Checks
The `/api/v1/health` endpoint distinguishes granular subsystem health:
- `api`: Gateway and HTTP routing.
- `storage`: Cache directory read/write access.
- `job_queue`: Background worker availability.
- `model_registry`: Model checkpoint index readiness.

---

## 7. System Diagnostics Dashboard
Located at [`/settings`](../frontend/src/app/settings/page.tsx) with Stitch AI MCP UI/UX:
- Subsystem health status indicators.
- Live operational metrics (Requests, Error Rate, Avg/P95 Latency, Uptime).
- Active background jobs summary.
- Recent failure inspection modal with request correlation.
