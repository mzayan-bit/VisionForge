"""VisionForge Operational Telemetry & System Diagnostics Collector."""

import time
from collections import deque
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field


class FailureRecord(BaseModel):
    """Structured record of a recent subsystem failure for observability."""

    failure_id: str
    timestamp: str
    service: str
    error_code: str
    message: str
    request_id: str | None = None
    job_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SystemDiagnosticsSnapshot(BaseModel):
    """System-level runtime diagnostics and performance telemetry."""

    timestamp: str
    uptime_seconds: float
    total_requests: int
    total_errors: int
    error_rate_pct: float
    avg_latency_ms: float
    p95_latency_ms: float
    active_jobs_count: int
    queued_jobs_count: int
    failed_jobs_count: int
    storage_healthy: bool
    recent_failures: list[FailureRecord]


class MetricsCollector:
    """Thread-safe collector for real system-level operational telemetry."""

    def __init__(self, max_recent_failures: int = 20, max_latencies: int = 500):
        self._lock = Lock()
        self._start_time = time.time()
        self._total_requests = 0
        self._total_errors = 0
        self._latencies: deque[float] = deque(maxlen=max_latencies)
        self._recent_failures: deque[FailureRecord] = deque(maxlen=max_recent_failures)
        self._active_jobs: set[str] = set()
        self._queued_jobs: set[str] = set()

    def record_request(self, duration_ms: float, is_error: bool = False) -> None:
        """Record an API request duration and status."""
        with self._lock:
            self._total_requests += 1
            if is_error:
                self._total_errors += 1
            self._latencies.append(duration_ms)

    def record_failure(
        self,
        service: str,
        error_code: str,
        message: str,
        request_id: str | None = None,
        job_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a failure event with structured context."""
        record = FailureRecord(
            failure_id=f"fail_{int(time.time() * 1000)}",
            timestamp=datetime.now(UTC).isoformat(),
            service=service,
            error_code=error_code,
            message=message,
            request_id=request_id,
            job_id=job_id,
            details=details or {},
        )
        with self._lock:
            self._total_errors += 1
            self._recent_failures.appendleft(record)

    def register_job(self, job_id: str, is_running: bool = False) -> None:
        """Track an active or queued background job."""
        with self._lock:
            if is_running:
                self._queued_jobs.discard(job_id)
                self._active_jobs.add(job_id)
            else:
                self._queued_jobs.add(job_id)

    def complete_job(self, job_id: str) -> None:
        """Remove a completed or failed job from tracking."""
        with self._lock:
            self._active_jobs.discard(job_id)
            self._queued_jobs.discard(job_id)

    def get_snapshot(self) -> SystemDiagnosticsSnapshot:
        """Compute live snapshot of operational telemetry."""
        with self._lock:
            uptime = time.time() - self._start_time
            reqs = self._total_requests
            errs = self._total_errors
            err_pct = (errs / reqs * 100) if reqs > 0 else 0.0

            lats = list(self._latencies)
            if lats:
                avg_lat = sum(lats) / len(lats)
                sorted_lats = sorted(lats)
                p95_idx = int(len(sorted_lats) * 0.95)
                p95_lat = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]
            else:
                avg_lat = 0.0
                p95_lat = 0.0

            return SystemDiagnosticsSnapshot(
                timestamp=datetime.now(UTC).isoformat(),
                uptime_seconds=uptime,
                total_requests=reqs,
                total_errors=errs,
                error_rate_pct=round(err_pct, 2),
                avg_latency_ms=round(avg_lat, 2),
                p95_latency_ms=round(p95_lat, 2),
                active_jobs_count=len(self._active_jobs),
                queued_jobs_count=len(self._queued_jobs),
                failed_jobs_count=len(self._recent_failures),
                storage_healthy=True,
                recent_failures=list(self._recent_failures),
            )


# Global singleton metrics collector
_METRICS_COLLECTOR = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Return singleton instance of MetricsCollector."""
    return _METRICS_COLLECTOR
