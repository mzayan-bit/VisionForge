"""VisionForge Operational Telemetry, Metrics & Job Observability Collector."""

import time
from collections import deque
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Supported background job execution states."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobRecord(BaseModel):
    """Detailed metadata for an observed background workload."""

    job_id: str
    job_type: str = Field(description="Type of job e.g. training, evaluation, dataset_prep, video")
    name: str = Field(description="Human readable name of the job")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    progress_pct: float = 0.0
    error_code: str | None = None
    error_summary: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class CVOperationalMetrics(BaseModel):
    """Real-time operational computer vision telemetry metrics."""

    total_inferences: int = 0
    avg_inference_latency_ms: float = 0.0
    total_search_queries: int = 0
    avg_search_latency_ms: float = 0.0
    total_video_frames_processed: int = 0
    total_active_models_loaded: int = 0


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
    cv_metrics: CVOperationalMetrics
    recent_jobs: list[JobRecord]
    recent_failures: list[FailureRecord]


class MetricsCollector:
    """Thread-safe collector for real system-level operational telemetry and job history."""

    def __init__(
        self,
        max_recent_failures: int = 30,
        max_latencies: int = 500,
        max_recent_jobs: int = 50,
    ):
        self._lock = Lock()
        self._start_time = time.time()
        self._total_requests = 0
        self._total_errors = 0
        self._http_request_counts: dict[str, int] = {}
        self._latencies: deque[float] = deque(maxlen=max_latencies)
        self._recent_failures: deque[FailureRecord] = deque(maxlen=max_recent_failures)
        self._jobs: dict[str, JobRecord] = {}
        self._recent_jobs_order: deque[str] = deque(maxlen=max_recent_jobs)

        # CV Operational Telemetry
        self._total_inferences = 0
        self._inference_latencies: deque[float] = deque(maxlen=200)
        self._total_search_queries = 0
        self._search_latencies: deque[float] = deque(maxlen=200)
        self._total_video_frames = 0
        self._active_models_loaded = 0

    def record_request(
        self,
        duration_ms: float,
        is_error: bool = False,
        method: str = "GET",
        path: str = "/",
        status_code: int = 200,
    ) -> None:
        """Record an API request duration and status."""
        key = f"{method}_{status_code}"
        with self._lock:
            self._total_requests += 1
            if is_error or status_code >= 400:
                self._total_errors += 1
            self._http_request_counts[key] = self._http_request_counts.get(key, 0) + 1
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

    def record_inference(self, model_name: str, duration_ms: float) -> None:
        """Record a single computer vision inference execution."""
        with self._lock:
            self._total_inferences += 1
            self._inference_latencies.append(duration_ms)

    def record_search(self, duration_ms: float, result_count: int = 0) -> None:
        """Record visual vector memory search operation."""
        with self._lock:
            self._total_search_queries += 1
            self._search_latencies.append(duration_ms)

    def record_video_frames(self, frame_count: int, duration_ms: float = 0.0) -> None:
        """Record processed video frames."""
        with self._lock:
            self._total_video_frames += frame_count

    def set_active_models_count(self, count: int) -> None:
        """Update count of deep learning models loaded in memory."""
        with self._lock:
            self._active_models_loaded = max(0, count)

    # ─── Job Observability & Lifecycle ─────────────────────────────────────────

    def register_job(
        self,
        job_id: str,
        job_type: str = "generic",
        name: str = "Background Job",
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        is_running: bool = False,
    ) -> JobRecord:
        """Register a new background job into the observatory."""
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            name=name,
            status=JobStatus.RUNNING if is_running else JobStatus.QUEUED,
            started_at=datetime.now(UTC).isoformat() if is_running else None,
            request_id=request_id,
            metadata=metadata or {},
        )
        with self._lock:
            self._jobs[job_id] = record
            if job_id in self._recent_jobs_order:
                self._recent_jobs_order.remove(job_id)
            self._recent_jobs_order.appendleft(job_id)
        return record

    def start_job(self, job_id: str) -> None:
        """Transition a job to RUNNING state."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.RUNNING
                self._jobs[job_id].started_at = datetime.now(UTC).isoformat()

    def update_job_progress(self, job_id: str, progress_pct: float) -> None:
        """Update job progress percentage."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress_pct = min(100.0, max(0.0, progress_pct))

    def complete_job(self, job_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Mark a job as COMPLETED."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.progress_pct = 100.0
                job.completed_at = datetime.now(UTC).isoformat()
                if job.started_at:
                    try:
                        start = datetime.fromisoformat(job.started_at)
                        end = datetime.fromisoformat(job.completed_at)
                        job.duration_seconds = round((end - start).total_seconds(), 2)
                    except Exception:
                        job.duration_seconds = 0.0
                if metadata:
                    job.metadata.update(metadata)

    def fail_job(
        self,
        job_id: str,
        error_code: str,
        error_summary: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Mark a job as FAILED and record diagnostic failure event."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now(UTC).isoformat()
                job.error_code = error_code
                job.error_summary = error_summary
                if job.started_at:
                    try:
                        start = datetime.fromisoformat(job.started_at)
                        end = datetime.fromisoformat(job.completed_at)
                        job.duration_seconds = round((end - start).total_seconds(), 2)
                    except Exception:
                        job.duration_seconds = 0.0

        # Also record in failure stream
        self.record_failure(
            service=job.job_type if "job" in locals() and job else "background_job",
            error_code=error_code,
            message=error_summary,
            job_id=job_id,
            details=details,
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        """Retrieve metadata for a specific job."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        """Return list of recent jobs ordered newest first."""
        with self._lock:
            result = []
            for jid in list(self._recent_jobs_order)[:limit]:
                if jid in self._jobs:
                    result.append(self._jobs[jid])
            return result

    # ─── Snapshots & Metrics Exposition ───────────────────────────────────────

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

            active_cnt = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            queued_cnt = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
            failed_cnt = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)

            avg_inf_lat = (
                sum(self._inference_latencies) / len(self._inference_latencies)
                if self._inference_latencies
                else 0.0
            )
            avg_search_lat = (
                sum(self._search_latencies) / len(self._search_latencies)
                if self._search_latencies
                else 0.0
            )

            cv_metrics = CVOperationalMetrics(
                total_inferences=self._total_inferences,
                avg_inference_latency_ms=round(avg_inf_lat, 2),
                total_search_queries=self._total_search_queries,
                avg_search_latency_ms=round(avg_search_lat, 2),
                total_video_frames_processed=self._total_video_frames,
                total_active_models_loaded=self._active_models_loaded,
            )

            recent_jobs_list = [
                self._jobs[jid] for jid in self._recent_jobs_order if jid in self._jobs
            ]

            return SystemDiagnosticsSnapshot(
                timestamp=datetime.now(UTC).isoformat(),
                uptime_seconds=round(uptime, 2),
                total_requests=reqs,
                total_errors=errs,
                error_rate_pct=round(err_pct, 2),
                avg_latency_ms=round(avg_lat, 2),
                p95_latency_ms=round(p95_lat, 2),
                active_jobs_count=active_cnt,
                queued_jobs_count=queued_cnt,
                failed_jobs_count=failed_cnt,
                storage_healthy=True,
                cv_metrics=cv_metrics,
                recent_jobs=recent_jobs_list,
                recent_failures=list(self._recent_failures),
            )

    def export_prometheus_metrics(self) -> str:
        """Format and return real system metrics in Prometheus text exposition format."""
        snapshot = self.get_snapshot()
        lines = [
            "# HELP visionforge_uptime_seconds Total runtime of the VisionForge backend process.",
            "# TYPE visionforge_uptime_seconds gauge",
            f"visionforge_uptime_seconds {snapshot.uptime_seconds}",
            "",
            "# HELP visionforge_http_requests_total Total count of received HTTP API requests.",
            "# TYPE visionforge_http_requests_total counter",
            f"visionforge_http_requests_total {snapshot.total_requests}",
            "",
            "# HELP visionforge_http_errors_total Total count of failed HTTP API requests.",
            "# TYPE visionforge_http_errors_total counter",
            f"visionforge_http_errors_total {snapshot.total_errors}",
            "",
            "# HELP visionforge_http_latency_avg_ms Average HTTP request response latency in milliseconds.",
            "# TYPE visionforge_http_latency_avg_ms gauge",
            f"visionforge_http_latency_avg_ms {snapshot.avg_latency_ms}",
            "",
            "# HELP visionforge_http_latency_p95_ms P95 percentile HTTP request response latency in milliseconds.",
            "# TYPE visionforge_http_latency_p95_ms gauge",
            f"visionforge_http_latency_p95_ms {snapshot.p95_latency_ms}",
            "",
            "# HELP visionforge_jobs_active Current number of running background workloads.",
            "# TYPE visionforge_jobs_active gauge",
            f"visionforge_jobs_active {snapshot.active_jobs_count}",
            "",
            "# HELP visionforge_jobs_queued Current number of queued background workloads.",
            "# TYPE visionforge_jobs_queued gauge",
            f"visionforge_jobs_queued {snapshot.queued_jobs_count}",
            "",
            "# HELP visionforge_jobs_failed_total Total number of failed background workloads.",
            "# TYPE visionforge_jobs_failed_total counter",
            f"visionforge_jobs_failed_total {snapshot.failed_jobs_count}",
            "",
            "# HELP visionforge_cv_inferences_total Total count of deep learning inferences executed.",
            "# TYPE visionforge_cv_inferences_total counter",
            f"visionforge_cv_inferences_total {snapshot.cv_metrics.total_inferences}",
            "",
            "# HELP visionforge_cv_search_queries_total Total visual memory embedding search queries.",
            "# TYPE visionforge_cv_search_queries_total counter",
            f"visionforge_cv_search_queries_total {snapshot.cv_metrics.total_search_queries}",
            "",
            "# HELP visionforge_video_frames_processed_total Total video frames processed in temporal pipelines.",
            "# TYPE visionforge_video_frames_processed_total counter",
            f"visionforge_video_frames_processed_total {snapshot.cv_metrics.total_video_frames_processed}",
            "",
            "# HELP visionforge_models_loaded Current loaded deep learning models in memory cache.",
            "# TYPE visionforge_models_loaded gauge",
            f"visionforge_models_loaded {snapshot.cv_metrics.total_active_models_loaded}",
        ]
        return "\n".join(lines) + "\n"


# Global singleton metrics collector
_METRICS_COLLECTOR = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Return singleton instance of MetricsCollector."""
    return _METRICS_COLLECTOR
