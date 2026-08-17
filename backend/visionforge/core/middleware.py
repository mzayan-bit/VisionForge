"""Middleware components for request tracing, timing, and security headers."""

import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from visionforge.core.config import VisionForgeSettings
from visionforge.core.telemetry import get_metrics_collector

logger = logging.getLogger("visionforge.request")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware attaching request IDs, measuring duration, and logging requests."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        # Extract or generate X-Request-ID header
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        collector = get_metrics_collector()
        is_error = False

        try:
            response = await call_next(request)
            if response.status_code >= 400:
                is_error = True
            return response
        except Exception as exc:
            is_error = True
            collector.record_failure(
                service="http_api",
                error_code="UNCAUGHT_HTTP_EXCEPTION",
                message=str(exc),
                request_id=request_id,
            )
            raise
        finally:
            process_time_ms = (time.perf_counter() - start_time) * 1000
            collector.record_request(duration_ms=process_time_ms, is_error=is_error)

            if "response" in locals():
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"
                logger.info(
                    "%s %s -> %d (%.2fms) [req_id=%s]",
                    request.method,
                    request.url.path,
                    response.status_code,
                    process_time_ms,
                    request_id,
                )


def register_middleware(app: FastAPI, settings: VisionForgeSettings) -> None:
    """Attach active middleware to FastAPI instance in proper order."""
    # 1. Tracing & Timing middleware
    app.add_middleware(RequestTracingMiddleware)

    # 2. CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
