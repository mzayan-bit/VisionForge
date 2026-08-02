"""Centralized Exception Handling Layer."""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from visionforge.core.responses import error_response

logger = logging.getLogger("visionforge.exceptions")


class VisionForgeException(Exception):  # noqa: N818
    """Base domain exception for VisionForge platform."""

    def __init__(
        self,
        message: str = "An internal platform error occurred",
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or []


class ResourceNotFoundException(VisionForgeException):
    """Raised when a requested domain resource is not found."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} with ID '{resource_id}' was not found",
            code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ValidationException(VisionForgeException):
    """Raised when payload or query parameters fail validation constraints."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


async def visionforge_exception_handler(
    request: Request, exc: VisionForgeException
) -> JSONResponse:
    """Handle custom VisionForge domain exceptions."""
    request_id = getattr(request.state, "request_id", None)
    meta = {"request_id": request_id} if request_id else None

    logger.warning(
        "VisionForgeException [%s]: %s (path=%s)",
        exc.code,
        exc.message,
        request.url.path,
    )
    payload = error_response(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        meta=meta,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle standard HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)
    meta = {"request_id": request_id} if request_id else None

    code_map = {
        404: "NOT_FOUND",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        405: "METHOD_NOT_ALLOWED",
    }
    error_code = code_map.get(exc.status_code, "HTTP_ERROR")

    payload = error_response(
        code=error_code,
        message=str(exc.detail),
        meta=meta,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic payload validation errors."""
    request_id = getattr(request.state, "request_id", None)
    meta = {"request_id": request_id} if request_id else None

    formatted_errors = []
    for err in exc.errors():
        loc = " -> ".join(str(loc_part) for loc_part in err.get("loc", []))
        formatted_errors.append(
            {
                "location": loc,
                "message": err.get("msg", "Invalid parameter"),
                "type": err.get("type", "value_error"),
            }
        )

    payload = error_response(
        code="VALIDATION_ERROR",
        message="Request validation failed",
        details=formatted_errors,
        meta=meta,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump()
    )


async def uncaught_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled server exceptions."""
    request_id = getattr(request.state, "request_id", None)
    meta = {"request_id": request_id} if request_id else None

    logger.exception("Uncaught server exception on %s %s", request.method, request.url.path)

    payload = error_response(
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred. Please contact system administrator.",
        meta=meta,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all centralized exception handlers onto the FastAPI instance."""
    app.add_exception_handler(VisionForgeException, visionforge_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, uncaught_exception_handler)
