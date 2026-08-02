"""Unified API Response Models and Utilities."""

from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class APIErrorDetail(BaseModel):
    """Structured error detail representation."""

    code: str = Field(description="Machine-readable error identifier code")
    message: str = Field(description="Human-readable error description")
    details: list[dict[str, Any]] | None = Field(
        default=None, description="Optional extra contextual details or field errors"
    )


class APIResponse(BaseModel, Generic[DataT]):
    """Unified API response envelope returned by all backend endpoints."""

    success: bool = Field(default=True, description="Indicates operational success status")
    message: str = Field(default="Operation completed successfully", description="Status message")
    data: DataT | None = Field(default=None, description="Payload data returned on success")
    meta: dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": datetime.now(UTC).isoformat()},
        description="Response metadata including ISO timestamp",
    )
    error: APIErrorDetail | None = Field(
        default=None, description="Error detail payload populated on failure"
    )


def success_response(
    data: Any | None = None,
    message: str = "Operation completed successfully",
    meta: dict[str, Any] | None = None,
) -> APIResponse:
    """Construct a standard successful API response envelope."""
    response_meta = {"timestamp": datetime.now(UTC).isoformat()}
    if meta:
        response_meta.update(meta)

    return APIResponse(
        success=True,
        message=message,
        data=data,
        meta=response_meta,
        error=None,
    )


def error_response(
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> APIResponse:
    """Construct a standard error API response envelope."""
    response_meta = {"timestamp": datetime.now(UTC).isoformat()}
    if meta:
        response_meta.update(meta)

    return APIResponse(
        success=False,
        message=message,
        data=None,
        meta=response_meta,
        error=APIErrorDetail(code=code, message=message, details=details),
    )
