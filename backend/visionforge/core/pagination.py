"""Generic Pagination and Query Filtering Utilities."""

import math
from collections.abc import Sequence
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard pagination envelope for collection endpoints."""

    items: list[T] = Field(description="Paginated item collection")
    total: int = Field(description="Total number of items available matching query")
    page: int = Field(default=1, ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items returned per page")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether more items exist on subsequent pages")
    has_prev: bool = Field(description="Whether previous pages exist")


def paginate_sequence(
    items: Sequence[T], page: int = 1, page_size: int = 20
) -> PaginatedResponse[T]:
    """Slice in-memory sequence into a standardized PaginatedResponse."""
    page = max(1, page)
    page_size = min(max(1, page_size), 100)
    total = len(items)
    total_pages = max(1, math.ceil(total / page_size))

    start = (page - 1) * page_size
    end = start + page_size
    sliced = list(items[start:end])

    return PaginatedResponse(
        items=sliced,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )
