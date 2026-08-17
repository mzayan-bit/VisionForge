"""VisionForge Multimodal Vision-Language REST API Routes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.vision_language.schemas import (
    MultiTurnContext,
    SuggestedQueryItem,
    VisionEvidenceItem,
    VisionQuery,
    VisionQueryHistoryItem,
)
from visionforge.vision_language.service import (
    VisionQueryNotFoundError,
    get_vision_language_service,
)

logger = logging.getLogger("visionforge.api.v1.multimodal")

router = APIRouter(prefix="/multimodal", tags=["Multimodal Vision-Language"])


class AskVisionQueryRequest(BaseModel):
    query: str = Field(description="User natural language question")
    context: MultiTurnContext | None = Field(
        default=None, description="Active conversational context"
    )


@router.post(
    "/ask",
    response_model=VisionQuery,
    status_code=status.HTTP_200_OK,
    summary="Ask a multimodal vision-language question",
)
def ask_multimodal_query(payload: AskVisionQueryRequest) -> VisionQuery:
    """Interpret question, execute against domain services, enforce grounding, and synthesize answer with evidence."""
    service = get_vision_language_service()
    try:
        return service.ask(user_query=payload.query, context=payload.context)
    except Exception as exc:
        logger.error("Error handling multimodal query: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/queries/{query_id}",
    response_model=VisionQuery,
    summary="Get single vision query result detail",
)
def get_vision_query(query_id: str) -> VisionQuery:
    """Retrieve full query document and evidence references by query ID."""
    service = get_vision_language_service()
    try:
        return service.get_query(query_id)
    except VisionQueryNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/queries/{query_id}/replay",
    response_model=VisionQuery,
    summary="Replay a previous vision query",
)
def replay_vision_query(query_id: str) -> VisionQuery:
    """Replay and re-verify an existing historical vision query."""
    service = get_vision_language_service()
    try:
        return service.replay_query(query_id)
    except VisionQueryNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/history",
    response_model=list[VisionQueryHistoryItem],
    summary="List historical vision queries",
)
def list_vision_query_history(
    limit: int = Query(default=50, ge=1, le=200),
) -> list[VisionQueryHistoryItem]:
    """Retrieve summary history of past vision queries."""
    service = get_vision_language_service()
    return service.list_history(limit=limit)


@router.get(
    "/queries/{query_id}/evidence",
    response_model=list[VisionEvidenceItem],
    summary="Get visual evidence references for query",
)
def get_query_evidence(query_id: str) -> list[VisionEvidenceItem]:
    """Retrieve structured visual evidence references associated with a query."""
    service = get_vision_language_service()
    try:
        return service.get_evidence(query_id)
    except VisionQueryNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/suggested",
    response_model=list[SuggestedQueryItem],
    summary="Get context-aware suggested query prompts",
)
def get_suggested_queries(
    page_context: str = Query(default="global", description="Current UI page context"),
) -> list[SuggestedQueryItem]:
    """Retrieve helpful suggested questions tailored to the active workspace page."""
    service = get_vision_language_service()
    return service.get_suggested_queries(page_context=page_context)
