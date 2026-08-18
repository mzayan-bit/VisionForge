"""VisionForge Visual Query Layer API Routes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from visionforge.query.schemas import (
    QueryEvidenceItem,
    QueryHistoryItem,
    QueryResult,
    VisualQuery,
)
from visionforge.query.service import (
    QueryNotFoundError,
    QueryValidationError,
    get_visual_query_service,
)

logger = logging.getLogger("visionforge.api.v1.query")

router = APIRouter(prefix="/query", tags=["Visual Query Layer"])


class AskQuestionRequest(BaseModel):
    query_text: str = Field(default="", description="Natural language question string")
    question: str | None = Field(default=None, description="Alias for query_text")
    query: str | None = Field(default=None, description="Alias for query_text")
    run_id: str = Field(description="Target VideoInferenceRun ID")

    @property
    def prompt_text(self) -> str:
        return (self.query_text or self.question or self.query or "").strip()


class ExecuteStructuredQueryRequest(BaseModel):
    query: VisualQuery = Field(description="Structured VisualQuery DSL object")


@router.post(
    "/ask",
    response_model=QueryResult,
    status_code=status.HTTP_200_OK,
    summary="Ask a natural-language visual question",
)
def ask_question(payload: AskQuestionRequest) -> QueryResult:
    """Interpret natural language question into structured DSL, validate, execute against facts, and return evidence."""
    service = get_visual_query_service()
    try:
        return service.ask(text=payload.prompt_text, run_id=payload.run_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/execute",
    response_model=QueryResult,
    status_code=status.HTTP_200_OK,
    summary="Execute a structured VisualQuery DSL object directly",
)
def execute_structured_query(payload: ExecuteStructuredQueryRequest) -> QueryResult:
    """Execute pre-built structured VisualQuery DSL object directly."""
    service = get_visual_query_service()
    try:
        return service.execute_structured_query(payload.query)
    except QueryValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/history",
    response_model=list[QueryHistoryItem],
    summary="List historical query executions",
)
def list_query_history(limit: int = Query(default=50, ge=1, le=200)) -> list[QueryHistoryItem]:
    """Retrieve list of recent query execution history summary items."""
    service = get_visual_query_service()
    return service.list_history(limit=limit)


@router.get(
    "/{query_id}",
    response_model=QueryResult,
    summary="Get single query execution result detail",
)
def get_query_result(query_id: str) -> QueryResult:
    """Retrieve complete QueryResult document by query ID."""
    service = get_visual_query_service()
    try:
        return service.get_query_result(query_id)
    except QueryNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/rerun/{query_id}",
    response_model=QueryResult,
    summary="Re-run a historical query",
)
def rerun_query(query_id: str) -> QueryResult:
    """Re-run a historical query against its target run."""
    service = get_visual_query_service()
    try:
        return service.rerun_query(query_id)
    except QueryNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{query_id}/evidence",
    response_model=list[QueryEvidenceItem],
    summary="Get visual evidence links for a query result",
)
def get_query_evidence(query_id: str) -> list[QueryEvidenceItem]:
    """Retrieve visual evidence links for a query result."""
    service = get_visual_query_service()
    try:
        res = service.get_query_result(query_id)
        return res.evidence
    except QueryNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
