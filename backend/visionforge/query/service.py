"""VisionForge Visual Query Layer Service."""

import json
import logging
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.events.service import get_temporal_event_service
from visionforge.query.executor import QueryExecutor
from visionforge.query.interpreter import QueryInterpreter
from visionforge.query.schemas import (
    QueryHistoryItem,
    QueryResult,
    QueryStatus,
    VisualQuery,
)
from visionforge.query.validator import QueryValidationError, QueryValidator
from visionforge.video.service import get_video_intelligence_service

logger = logging.getLogger("visionforge.query.service")


class QueryNotFoundError(VisionForgeException):
    """Raised when looking up a historical query ID that does not exist."""

    def __init__(self, query_id: str):
        super().__init__(
            message=f"Visual query '{query_id}' was not found in history.",
            code="QUERY_NOT_FOUND",
            status_code=404,
        )


class VisualQueryService:
    """Service orchestrating natural language interpretation, validation, execution, and query history."""

    def __init__(self, storage_dir: Path | None = None):
        cache_root = Path(get_settings().model_cache_dir).expanduser().resolve()
        raw_path = storage_dir or (cache_root.parent / "queries")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._storage_dir / "queries_history.json"

        self._interpreter = QueryInterpreter()
        self._validator = QueryValidator()
        self._executor = QueryExecutor()

        self._history: dict[str, QueryResult] = {}
        self.load_from_disk()

    def ask(self, text: str, run_id: str) -> QueryResult:
        """End-to-end processing: Natural Language Question -> Interpretation -> Validation -> Execution -> QueryResult."""
        t_start = time.perf_counter()

        # 1. Fetch active region names for context
        video_svc = get_video_intelligence_service()
        event_svc = get_temporal_event_service()
        run = video_svc.get_run(run_id)
        regions = event_svc.list_regions(run.video_id)
        region_names = [r.name for r in regions]

        # 2. Interpretation
        interp = self._interpreter.parse_query(text, run_id, region_names)
        interp_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        if interp.status != QueryStatus.SUCCESS:
            res = QueryResult(
                query_id=interp.query.query_id,
                original_query=text,
                structured_query=interp.query,
                status=interp.status,
                result_type="EVENT_LIST",  # type: ignore
                records=[],
                summary=interp.explanation,
                evidence=[],
                interpretation_explanation=interp.explanation,
                interpretation_time_ms=interp_ms,
                execution_time_ms=0.0,
                total_query_time_ms=interp_ms,
                source_run_id=run_id,
                reproducibility_hash="",
            )
            self._history[res.query_id] = res
            self.save_to_disk()
            return res

        # 3. Validation
        try:
            self._validator.validate_query(interp.query)
        except QueryValidationError as exc:
            res = QueryResult(
                query_id=interp.query.query_id,
                original_query=text,
                structured_query=interp.query,
                status=QueryStatus.VALIDATION_ERROR,
                result_type="EVENT_LIST",  # type: ignore
                records=[],
                summary=str(exc),
                evidence=[],
                interpretation_explanation=interp.explanation,
                interpretation_time_ms=interp_ms,
                execution_time_ms=0.0,
                total_query_time_ms=interp_ms,
                source_run_id=run_id,
                reproducibility_hash="",
            )
            self._history[res.query_id] = res
            self.save_to_disk()
            return res

        # 4. Execution
        res = self._executor.execute_query(interp.query, text, interp.explanation, interp_ms)

        self._history[res.query_id] = res
        self.save_to_disk()
        logger.info(
            "Executed visual query '%s' (%s) -> %d records in %.1fms",
            res.query_id,
            text,
            len(res.records),
            res.total_query_time_ms,
        )
        return res

    def execute_structured_query(self, query: VisualQuery) -> QueryResult:
        """Execute pre-built structured VisualQuery DSL directly."""
        time.perf_counter()
        self._validator.validate_query(query)
        res = self._executor.execute_query(
            query=query,
            original_text=query.original_text or f"Structured Query {query.query_type.value}",
            interpretation_explanation=f"Direct Execution: {query.query_type.value}",
            interp_time_ms=0.0,
        )
        self._history[res.query_id] = res
        self.save_to_disk()
        return res

    def get_query_result(self, query_id: str) -> QueryResult:
        if query_id not in self._history:
            raise QueryNotFoundError(query_id)
        return self._history[query_id]

    def list_history(self, limit: int = 50) -> list[QueryHistoryItem]:
        """Return list of historical query summary items."""
        sorted_res = sorted(self._history.values(), key=lambda q: q.created_at, reverse=True)
        items = [
            QueryHistoryItem(
                query_id=q.query_id,
                original_query=q.original_query,
                query_type=q.structured_query.query_type,
                run_id=q.source_run_id,
                status=q.status,
                results_count=len(q.records),
                total_query_time_ms=q.total_query_time_ms,
                created_at=q.created_at,
            )
            for q in sorted_res
        ]
        return items[:limit]

    def rerun_query(self, query_id: str) -> QueryResult:
        """Re-run a previously executed query."""
        old_res = self.get_query_result(query_id)
        return self.ask(old_res.original_query, old_res.source_run_id)

    # ─── Persistence Helpers ──────────────────────────────────────────

    def save_to_disk(self) -> None:
        serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "history": [q.model_dump() for q in self._history.values()],
        }
        self._history_file.write_text(
            json.dumps(serializable, indent=2, default=str), encoding="utf-8"
        )

    def load_from_disk(self) -> None:
        if self._history_file.is_file():
            try:
                raw = json.loads(self._history_file.read_text(encoding="utf-8"))
                for item in raw.get("history", []):
                    res = QueryResult(**item)
                    self._history[res.query_id] = res
            except Exception as exc:
                logger.warning("Failed to restore query history from disk: %s", str(exc))


@lru_cache
def get_visual_query_service() -> VisualQueryService:
    """Return singleton instance of VisualQueryService."""
    return VisualQueryService()
