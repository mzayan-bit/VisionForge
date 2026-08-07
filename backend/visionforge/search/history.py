"""VisionForge Search History Telemetry and Audit System."""

import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from visionforge.core.config import get_settings

logger = logging.getLogger("visionforge.search.history")


class SearchHistoryRecord(BaseModel):
    """Execution telemetry log for a single visual similarity search query."""

    search_id: str = Field(description="Unique search transaction identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 query timestamp",
    )
    query_type: str = Field(description="Query modality ('image', 'memory_record', or 'vector')")
    query_info: dict[str, Any] = Field(
        default_factory=dict, description="Metadata summary of the query payload"
    )
    model_used: str = Field(description="Embedding model used for feature extraction")
    top_k: int = Field(description="Configured Top-K retrieval limit")
    threshold: float = Field(description="Configured minimum similarity threshold")
    metric_used: str = Field(description="Similarity distance metric used")
    candidate_count: int = Field(description="Total visual memory items evaluated")
    returned_count: int = Field(description="Total matches satisfying threshold cutoff")
    embedding_time_ms: float = Field(description="Embedding generation duration in milliseconds")
    search_time_ms: float = Field(description="Vector similarity search duration in milliseconds")
    total_time_ms: float = Field(description="Total end-to-end execution duration in milliseconds")


class SearchHistoryStore:
    """Thread-safe store for logging and auditing past visual search executions."""

    def __init__(self, storage_dir: str | None = None):
        raw_path = storage_dir or (Path(get_settings().model_cache_dir).parent / "history")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._storage_dir / "search_history.json"

        self._records: list[SearchHistoryRecord] = []
        self.load_from_disk()

    def record_search(self, record: SearchHistoryRecord) -> None:
        """Add a search execution record to history and sync to disk."""
        self._records.insert(0, record)  # Most recent first
        if len(self._records) > 500:  # Cap history log at 500 records
            self._records = self._records[:500]

        self.save_to_disk()
        logger.info(
            "Recorded search transaction '%s' (type=%s, returned=%d/%d, total_ms=%.2f)",
            record.search_id,
            record.query_type,
            record.returned_count,
            record.candidate_count,
            record.total_time_ms,
        )

    def get_history(self, limit: int = 50, offset: int = 0) -> list[SearchHistoryRecord]:
        """Return paginated list of historical search execution logs."""
        return self._records[offset : offset + limit]

    def clear_history(self) -> int:
        """Purge all search history logs."""
        count = len(self._records)
        self._records.clear()
        self.save_to_disk()
        logger.info("Purged %d records from SearchHistoryStore", count)
        return count

    def save_to_disk(self) -> None:
        """Persist search history logs to disk JSON."""
        serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "history": [rec.model_dump() for rec in self._records],
        }
        self._history_file.write_text(
            json.dumps(serializable, indent=2, default=str), encoding="utf-8"
        )

    def load_from_disk(self) -> None:
        """Restore search history logs from disk JSON if available."""
        if not self._history_file.is_file():
            return

        try:
            raw = json.loads(self._history_file.read_text(encoding="utf-8"))
            history_data = raw.get("history", [])
            self._records = [SearchHistoryRecord(**item) for item in history_data]
            logger.info("Restored %d search history records from disk", len(self._records))
        except Exception as exc:
            logger.error("Failed to restore search history from disk: %s", str(exc))


@lru_cache
def get_search_history_store() -> SearchHistoryStore:
    """Return a cached singleton instance of SearchHistoryStore."""
    return SearchHistoryStore()
