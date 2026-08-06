"""VisionForge Visual Memory Core Index and Persistence System."""

import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException

logger = logging.getLogger("visionforge.memory.index")


class MemoryRecordNotFoundError(VisionForgeException):
    """Raised when looking up a record ID that does not exist in VisualMemoryIndex."""

    def __init__(self, record_id: str):
        super().__init__(
            message=f"Memory record '{record_id}' was not found in VisualMemoryIndex",
            code="MEMORY_RECORD_NOT_FOUND",
            status_code=404,
        )


class VisualMemoryRecord(BaseModel):
    """Single indexed record in Visual Memory containing vector and metadata."""

    id: str = Field(description="Unique record identifier")
    embedding: list[float] = Field(description="Dense vector embedding")
    dimension: int = Field(default=768, description="Vector dimension")
    image_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Associated image parameters"
    )
    tags: list[str] = Field(default_factory=list, description="Optional search classification tags")
    indexed_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 indexing timestamp",
    )


class VisualMemoryStats(BaseModel):
    """Telemetry status and metrics for Visual Memory Store."""

    total_records: int = Field(default=0, description="Total vectors indexed in memory")
    vector_dimension: int = Field(default=768, description="Standard vector dimensionality")
    memory_size_mb: float = Field(default=0.0, description="Estimated VRAM/RAM footprint in MB")
    storage_path: str = Field(description="Disk persistence path")
    last_saved_at: str | None = Field(default=None, description="ISO timestamp of last save")


class VisualMemoryIndex:
    """Production-grade in-memory vector store with NumPy matrix operations & disk persistence."""

    def __init__(self, storage_dir: str | None = None):
        raw_path = storage_dir or (Path(get_settings().model_cache_dir).parent / "memory")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._records: dict[str, VisualMemoryRecord] = {}
        self._matrix_cache: np.ndarray | None = None
        self._matrix_ids: list[str] = []
        self._last_saved_at: str | None = None

        # Auto-restore if saved index exists on disk
        self.load_from_disk()

    @property
    def storage_dir(self) -> Path:
        """Absolute storage directory path."""
        return self._storage_dir

    def add_record(self, record: VisualMemoryRecord) -> None:
        """Add or update a vector record in the index."""
        self._records[record.id] = record
        self._invalidate_matrix_cache()
        logger.info("Indexed visual memory record '%s' (dim=%d)", record.id, record.dimension)

    def get_record(self, record_id: str) -> VisualMemoryRecord:
        """Retrieve a record by unique ID."""
        if record_id not in self._records:
            raise MemoryRecordNotFoundError(record_id)
        return self._records[record_id]

    def delete_record(self, record_id: str) -> bool:
        """Delete a record by unique ID."""
        if record_id in self._records:
            del self._records[record_id]
            self._invalidate_matrix_cache()
            logger.info("Deleted visual memory record '%s'", record_id)
            return True
        return False

    def list_records(self, limit: int = 100, offset: int = 0) -> list[VisualMemoryRecord]:
        """Return paginated list of indexed records."""
        all_records = list(self._records.values())
        return all_records[offset : offset + limit]

    def clear(self) -> int:
        """Purge all records from memory index."""
        count = len(self._records)
        self._records.clear()
        self._invalidate_matrix_cache()
        logger.info("Cleared %d record(s) from VisualMemoryIndex", count)
        return count

    def get_matrix_and_ids(self) -> tuple[np.ndarray, list[str]]:
        """Return cached (N, D) float32 NumPy matrix and matching ID list for similarity search."""
        if self._matrix_cache is None or len(self._matrix_ids) != len(self._records):
            if not self._records:
                return np.empty((0, 768), dtype=np.float32), []

            self._matrix_ids = list(self._records.keys())
            vectors = [self._records[rid].embedding for rid in self._matrix_ids]
            self._matrix_cache = np.array(vectors, dtype=np.float32)

        return self._matrix_cache, self._matrix_ids

    def save_to_disk(self) -> None:
        """Persist current visual memory index to disk JSON."""
        index_file = self._storage_dir / "visual_memory_index.json"
        serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "records": [rec.model_dump() for rec in self._records.values()],
        }
        index_file.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
        self._last_saved_at = serializable["saved_at"]
        logger.info("Saved %d records to '%s'", len(self._records), index_file)

    def load_from_disk(self) -> None:
        """Restore visual memory index from disk JSON if available."""
        index_file = self._storage_dir / "visual_memory_index.json"
        if not index_file.is_file():
            return

        try:
            raw = json.loads(index_file.read_text(encoding="utf-8"))
            records_data = raw.get("records", [])
            self._records.clear()
            for item in records_data:
                rec = VisualMemoryRecord(**item)
                self._records[rec.id] = rec
            self._invalidate_matrix_cache()
            self._last_saved_at = raw.get("saved_at")
            logger.info("Restored %d visual memory records from disk", len(self._records))
        except Exception as exc:
            logger.error("Failed to restore visual memory from disk: %s", str(exc))

    def get_stats(self) -> VisualMemoryStats:
        """Return memory index statistics and resource footprint."""
        total = len(self._records)
        dim = 768
        if total > 0:
            first = next(iter(self._records.values()))
            dim = first.dimension

        # Estimate memory size in MB (float32 elements + overhead)
        bytes_est = total * dim * 4
        memory_mb = round(bytes_est / (1024 * 1024), 4)

        return VisualMemoryStats(
            total_records=total,
            vector_dimension=dim,
            memory_size_mb=memory_mb,
            storage_path=str(self._storage_dir),
            last_saved_at=self._last_saved_at,
        )

    def _invalidate_matrix_cache(self) -> None:
        self._matrix_cache = None
        self._matrix_ids.clear()


@lru_cache
def get_visual_memory_index() -> VisualMemoryIndex:
    """Return a cached singleton instance of VisualMemoryIndex."""
    return VisualMemoryIndex()
