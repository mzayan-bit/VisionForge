"""Dataset Preparation Pipeline Service Layer & History Store."""

import json
import logging
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.datasets.leakage import detect_data_leakage
from visionforge.datasets.manifest import (
    build_manifest,
    export_manifest_csv,
    materialize_prepared_dataset,
)
from visionforge.datasets.schemas import (
    DatasetPreparationManifest,
    PreparationRun,
    PreparationStatus,
    SampleRef,
    SplitConfig,
    SplitStats,
)
from visionforge.datasets.splitting import partition_dataset
from visionforge.datasets.validation import validate_dataset
from visionforge.memory.index import VisualMemoryIndex, get_visual_memory_index

logger = logging.getLogger("visionforge.datasets.service")


class PreparationRunNotFoundError(VisionForgeException):
    """Raised when a preparation run ID is not found."""

    def __init__(self, prep_id: str):
        super().__init__(
            message=f"Preparation run '{prep_id}' was not found.",
            code="PREPARATION_RUN_NOT_FOUND",
            status_code=404,
        )


class PreparationHistoryStore:
    """Thread-safe store for logging past dataset preparation runs."""

    def __init__(self, storage_dir: str | None = None):
        raw_path = storage_dir or (Path(get_settings().model_cache_dir).parent / "datasets")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._storage_dir / "preparation_history.json"

        self._runs: dict[str, PreparationRun] = {}
        self._manifests: dict[str, DatasetPreparationManifest] = {}
        self.load_from_disk()

    def add_run(self, run: PreparationRun, manifest: DatasetPreparationManifest | None = None) -> None:
        """Add or update preparation run."""
        self._runs[run.preparation_id] = run
        if manifest:
            self._manifests[run.preparation_id] = manifest
        self.save_to_disk()

    def get_run(self, prep_id: str) -> PreparationRun:
        """Retrieve run by ID."""
        if prep_id not in self._runs:
            raise PreparationRunNotFoundError(prep_id)
        return self._runs[prep_id]

    def get_manifest(self, prep_id: str) -> DatasetPreparationManifest | None:
        """Retrieve manifest by preparation ID."""
        return self._manifests.get(prep_id)

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[PreparationRun]:
        """Return paginated list of preparation runs."""
        all_runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return all_runs[offset : offset + limit]

    def save_to_disk(self) -> None:
        """Persist history runs to disk JSON."""
        serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "runs": [r.model_dump() for r in self._runs.values()],
        }
        self._history_file.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    def load_from_disk(self) -> None:
        """Restore history runs from disk JSON if available."""
        if not self._history_file.is_file():
            return
        try:
            raw = json.loads(self._history_file.read_text(encoding="utf-8"))
            for item in raw.get("runs", []):
                run = PreparationRun(**item)
                self._runs[run.preparation_id] = run
        except Exception as exc:
            logger.error("Failed to restore dataset preparation history: %s", str(exc))


class DatasetPreparationService:
    """Service layer orchestrating dataset validation, leakage checking, splitting, and manifest generation."""

    def __init__(
        self,
        memory_index: VisualMemoryIndex | None = None,
        history_store: PreparationHistoryStore | None = None,
    ):
        self._memory_index = memory_index or get_visual_memory_index()
        self._history_store = history_store or get_search_preparation_history_store()

    def create_preparation_run(
        self,
        dataset_id: str,
        dataset_version: str,
        split_config: SplitConfig,
    ) -> PreparationRun:
        """Execute full Dataset Preparation Pipeline."""
        prep_id = f"prep_{uuid.uuid4().hex[:10]}"
        run = PreparationRun(
            preparation_id=prep_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            status=PreparationStatus.CREATED,
            split_config=split_config,
        )

        try:
            # 1. State: VALIDATING
            run.status = PreparationStatus.VALIDATING
            records = self._memory_index.list_records(limit=100000)
            validation_report = validate_dataset(records)
            run.validation_report = validation_report

            if validation_report.status == "FAILED" and validation_report.valid_samples == 0:
                run.status = PreparationStatus.FAILED
                run.error_message = "Dataset validation failed: 0 valid samples available."
                self._history_store.add_run(run)
                return run

            # 2. State: SPLITTING & LEAKAGE CHECK
            run.status = PreparationStatus.SPLITTING
            leakage_findings, sample_to_leak_group = detect_data_leakage(records)
            run.leakage_findings = leakage_findings

            sample_refs, counts = partition_dataset(
                records=records,
                sample_to_leakage_group=sample_to_leak_group,
                config=split_config,
            )

            # 3. State: VERIFYING
            run.status = PreparationStatus.VERIFYING
            split_stats = self._compute_split_stats(sample_refs, counts)
            run.split_stats = split_stats

            # 4. State: MATERIALIZING
            run.status = PreparationStatus.MATERIALIZING
            manifest = build_manifest(
                preparation_id=prep_id,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                split_config=split_config,
                samples=sample_refs,
                counts=counts,
                leakage_findings=leakage_findings,
            )

            json_path = materialize_prepared_dataset(manifest)
            run.manifest_path = str(json_path)

            # 5. State: COMPLETED
            run.status = PreparationStatus.COMPLETED
            run.completed_at = datetime.now(UTC).isoformat()

            self._history_store.add_run(run, manifest)
            logger.info("Successfully completed dataset preparation run '%s'", prep_id)
            return run

        except Exception as exc:
            run.status = PreparationStatus.FAILED
            run.error_message = str(exc)
            self._history_store.add_run(run)
            logger.error("Dataset preparation run '%s' failed: %s", prep_id, str(exc))
            raise

    def get_run(self, prep_id: str) -> PreparationRun:
        """Get preparation run details."""
        return self._history_store.get_run(prep_id)

    def list_history(self, limit: int = 50, offset: int = 0) -> list[PreparationRun]:
        """List historical preparation runs."""
        return self._history_store.list_runs(limit=limit, offset=offset)

    def export_manifest(self, prep_id: str, fmt: str = "json") -> Any:
        """Export preparation manifest in JSON or CSV format."""
        manifest = self._history_store.get_manifest(prep_id)
        if not manifest:
            # Try reloading from materialized file
            run = self._history_store.get_run(prep_id)
            if run.manifest_path and Path(run.manifest_path).is_file():
                raw = json.loads(Path(run.manifest_path).read_text(encoding="utf-8"))
                manifest = DatasetPreparationManifest(**raw)

        if not manifest:
            raise PreparationRunNotFoundError(prep_id)

        if fmt == "csv":
            return export_manifest_csv(manifest)
        return manifest.model_dump()

    def _compute_split_stats(
        self, samples: list[SampleRef], counts: dict[str, int]
    ) -> dict[str, SplitStats]:
        """Compute summary statistics for each partition."""
        total = len(samples) if samples else 1
        stats_map: dict[str, SplitStats] = {}

        for split_name in ["train", "validation", "test"]:
            split_samples = [s for s in samples if s.split == split_name]
            cnt = counts.get(split_name, len(split_samples))

            format_dist: dict[str, int] = {}
            category_dist: dict[str, int] = {}

            for s in split_samples:
                fmt = s.image_metadata.get("format", "UNKNOWN")
                format_dist[fmt] = format_dist.get(fmt, 0) + 1

                for tag in s.tags:
                    category_dist[tag] = category_dist.get(tag, 0) + 1

            stats_map[split_name] = SplitStats(
                split_name=split_name,
                count=cnt,
                ratio=round(cnt / total, 4),
                format_distribution=format_dist,
                category_distribution=category_dist,
            )

        return stats_map


@lru_cache
def get_search_preparation_history_store() -> PreparationHistoryStore:
    """Return singleton instance of PreparationHistoryStore."""
    return PreparationHistoryStore()


@lru_cache
def get_dataset_preparation_service() -> DatasetPreparationService:
    """Return singleton instance of DatasetPreparationService."""
    return DatasetPreparationService()
