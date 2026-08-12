"""VisionForge Active Learning Service & Test-Set Protection Orchestrator."""

import json
import logging
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from visionforge.active_learning.loop import execute_active_learning_loop_iteration
from visionforge.active_learning.schemas import (
    ActiveLearningIteration,
    ActiveLearningRun,
    ReviewDecisionRequest,
    SelectionBiasReport,
    SelectionStrategy,
    SignalWeights,
    StrategyComparisonResult,
)
from visionforge.active_learning.selector import rank_candidate_samples
from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.datasets.service import (
    DatasetPreparationService,
    get_dataset_preparation_service,
)
from visionforge.memory.index import get_visual_memory_index

logger = logging.getLogger("visionforge.active_learning.service")


class TestSetProtectionError(VisionForgeException):
    """Raised when active learning candidate pool contains test set samples reserved for evaluation."""

    __test__ = False

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="TEST_SET_PROTECTION_VIOLATION",
            status_code=400,
        )


class ActiveLearningRunNotFoundError(VisionForgeException):
    """Raised when looking up a run ID that does not exist."""

    def __init__(self, run_id: str):
        super().__init__(
            message=f"Active learning run '{run_id}' was not found.",
            code="ACTIVE_LEARNING_RUN_NOT_FOUND",
            status_code=404,
        )


class ActiveLearningService:
    """Service orchestrating candidate pool validation, multi-signal ranking, and review decisions."""

    def __init__(
        self,
        dataset_service: DatasetPreparationService | None = None,
        storage_dir: Path | None = None,
    ):
        self._dataset_service = dataset_service or get_dataset_preparation_service()
        cache_root = Path(get_settings().model_cache_dir).expanduser().resolve()
        raw_path = storage_dir or (cache_root.parent / "active_learning")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._runs_file = self._storage_dir / "active_learning_runs.json"
        self._iterations_file = self._storage_dir / "active_learning_iterations.json"
        self._runs: dict[str, ActiveLearningRun] = {}
        self._iterations: dict[str, ActiveLearningIteration] = {}
        self.load_from_disk()

    # ─── Closed-Loop Retraining & Iterations ──────────────────────────

    def execute_loop(
        self, active_learning_run_id: str, new_version_tag: str | None = None
    ) -> ActiveLearningIteration:
        """Execute closed-loop retraining iteration and compute empirical metric delta on untouched test split."""
        run = self.get_run(active_learning_run_id)
        iteration = execute_active_learning_loop_iteration(run, new_version_tag)
        self._iterations[iteration.iteration_id] = iteration
        self.save_to_disk()
        return iteration

    def get_iteration(self, iteration_id: str) -> ActiveLearningIteration:
        if iteration_id not in self._iterations:
            raise ActiveLearningRunNotFoundError(f"Active learning iteration '{iteration_id}' not found.")
        return self._iterations[iteration_id]

    def list_iterations(self, limit: int = 50, offset: int = 0) -> list[ActiveLearningIteration]:
        all_iters = sorted(self._iterations.values(), key=lambda i: i.created_at, reverse=True)
        return all_iters[offset : offset + limit]

    # ─── Mandatory Test-Set Protection ───────────────────────────────

    def validate_candidate_pool(
        self, dataset_id: str, candidate_paths: list[str]
    ) -> tuple[list[str], int]:
        """Enforce strict Test-Set Protection preventing evaluation test samples from active learning selection."""
        valid_candidates: list[str] = []
        excluded_count = 0

        for path_str in candidate_paths:
            normalized = path_str.lower().replace("\\", "/")
            # Strict test split path indicator check
            if "/test/" in normalized or "split_test" in normalized or normalized.endswith("_test.jpg"):
                excluded_count += 1
                logger.warning(
                    "Test-Set Protection: Blocked test sample '%s' from active learning pool", path_str
                )
            else:
                valid_candidates.append(path_str)

        if not valid_candidates and candidate_paths:
            raise TestSetProtectionError(
                f"Candidate pool for dataset '{dataset_id}' contains ONLY reserved test-set samples ({excluded_count} excluded). Test set samples cannot be used for active learning."
            )

        return valid_candidates, excluded_count

    # ─── Active Learning Runs & Recommendations ───────────────────────

    def create_run(
        self,
        dataset_id: str,
        model_id: str = "yolo11s.pt",
        candidate_paths: list[str] | None = None,
        strategy: SelectionStrategy = SelectionStrategy.UNCERTAINTY_DIVERSITY,
        weights: SignalWeights | None = None,
        top_k: int = 25,
        experiment_id: str | None = None,
    ) -> ActiveLearningRun:
        """Create and execute an Active Learning sample selection run."""
        raw_candidates = candidate_paths or []
        if not raw_candidates:
            # Generate synthetic candidate pool paths if none provided
            raw_candidates = [
                f"/tmp/active_learning/pool/sample_{i:03d}.jpg" for i in range(1, 101)
            ]

        # 1. Enforce Test-Set Protection
        clean_candidates, excluded_test_count = self.validate_candidate_pool(
            dataset_id=dataset_id, candidate_paths=raw_candidates
        )

        pool_id = f"pool_{uuid.uuid4().hex[:8]}"
        run_id = f"al_run_{uuid.uuid4().hex[:10]}"
        cfg_weights = weights or SignalWeights()

        # 2. Build candidate data items with embeddings & predictions
        memory_index = get_visual_memory_index()
        ds_matrix, _ = memory_index.get_matrix_and_ids()

        candidate_items: list[dict[str, Any]] = []
        for i, c_path in enumerate(clean_candidates):
            # Deterministic mock predictions & 768-d embeddings for ranking
            mock_conf = round(0.35 + (0.55 * ((i * 17) % 100) / 100.0), 3)
            mock_emb = [0.01 * ((i * j + 7) % 50) for j in range(768)]
            candidate_items.append(
                {
                    "image_id": f"img_cand_{i+1:03d}",
                    "image_path": c_path,
                    "predictions": [{"class_name": "helmet", "confidence": mock_conf}],
                    "embedding": mock_emb,
                    "quality_score": 0.75,
                    "failure_score": 0.80 if i % 7 == 0 else 0.10,
                }
            )

        # 3. Rank Candidates via Multi-Signal Engine
        ranked_samples = rank_candidate_samples(
            candidate_data=candidate_items,
            dataset_matrix=ds_matrix,
            strategy=strategy,
            weights=cfg_weights,
            top_k=top_k,
        )

        run = ActiveLearningRun(
            run_id=run_id,
            experiment_id=experiment_id,
            model_id=model_id,
            dataset_id=dataset_id,
            candidate_pool_id=pool_id,
            strategy=strategy,
            weights=cfg_weights,
            top_k=top_k,
            selected_samples=ranked_samples,
        )

        self._runs[run_id] = run
        self.save_to_disk()
        logger.info(
            "Completed Active Learning run '%s': Selected %d samples (strategy=%s, %d test samples excluded)",
            run_id,
            len(ranked_samples),
            strategy,
            excluded_test_count,
        )
        return run

    def get_run(self, run_id: str) -> ActiveLearningRun:
        if run_id not in self._runs:
            raise ActiveLearningRunNotFoundError(run_id)
        return self._runs[run_id]

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[ActiveLearningRun]:
        all_runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return all_runs[offset : offset + limit]

    # ─── Human Review Queue ──────────────────────────────────────────

    def submit_review_decision(self, payload: ReviewDecisionRequest) -> ActiveLearningRun:
        """Submit a human review decision (accept/reject/skip/label) for a candidate sample."""
        run = self.get_run(payload.run_id)
        found = False

        for sample in run.selected_samples:
            if sample.image_id == payload.image_id:
                sample.review_status = payload.status
                if payload.notes:
                    sample.notes = payload.notes
                found = True
                break

        if not found:
            raise ActiveLearningRunNotFoundError(
                f"Sample '{payload.image_id}' not found in active learning run '{payload.run_id}'"
            )

        self.save_to_disk()
        logger.info(
            "Submitted review decision '%s' for sample '%s' in run '%s'",
            payload.status,
            payload.image_id,
            payload.run_id,
        )
        return run

    # ─── Selection Bias & Strategy Comparison ─────────────────────────

    def analyze_selection_bias(self, run_id: str) -> SelectionBiasReport:
        """Analyze potential selection bias in the recommended candidate set."""
        run = self.get_run(run_id)
        selected = run.selected_samples

        class_counts: dict[str, int] = {"helmet": 0, "person": 0, "vest": 0}
        quality_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}

        conf_values: list[float] = []

        for s in selected:
            class_counts["helmet"] += 1
            if s.signals.quality_score > 0.7:
                quality_counts["high"] += 1
            elif s.signals.quality_score > 0.4:
                quality_counts["medium"] += 1
            else:
                quality_counts["low"] += 1

            conf_values.append(1.0 - s.signals.uncertainty_score)

        conf_values.sort()
        q1 = conf_values[len(conf_values) // 4] if conf_values else 0.0
        q2 = conf_values[len(conf_values) // 2] if conf_values else 0.0
        q3 = conf_values[(3 * len(conf_values)) // 4] if conf_values else 0.0

        summary = (
            f"Selection Bias Report for Run '{run_id}' ({run.strategy}): Selected {len(selected)} samples. "
            f"Uncertainty distribution: 25th percentile={q1:.2f}, median={q2:.2f}, 75th percentile={q3:.2f}."
        )

        return SelectionBiasReport(
            run_id=run_id,
            strategy=run.strategy,
            total_selected=len(selected),
            class_distribution=class_counts,
            quality_distribution=quality_counts,
            confidence_distribution={"q1": round(q1, 2), "median": round(q2, 2), "q3": round(q3, 2)},
            bias_summary=summary,
        )

    def compare_strategies(
        self,
        dataset_id: str,
        model_id: str,
        strategy_a: SelectionStrategy,
        strategy_b: SelectionStrategy,
        top_k: int = 25,
    ) -> StrategyComparisonResult:
        """Compare sample selection recommendations between two active learning strategies."""
        run_a = self.create_run(
            dataset_id=dataset_id, model_id=model_id, strategy=strategy_a, top_k=top_k
        )
        run_b = self.create_run(
            dataset_id=dataset_id, model_id=model_id, strategy=strategy_b, top_k=top_k
        )

        ids_a = {s.image_id for s in run_a.selected_samples}
        ids_b = {s.image_id for s in run_b.selected_samples}

        overlap = len(ids_a.intersection(ids_b))
        unique_a = len(ids_a - ids_b)
        unique_b = len(ids_b - ids_a)

        u_a = sum(s.signals.uncertainty_score for s in run_a.selected_samples) / max(1, len(run_a.selected_samples))
        u_b = sum(s.signals.uncertainty_score for s in run_b.selected_samples) / max(1, len(run_b.selected_samples))

        d_a = sum(s.signals.diversity_score for s in run_a.selected_samples) / max(1, len(run_a.selected_samples))
        d_b = sum(s.signals.diversity_score for s in run_b.selected_samples) / max(1, len(run_b.selected_samples))

        notes = (
            f"Strategy Comparison between '{strategy_a}' and '{strategy_b}': Overlap of {overlap} samples. "
            f"Uncertainty Delta: {round(u_b - u_a, 3)}, Diversity Delta: {round(d_b - d_a, 3)}."
        )

        return StrategyComparisonResult(
            dataset_id=dataset_id,
            model_id=model_id,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            overlap_count=overlap,
            unique_a_count=unique_a,
            unique_b_count=unique_b,
            diversity_delta=round(d_b - d_a, 3),
            uncertainty_delta=round(u_b - u_a, 3),
            summary_notes=notes,
        )

    # ─── Persistence Helpers ──────────────────────────────────────────

    def save_to_disk(self) -> None:
        serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "runs": [r.model_dump() for r in self._runs.values()],
        }
        self._runs_file.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

        iters_serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "iterations": [it.model_dump() for it in self._iterations.values()],
        }
        self._iterations_file.write_text(
            json.dumps(iters_serializable, indent=2, default=str), encoding="utf-8"
        )

    def load_from_disk(self) -> None:
        if self._runs_file.is_file():
            try:
                raw = json.loads(self._runs_file.read_text(encoding="utf-8"))
                for item in raw.get("runs", []):
                    run = ActiveLearningRun(**item)
                    self._runs[run.run_id] = run
            except Exception as exc:
                logger.warning("Failed to restore active learning runs from disk: %s", str(exc))

        if self._iterations_file.is_file():
            try:
                raw_iters = json.loads(self._iterations_file.read_text(encoding="utf-8"))
                for item in raw_iters.get("iterations", []):
                    iter_obj = ActiveLearningIteration(**item)
                    self._iterations[iter_obj.iteration_id] = iter_obj
            except Exception as exc:
                logger.warning("Failed to restore active learning iterations from disk: %s", str(exc))


@lru_cache
def get_active_learning_service() -> ActiveLearningService:
    """Return singleton instance of ActiveLearningService."""
    return ActiveLearningService()
