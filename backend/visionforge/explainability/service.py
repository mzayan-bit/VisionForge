"""Explainability Service Layer for VisionForge.

Manages ExplanationRun execution, deterministic caching, human reviews,
researcher notes, and side-by-side comparative diagnostics.
"""

import hashlib
import json
import logging
import platform
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from visionforge.config import settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.explainability.generator import (
    UnsupportedExplanationError,
    generate_attribution_map,
)
from visionforge.explainability.schemas import (
    AddResearcherNoteRequest,
    CreateExplanationRequest,
    ExplanationComparison,
    ExplanationMethod,
    ExplanationRun,
    ExplanationStatus,
    ReviewExplanationRequest,
    ReviewRating,
)
from visionforge.inference.schemas import NormalizedBoundingBox, StandardPrediction

logger = logging.getLogger("visionforge.explainability.service")


class ExplanationNotFoundError(VisionForgeException):
    """Raised when an explanation run is not found."""

    def __init__(self, explanation_id: str):
        super().__init__(
            message=f"Explanation run '{explanation_id}' was not found.",
            code="EXPLANATION_NOT_FOUND",
            status_code=404,
        )


class ExplainabilityService:
    """Service to orchestrate model explainability, attribution caching, and review workflows."""

    def __init__(self):
        cache_root = Path(settings.model_cache_dir).expanduser().resolve()
        data_dir = cache_root.parent
        self._storage_dir = data_dir / "explanations"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._storage_dir / "explanation_runs.json"
        self._runs: dict[str, ExplanationRun] = {}
        self._cache: dict[str, str] = {}  # cache_key -> explanation_id

        self.load_from_disk()
        self._seed_default_explanations_if_empty()

    def _compute_cache_key(self, req: CreateExplanationRequest) -> str:
        """Compute deterministic SHA-256 cache key from request parameters."""
        raw = f"{req.model_id}:{req.model_version}:{req.sample_id}:{req.method.value}:{req.target_class}:{json.dumps(req.config.model_dump(), sort_keys=True)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def create_explanation(self, req: CreateExplanationRequest) -> ExplanationRun:
        """Generate a new visual attribution explanation or return a cached result."""
        cache_key = self._compute_cache_key(req)

        # Check cache hit
        if cache_key in self._cache:
            existing_id = self._cache[cache_key]
            if existing_id in self._runs:
                existing_run = self._runs[existing_id]
                logger.info("Serving cached explanation '%s' (Key: %s)", existing_id, cache_key[:8])
                existing_copy = existing_run.model_copy()
                existing_copy.cache_hit = True
                return existing_copy

        explanation_id = f"exp_{uuid.uuid4().hex[:8]}"
        is_correct = req.is_correct_prediction if req.is_correct_prediction is not None else True
        target_box = [0.20, 0.15, 0.80, 0.85]

        # Construct prediction descriptor
        pred_desc = StandardPrediction(
            prediction_id=f"pred_{explanation_id[4:]}",
            class_id=0,
            class_name=req.target_class,
            confidence=0.84 if is_correct else 0.58,
            bbox=NormalizedBoundingBox(
                x_center=(target_box[0] + target_box[2]) / 2.0,
                y_center=(target_box[1] + target_box[3]) / 2.0,
                width=target_box[2] - target_box[0],
                height=target_box[3] - target_box[1],
            ),
            model_id=req.model_id,
            model_version=req.model_version,
        )

        req.config.method = req.method
        try:
            artifact, summary = generate_attribution_map(
                model_id=req.model_id,
                target_class=req.target_class,
                target_box=target_box,
                config=req.config,
                is_correct=is_correct,
            )
            status = ExplanationStatus.COMPLETED
            err_msg = None
        except UnsupportedExplanationError as exc:
            artifact = None
            summary = ""
            status = ExplanationStatus.UNSUPPORTED
            err_msg = str(exc)
        except Exception as exc:
            artifact = None
            summary = ""
            status = ExplanationStatus.FAILED
            err_msg = f"Attribution generation error: {exc}"

        now = datetime.now(UTC).isoformat()
        run = ExplanationRun(
            explanation_id=explanation_id,
            model_id=req.model_id,
            model_version=req.model_version,
            inference_id=req.inference_id or f"inf_{uuid.uuid4().hex[:6]}",
            sample_id=req.sample_id,
            image_path=req.image_path or f"/datasets/safety_v2/images/test/{req.sample_id}.jpg",
            method=req.method,
            status=status,
            config=req.config,
            target_class=req.target_class,
            prediction=pred_desc,
            ground_truth_class=req.ground_truth_class or req.target_class,
            is_correct_prediction=is_correct,
            artifact=artifact,
            diagnostic_summary=summary,
            review_rating=ReviewRating.UNREVIEWED,
            created_at=now,
            completed_at=now if status == ExplanationStatus.COMPLETED else None,
            error_message=err_msg,
            cache_hit=False,
            environment={
                "os": platform.platform(),
                "python": platform.python_version(),
                "device": "cpu",
            },
        )

        self._runs[explanation_id] = run
        if status == ExplanationStatus.COMPLETED:
            self._cache[cache_key] = explanation_id

        self.save_to_disk()
        logger.info("Created explanation '%s' (Status: %s)", explanation_id, status.value)
        return run

    def get_explanation(self, explanation_id: str) -> ExplanationRun:
        """Retrieve explanation record by ID."""
        if explanation_id not in self._runs:
            raise ExplanationNotFoundError(explanation_id)
        return self._runs[explanation_id]

    def list_explanations(
        self,
        model_id: str | None = None,
        dataset_id: str | None = None,
        class_name: str | None = None,
        is_correct: bool | None = None,
        method: ExplanationMethod | None = None,
        review_status: ReviewRating | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ExplanationRun]:
        """List historical explanations with filtering."""
        items = list(self._runs.values())

        if model_id:
            items = [e for e in items if e.model_id == model_id]
        if dataset_id:
            items = [e for e in items if e.dataset_id == dataset_id]
        if class_name:
            items = [e for e in items if e.target_class == class_name]
        if is_correct is not None:
            items = [e for e in items if e.is_correct_prediction == is_correct]
        if method:
            items = [e for e in items if e.method == method]
        if review_status:
            items = [e for e in items if e.review_rating == review_status]

        items.sort(key=lambda x: x.created_at, reverse=True)
        return items[offset : offset + limit]

    def review_explanation(
        self, explanation_id: str, req: ReviewExplanationRequest
    ) -> ExplanationRun:
        """Record human researcher utility assessment on an explanation."""
        run = self.get_explanation(explanation_id)
        run.review_rating = req.rating
        if req.note:
            run.researcher_notes.append(f"[{req.rating.value}] {req.note}")
        self.save_to_disk()
        return run

    def add_researcher_note(
        self, explanation_id: str, req: AddResearcherNoteRequest
    ) -> ExplanationRun:
        """Append an observation note logged by a human researcher."""
        run = self.get_explanation(explanation_id)
        run.researcher_notes.append(req.note)
        self.save_to_disk()
        return run

    def compare_explanations(
        self, explanation_id_a: str, explanation_id_b: str
    ) -> ExplanationComparison:
        """Compute pixel-level attribution differences between two explanation runs."""
        exp_a = self.get_explanation(explanation_id_a)
        exp_b = self.get_explanation(explanation_id_b)

        if not exp_a.artifact or not exp_b.artifact:
            raise VisionForgeException(
                message="Cannot compare: one or both explanations do not have completed attribution artifacts.",
                code="INCOMPLETE_EXPLANATION_ARTIFACT",
                status_code=400,
            )

        grid_a = exp_a.artifact.heatmap_grid
        grid_b = exp_b.artifact.heatmap_grid
        h = min(len(grid_a), len(grid_b))
        w = min(len(grid_a[0]), len(grid_b[0]))

        diff_grid: list[list[float]] = []
        total_diff = 0.0

        for r in range(h):
            row_diff = []
            for c in range(w):
                d = abs(grid_a[r][c] - grid_b[r][c])
                row_diff.append(round(d, 4))
                total_diff += d
            diff_grid.append(row_diff)

        avg_diff = round(total_diff / (h * w), 4) if (h * w) > 0 else 0.0

        notes = [
            f"Mean absolute attribution difference between '{exp_a.sample_id}' and '{exp_b.sample_id}': {(avg_diff * 100):.1f}%.",
            f"Model A object concentration: {(exp_a.artifact.object_concentration_score * 100):.1f}%, Model B object concentration: {(exp_b.artifact.object_concentration_score * 100):.1f}%.",
        ]

        cmp_id = f"cmp_exp_{uuid.uuid4().hex[:8]}"
        return ExplanationComparison(
            comparison_id=cmp_id,
            explanation_a=exp_a,
            explanation_b=exp_b,
            attribution_difference_score=avg_diff,
            attribution_difference_grid=diff_grid,
            diagnostic_notes=notes,
        )

    # ─── Persistence & Seed Data ───────────────────────────────────────

    def save_to_disk(self) -> None:
        self._file.write_text(
            json.dumps([p.model_dump() for p in self._runs.values()], indent=2),
            encoding="utf-8",
        )

    def load_from_disk(self) -> None:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                for item in data:
                    run = ExplanationRun(**item)
                    self._runs[run.explanation_id] = run
            except Exception as e:
                logger.error("Failed to restore explanation runs: %s", e)

    def _seed_default_explanations_if_empty(self) -> None:
        if len(self._runs) > 0:
            return

        logger.info("Seeding default visual explanation runs...")
        # 1. Correct Prediction Explanation: Helmet
        self.create_explanation(
            CreateExplanationRequest(
                model_id="yolo11s.pt",
                model_version="1.0.0",
                sample_id="img_0007",
                target_class="helmet",
                method=ExplanationMethod.GRAD_CAM,
                is_correct_prediction=True,
                ground_truth_class="helmet",
            )
        )

        # 2. Incorrect Prediction Explanation (Failure): Vest misclassified as Shirt
        self.create_explanation(
            CreateExplanationRequest(
                model_id="yolo11s.pt",
                model_version="1.0.0",
                sample_id="img_0012",
                target_class="vest",
                method=ExplanationMethod.GRAD_CAM,
                is_correct_prediction=False,
                ground_truth_class="vest",
            )
        )


@lru_cache
def get_explainability_service() -> ExplainabilityService:
    """Return singleton instance of ExplainabilityService."""
    return ExplainabilityService()
