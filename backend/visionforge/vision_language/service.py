"""VisionForge Multimodal Vision-Language Service.

Orchestrates user questions, multi-turn contexts, query interpretation, domain execution,
grounded language synthesis, query history, and reproducibility.
"""

import hashlib
import json
import logging
import time
from functools import lru_cache
from pathlib import Path

from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.vision_language.executor import MultimodalQueryExecutor
from visionforge.vision_language.generator import GroundedVisionLanguageSynthesizer
from visionforge.vision_language.interpreter import MultimodalQueryInterpreter
from visionforge.vision_language.schemas import (
    MultimodalQueryStatus,
    MultiTurnContext,
    SuggestedQueryItem,
    VisionEvidenceItem,
    VisionQuery,
    VisionQueryHistoryItem,
    VisionQueryType,
)

logger = logging.getLogger("visionforge.vision_language.service")


class VisionQueryNotFoundError(VisionForgeException):
    """Raised when looking up a VisionQuery ID that does not exist."""

    def __init__(self, query_id: str):
        super().__init__(
            message=f"Vision query '{query_id}' was not found in history.",
            code="VISION_QUERY_NOT_FOUND",
            status_code=404,
        )


class VisionLanguageService:
    """Core service for grounded Multimodal Vision-Language interaction."""

    def __init__(self, storage_dir: Path | None = None):
        cache_root = Path(get_settings().model_cache_dir).expanduser().resolve()
        raw_path = storage_dir or (cache_root.parent / "vision_language")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._storage_dir / "multimodal_queries_history.json"

        self._interpreter = MultimodalQueryInterpreter()
        self._executor = MultimodalQueryExecutor()
        self._synthesizer = GroundedVisionLanguageSynthesizer()

        self._history: dict[str, VisionQuery] = {}
        self.load_from_disk()

    def ask(
        self,
        user_query: str,
        context: MultiTurnContext | None = None,
    ) -> VisionQuery:
        """Process user question through interpretation, execution, and grounded answer synthesis."""
        t_start = time.perf_counter()

        # 1. Parse question into structured VisionQuery
        query = self._interpreter.parse_query(
            query_text=user_query,
            context=context,
        )

        # If query is ambiguous or unsupported, return immediately
        if query.status in (MultimodalQueryStatus.AMBIGUOUS, MultimodalQueryStatus.UNSUPPORTED):
            query.execution_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            self._history[query.query_id] = query
            self.save_to_disk()
            return query

        # 2. Execute structured query through existing VisionForge domain systems
        exec_result, evidence = self._executor.execute(query)
        query.execution_result = exec_result
        query.evidence = evidence

        # Set status based on result count
        res_count = exec_result.get("count", len(evidence))
        if res_count == 0:
            query.status = MultimodalQueryStatus.NO_RESULTS

        # 3. Grounded Answer Synthesis
        answer, is_verified, discrepancies = self._synthesizer.synthesize_grounded_answer(
            query, exec_result, evidence
        )
        query.answer = answer
        query.grounding_verified = is_verified

        # 4. Generate deterministic reproducibility hash
        raw_hash_input = f"{query.query_type.value}:{json.dumps(query.structured_query, sort_keys=True)}:{res_count}"
        query.reproducibility_hash = hashlib.sha256(raw_hash_input.encode("utf-8")).hexdigest()[:16]

        query.execution_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        self._history[query.query_id] = query
        self.save_to_disk()
        return query

    def get_query(self, query_id: str) -> VisionQuery:
        """Retrieve stored query record by ID."""
        if query_id not in self._history:
            raise VisionQueryNotFoundError(query_id)
        return self._history[query_id]

    def list_history(self, limit: int = 50) -> list[VisionQueryHistoryItem]:
        """List historical query summary records."""
        items = []
        for q in sorted(self._history.values(), key=lambda x: x.created_timestamp, reverse=True):
            cnt = q.execution_result.get("count", len(q.evidence))
            items.append(
                VisionQueryHistoryItem(
                    query_id=q.query_id,
                    user_query=q.user_query,
                    query_type=q.query_type,
                    status=q.status,
                    results_count=cnt,
                    created_timestamp=q.created_timestamp,
                    execution_time_ms=q.execution_time_ms,
                )
            )
        return items[:limit]

    def replay_query(self, query_id: str) -> VisionQuery:
        """Replay and re-verify a historical query."""
        old_query = self.get_query(query_id)
        return self.ask(user_query=old_query.user_query)

    def get_evidence(self, query_id: str) -> list[VisionEvidenceItem]:
        """Retrieve evidence list for a query."""
        query = self.get_query(query_id)
        return query.evidence

    def get_suggested_queries(self, page_context: str = "global") -> list[SuggestedQueryItem]:
        """Retrieve context-aware suggested query prompts."""
        suggestions = {
            "failure_gallery": [
                SuggestedQueryItem(
                    text="Show helmet failures with confidence below 0.50",
                    query_type=VisionQueryType.FAILURE_QUERY,
                    page_context="failure_gallery",
                ),
                SuggestedQueryItem(
                    text="Why is sample 1024 considered a failure?",
                    query_type=VisionQueryType.FAILURE_QUERY,
                    page_context="failure_gallery",
                ),
                SuggestedQueryItem(
                    text="Which class has the most false positives?",
                    query_type=VisionQueryType.FAILURE_QUERY,
                    page_context="failure_gallery",
                ),
            ],
            "video_lab": [
                SuggestedQueryItem(
                    text="Which objects entered Zone A?",
                    query_type=VisionQueryType.EVENT_QUERY,
                    page_context="video_lab",
                ),
                SuggestedQueryItem(
                    text="Which person stayed longer than 3 seconds?",
                    query_type=VisionQueryType.EVENT_QUERY,
                    page_context="video_lab",
                ),
                SuggestedQueryItem(
                    text="Show trajectory and velocity for Track 1",
                    query_type=VisionQueryType.TRACK_QUERY,
                    page_context="video_lab",
                ),
            ],
            "datasets": [
                SuggestedQueryItem(
                    text="How many samples are in this dataset?",
                    query_type=VisionQueryType.DATASET_QUERY,
                    page_context="datasets",
                ),
                SuggestedQueryItem(
                    text="Show underrepresented classes",
                    query_type=VisionQueryType.DATASET_QUERY,
                    page_context="datasets",
                ),
                SuggestedQueryItem(
                    text="Show duplicate candidates and quality issues",
                    query_type=VisionQueryType.DATASET_QUERY,
                    page_context="datasets",
                ),
            ],
            "evaluation": [
                SuggestedQueryItem(
                    text="Compare model v7 and v8",
                    query_type=VisionQueryType.MODEL_QUERY,
                    page_context="evaluation",
                ),
                SuggestedQueryItem(
                    text="Which model performs best on the benchmark suite?",
                    query_type=VisionQueryType.EVALUATION_QUERY,
                    page_context="evaluation",
                ),
                SuggestedQueryItem(
                    text="What changed between these two model evaluations?",
                    query_type=VisionQueryType.EVALUATION_QUERY,
                    page_context="evaluation",
                ),
            ],
            "global": [
                SuggestedQueryItem(
                    text="Show helmet failures",
                    query_type=VisionQueryType.FAILURE_QUERY,
                    page_context="global",
                ),
                SuggestedQueryItem(
                    text="Which people entered Zone A?",
                    query_type=VisionQueryType.EVENT_QUERY,
                    page_context="global",
                ),
                SuggestedQueryItem(
                    text="Compare model v7 and v8",
                    query_type=VisionQueryType.MODEL_QUERY,
                    page_context="global",
                ),
                SuggestedQueryItem(
                    text="Show underrepresented classes in safety_v2",
                    query_type=VisionQueryType.DATASET_QUERY,
                    page_context="global",
                ),
            ],
        }
        return suggestions.get(page_context, suggestions["global"])

    def save_to_disk(self) -> None:
        try:
            raw_data = [q.model_dump() for q in self._history.values()]
            self._history_file.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed saving query history: %s", exc)

    def load_from_disk(self) -> None:
        if self._history_file.exists():
            try:
                data = json.loads(self._history_file.read_text(encoding="utf-8"))
                for item in data:
                    q = VisionQuery(**item)
                    self._history[q.query_id] = q
            except Exception as exc:
                logger.warning("Failed loading query history: %s", exc)


@lru_cache
def get_vision_language_service() -> VisionLanguageService:
    """Singleton getter for VisionLanguageService."""
    return VisionLanguageService()
