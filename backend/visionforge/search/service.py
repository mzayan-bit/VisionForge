"""VisionForge Visual Search Service — Orchestration & Business Logic Layer.

Decouples visual search orchestration from FastAPI REST controllers.
Handles visual asset registration, frame/object/event search resolution,
history recording, and near-duplicate discovery.
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from visionforge.ai.schemas_embedding import ImageEmbeddingResult
from visionforge.ai.types import TaskType
from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.engine.runner import VisionEngine, get_vision_engine
from visionforge.memory.index import VisualMemoryIndex, VisualMemoryRecord, get_visual_memory_index
from visionforge.search.engine import (
    SearchResponsePayload,
    SimilarityMetric,
    VisualSearchEngine,
    get_visual_search_engine,
)
from visionforge.search.history import (
    SearchHistoryRecord,
    SearchHistoryStore,
    get_search_history_store,
)
from visionforge.search.schemas import (
    NearDuplicateResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
    VisualAsset,
    VisualAssetType,
)

logger = logging.getLogger("visionforge.search.service")


class VisualSearchServiceException(VisionForgeException):
    """Base exception for VisualSearchService errors."""

    def __init__(self, message: str, code: str = "SEARCH_SERVICE_ERROR", status_code: int = 400):
        super().__init__(message=message, code=code, status_code=status_code)


class VisualAssetNotFoundError(VisionForgeException):
    """Raised when looking up an asset ID that does not exist."""

    def __init__(self, asset_id: str):
        super().__init__(
            message=f"Visual asset '{asset_id}' was not found in Visual Search Index",
            code="ASSET_NOT_FOUND",
            status_code=404,
        )


class VisualSearchService:
    """Service layer orchestrating multi-modal visual search across images, frames, objects, moments, and dataset samples."""

    def __init__(
        self,
        search_engine: VisualSearchEngine | None = None,
        memory_index: VisualMemoryIndex | None = None,
        history_store: SearchHistoryStore | None = None,
        vision_engine: VisionEngine | None = None,
        storage_dir: Path | None = None,
    ):
        self._search_engine = search_engine or get_visual_search_engine()
        self._memory_index = memory_index or get_visual_memory_index()
        self._history_store = history_store or get_search_history_store()
        self._vision_engine = vision_engine or get_vision_engine()

        raw_path = storage_dir or (Path(get_settings().model_cache_dir).parent / "search")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._assets_file = self._storage_dir / "visual_assets_registry.json"

        self._assets: dict[str, VisualAsset] = {}
        self.load_assets_from_disk()

    # ─── Asset Registration & Ingestion ────────────────────────────────

    def register_asset(self, asset: VisualAsset) -> None:
        """Register or update a VisualAsset in the search catalog."""
        self._assets[asset.embedding_id] = asset
        self.save_assets_to_disk()

    def get_asset_by_embedding_id(self, embedding_id: str) -> VisualAsset | None:
        return self._assets.get(embedding_id)

    def list_assets(
        self,
        asset_type: VisualAssetType | None = None,
        video_id: str | None = None,
        dataset_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[VisualAsset]:
        """List registered searchable visual assets with filtering."""
        all_assets = list(self._assets.values())
        if asset_type:
            all_assets = [a for a in all_assets if a.asset_type == asset_type]
        if video_id:
            all_assets = [a for a in all_assets if a.source_video_id == video_id]
        if dataset_id:
            all_assets = [a for a in all_assets if a.source_dataset_id == dataset_id]
        return all_assets[offset : offset + limit]

    def index_video_run_assets(self, run_id: str) -> int:
        """Index video frames, track appearances, and event moments from a VideoInferenceRun into visual memory."""
        from visionforge.events.service import get_temporal_event_service
        from visionforge.video.service import get_video_intelligence_service

        video_svc = get_video_intelligence_service()
        event_svc = get_temporal_event_service()

        run = video_svc.get_run(run_id)
        events = event_svc.get_events_for_run(run_id)

        indexed_count = 0

        # 1. Index sampled video frames
        for t_step in range(int(run.duration_sec)):
            rec_id = f"vframe_{run.video_id}_{t_step:03d}"
            # Deterministic synthetic embedding for video frame if not real image bytes
            vec = self._generate_deterministic_vector(f"{run.video_id}_frame_{t_step}")
            mem_rec = VisualMemoryRecord(
                id=rec_id,
                embedding=vec,
                dimension=768,
                image_metadata={
                    "asset_type": VisualAssetType.FRAME.value,
                    "video_id": run.video_id,
                    "run_id": run.run_id,
                    "timestamp_sec": float(t_step),
                    "frame_idx": t_step * 30,
                    "title": f"Frame @ {t_step}s ({run.video_id})",
                },
                tags=["video_frame", run.video_id],
            )
            self._memory_index.add_record(mem_rec)
            self.register_asset(
                VisualAsset(
                    asset_id=f"asset_{rec_id}",
                    asset_type=VisualAssetType.FRAME,
                    title=f"Frame @ {t_step}s ({run.video_id})",
                    embedding_id=rec_id,
                    source_video_id=run.video_id,
                    source_run_id=run.run_id,
                    timestamp_sec=float(t_step),
                    frame_idx=t_step * 30,
                    metadata={"video_id": run.video_id, "fps": 30.0},
                )
            )
            indexed_count += 1

        # 2. Index track object appearances
        for track in run.tracks:
            rec_id = f"vobj_{run.run_id}_tr{track.track_id}"
            vec = self._generate_deterministic_vector(f"{track.class_name}_tr{track.track_id}")
            mid_pt = track.trajectory[len(track.trajectory) // 2] if track.trajectory else None
            mem_rec = VisualMemoryRecord(
                id=rec_id,
                embedding=vec,
                dimension=768,
                image_metadata={
                    "asset_type": VisualAssetType.OBJECT_CROP.value,
                    "video_id": run.video_id,
                    "run_id": run.run_id,
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "timestamp_sec": track.first_timestamp_sec,
                    "bbox": mid_pt.bbox if mid_pt else [100.0, 100.0, 200.0, 200.0],
                    "title": f"Track #{track.track_id} ({track.class_name})",
                },
                tags=["object_crop", track.class_name, f"track_{track.track_id}"],
            )
            self._memory_index.add_record(mem_rec)
            self.register_asset(
                VisualAsset(
                    asset_id=f"asset_{rec_id}",
                    asset_type=VisualAssetType.OBJECT_CROP,
                    title=f"Track #{track.track_id} ({track.class_name})",
                    embedding_id=rec_id,
                    source_video_id=run.video_id,
                    source_run_id=run.run_id,
                    track_id=track.track_id,
                    class_name=track.class_name,
                    timestamp_sec=track.first_timestamp_sec,
                    bbox=mid_pt.bbox if mid_pt else [100.0, 100.0, 200.0, 200.0],
                    metadata={"avg_speed": track.avg_speed_px_per_sec, "confidence": track.avg_confidence},
                )
            )
            indexed_count += 1

        # 3. Index event moments
        for evt in events:
            rec_id = f"vevt_{evt.event_id}"
            vec = self._generate_deterministic_vector(f"event_{evt.event_type.value}_{evt.event_id}")
            mem_rec = VisualMemoryRecord(
                id=rec_id,
                embedding=vec,
                dimension=768,
                image_metadata={
                    "asset_type": VisualAssetType.EVENT_FRAME.value,
                    "video_id": evt.video_id,
                    "run_id": evt.run_id,
                    "event_id": evt.event_id,
                    "event_type": evt.event_type.value,
                    "timestamp_sec": evt.start_timestamp_sec,
                    "title": f"Event: {evt.event_type.value} @ {evt.start_timestamp_sec:.1f}s",
                },
                tags=["event_moment", evt.event_type.value],
            )
            self._memory_index.add_record(mem_rec)
            self.register_asset(
                VisualAsset(
                    asset_id=f"asset_{rec_id}",
                    asset_type=VisualAssetType.EVENT_FRAME,
                    title=f"Event: {evt.event_type.value} @ {evt.start_timestamp_sec:.1f}s",
                    embedding_id=rec_id,
                    source_video_id=evt.video_id,
                    source_run_id=evt.run_id,
                    source_event_id=evt.event_id,
                    timestamp_sec=evt.start_timestamp_sec,
                    metadata={"event_type": evt.event_type.value, "description": evt.description},
                )
            )
            indexed_count += 1

        self._memory_index.save_to_disk()
        logger.info("Indexed %d visual assets from run '%s'", indexed_count, run_id)
        return indexed_count

    # ─── Unified Visual Search ─────────────────────────────────────────

    async def search_unified(
        self,
        req: UnifiedSearchRequest,
        image_bytes: bytes | None = None,
    ) -> UnifiedSearchResponse:
        """Execute unified visual search across images, frames, object crops, moments, or dataset samples."""
        t_start = time.perf_counter()
        search_id = f"usrch_{uuid.uuid4().hex[:10]}"

        query_vec: list[float]
        query_asset: VisualAsset | None = None
        query_summary = "Unified Visual Query"
        embed_time_ms = 0.0

        # 1. Resolve Query Vector
        if req.vector:
            query_vec = req.vector
            query_summary = f"Direct Vector ({len(query_vec)}D)"

        elif image_bytes or req.query_type == VisualAssetType.IMAGE:
            if not image_bytes:
                raise VisualSearchServiceException("Query type 'IMAGE' requires image file upload.")

            embed_start = time.perf_counter()
            res = await self._vision_engine.run_task(
                task_type=TaskType.RETRIEVAL,
                payload=image_bytes,
                model_name="siglip-base-patch16-224",
            )
            if not res.success or res.data is None:
                raise VisualSearchServiceException(f"Failed to generate query embedding: {res.error}")

            embedding_res: ImageEmbeddingResult = res.data
            query_vec = embedding_res.embedding
            embed_time_ms = (time.perf_counter() - embed_start) * 1000
            query_summary = f"Uploaded Image ({embedding_res.image_metadata.width}x{embedding_res.image_metadata.height})"

        elif req.asset_id:
            # Lookup existing asset
            matching = [a for a in self._assets.values() if a.asset_id == req.asset_id]
            if not matching:
                raise VisualAssetNotFoundError(req.asset_id)
            query_asset = matching[0]
            rec = self._memory_index.get_record(query_asset.embedding_id)
            query_vec = rec.embedding
            query_summary = f"Asset: {query_asset.title} ({query_asset.asset_type.value})"

        elif req.query_type == VisualAssetType.FRAME and req.video_id:
            ts = req.timestamp_sec or 0.0
            rec_id = f"vframe_{req.video_id}_{int(ts):03d}"
            try:
                rec = self._memory_index.get_record(rec_id)
                query_vec = rec.embedding
            except Exception:
                # Generate deterministic vector if not pre-indexed
                query_vec = self._generate_deterministic_vector(f"{req.video_id}_frame_{int(ts)}")
            query_summary = f"Frame @ {ts:.1f}s in video '{req.video_id}'"

        elif req.query_type == VisualAssetType.OBJECT_CROP and req.run_id and req.track_id is not None:
            rec_id = f"vobj_{req.run_id}_tr{req.track_id}"
            try:
                rec = self._memory_index.get_record(rec_id)
                query_vec = rec.embedding
            except Exception:
                query_vec = self._generate_deterministic_vector(f"obj_tr{req.track_id}")
            query_summary = f"Object Crop: Track #{req.track_id} in run '{req.run_id}'"

        elif req.query_type == VisualAssetType.EVENT_FRAME and req.event_id:
            rec_id = f"vevt_{req.event_id}"
            try:
                rec = self._memory_index.get_record(rec_id)
                query_vec = rec.embedding
            except Exception:
                query_vec = self._generate_deterministic_vector(f"event_{req.event_id}")
            query_summary = f"Event Moment: Event '{req.event_id}'"

        else:
            # Fallback to zero vector if empty index or unspecified
            query_vec = self._generate_deterministic_vector("fallback_query")
            query_summary = "General Visual Similarity"

        # 2. Execute Vector Similarity Search with Provenance Filtering
        results, candidate_count, search_time_ms, filter_time_ms = self._search_engine.search_unified_vectors(
            query_vector=query_vec,
            assets_map=self._assets,
            top_k=req.top_k,
            metric=req.metric,
            threshold=req.threshold,
            filter_asset_types=req.filter_asset_types,
            filter_dataset_id=req.filter_dataset_id,
            filter_video_id=req.filter_video_id,
            filter_class_name=req.filter_class_name,
            filter_event_type=req.filter_event_type,
            query_model="siglip-base-patch16-224",
        )

        total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        resp = UnifiedSearchResponse(
            search_id=search_id,
            timestamp=datetime.now(UTC).isoformat(),
            query_summary=query_summary,
            query_asset=query_asset,
            results=results,
            candidate_count=candidate_count,
            returned_count=len(results),
            metric_used=req.metric,
            model_used="siglip-base-patch16-224",
            embedding_time_ms=round(embed_time_ms, 2),
            search_time_ms=search_time_ms,
            filtering_time_ms=filter_time_ms,
            total_execution_time_ms=total_time_ms,
            explanation=f"Ranked {len(results)} matches by dense vector similarity (SigLIP-base-patch16-224). Metric: {req.metric.value.capitalize()}.",
        )

        self._record_unified_history(resp, req)
        return resp

    def find_near_duplicates(
        self,
        threshold: float = 0.95,
        filter_asset_type: VisualAssetType | None = None,
        filter_dataset_id: str | None = None,
    ) -> NearDuplicateResponse:
        """Find candidate near-duplicate visual assets in Visual Memory."""
        return self._search_engine.find_near_duplicates(
            assets_map=self._assets,
            threshold=threshold,
            filter_asset_type=filter_asset_type,
            filter_dataset_id=filter_dataset_id,
        )

    # ─── Backward-Compatible Wrappers ──────────────────────────────────

    async def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute visual search given query image file upload."""
        if not image_bytes:
            raise VisualSearchServiceException("Uploaded query image file is empty")

        response = await self._search_engine.search_by_image(
            image_bytes=image_bytes,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
        )
        self._record_history(response, query_type="image")
        return response

    def search_by_record(
        self,
        record_id: str,
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute visual search using an existing Visual Memory record ID as query."""
        record = self._memory_index.get_record(record_id)
        meta = record.image_metadata
        query_info = {
            "type": "memory_record",
            "query_record_id": record_id,
            "resolution": f"{meta.get('width', 0)}x{meta.get('height', 0)}",
            "tags": record.tags,
        }
        response = self._search_engine.search_by_vector(
            query_vector=record.embedding,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
            query_info=query_info,
            embedding_time_ms=0.0,
            model_used="siglip-base-patch16-224",
        )
        self._record_history(response, query_type="memory_record")
        return response

    def search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute visual search directly using a dense query vector."""
        query_info = {"type": "vector", "dimension": len(query_vector)}
        response = self._search_engine.search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
            query_info=query_info,
            embedding_time_ms=0.0,
            model_used="siglip-base-patch16-224",
        )
        self._record_history(response, query_type="vector")
        return response

    def get_search_history(
        self, limit: int = 50, offset: int = 0
    ) -> list[SearchHistoryRecord]:
        return self._history_store.get_history(limit=limit, offset=offset)

    def _record_history(self, payload: SearchResponsePayload, query_type: str) -> None:
        rec = SearchHistoryRecord(
            search_id=payload.search_id,
            timestamp=payload.timestamp,
            query_type=query_type,
            query_info=payload.query_info,
            model_used=payload.model_used,
            top_k=len(payload.results),
            threshold=0.0,
            metric_used=payload.metric_used.value,
            candidate_count=payload.candidate_count,
            returned_count=payload.returned_count,
            embedding_time_ms=payload.embedding_time_ms,
            search_time_ms=payload.search_time_ms,
            total_time_ms=payload.total_execution_time_ms,
        )
        self._history_store.record_search(rec)

    def _record_unified_history(self, payload: UnifiedSearchResponse, req: UnifiedSearchRequest) -> None:
        rec = SearchHistoryRecord(
            search_id=payload.search_id,
            timestamp=payload.timestamp,
            query_type=str(req.query_type),
            query_info={"summary": payload.query_summary, "filters": req.model_dump(exclude={"vector"})},
            model_used=payload.model_used,
            top_k=req.top_k,
            threshold=req.threshold,
            metric_used=payload.metric_used.value,
            candidate_count=payload.candidate_count,
            returned_count=payload.returned_count,
            embedding_time_ms=payload.embedding_time_ms,
            search_time_ms=payload.search_time_ms,
            total_time_ms=payload.total_execution_time_ms,
        )
        self._history_store.record_search(rec)

    # ─── Persistence & Helpers ────────────────────────────────────────

    def _generate_deterministic_vector(self, seed_str: str) -> list[float]:
        """Generate normalized 768D pseudo-random float vector for deterministic testing/fallback."""
        import hashlib

        import numpy as np

        seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        vec = rng.standard_normal(768).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return [float(x) for x in vec]

    def save_assets_to_disk(self) -> None:
        serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "assets": [a.model_dump() for a in self._assets.values()],
        }
        self._assets_file.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    def load_assets_from_disk(self) -> None:
        if self._assets_file.is_file():
            try:
                raw = json.loads(self._assets_file.read_text(encoding="utf-8"))
                for item in raw.get("assets", []):
                    asset = VisualAsset(**item)
                    self._assets[asset.embedding_id] = asset
                logger.info("Loaded %d visual assets from disk registry", len(self._assets))
            except Exception as exc:
                logger.warning("Failed to restore assets registry: %s", str(exc))


@lru_cache
def get_visual_search_service() -> VisualSearchService:
    """Return a cached singleton instance of VisualSearchService."""
    return VisualSearchService()
