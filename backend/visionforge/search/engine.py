"""VisionForge Visual Search Engine — Fast Similarity Search Orchestrator."""

import logging
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from visionforge.ai.schemas_embedding import ImageEmbeddingResult
from visionforge.ai.types import TaskType
from visionforge.core.exceptions import VisionForgeException
from visionforge.engine.runner import VisionEngine, get_vision_engine
from visionforge.memory.index import VisualMemoryIndex, get_visual_memory_index
from visionforge.search.schemas import (
    NearDuplicatePair,
    NearDuplicateResponse,
    UnifiedSearchResultItem,
    VisualAsset,
    VisualAssetType,
)
from visionforge.search.similarity import (
    SimilarityMetric,
    compute_matrix_cosine_similarity,
    compute_matrix_euclidean_distance,
    validate_embedding_vector,
)

logger = logging.getLogger("visionforge.search.engine")


class EmbeddingModelMismatchError(VisionForgeException):
    """Raised when comparing embeddings generated from incompatible embedding models."""

    def __init__(self, query_model: str, target_model: str):
        super().__init__(
            message=f"Incompatible embedding spaces: query ({query_model}) vs target ({target_model})",
            code="EMBEDDING_MODEL_MISMATCH",
            status_code=400,
        )


class SearchResultItem(BaseModel):
    """Descriptor for a single ranked visual memory match."""

    rank: int = Field(description="1-indexed similarity rank order")
    id: str = Field(description="Record ID of the visual memory match")
    similarity_score: float = Field(
        description="Calculated similarity percentage score (0.0 to 1.0)"
    )
    distance: float = Field(description="Distance measurement value")
    image_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata of the matched image"
    )
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    indexed_at: str = Field(description="ISO timestamp when item was indexed")
    embedding_model: str = Field(
        default="siglip-base-patch16-224", description="Model used to encode candidate vector"
    )


class SearchResponsePayload(BaseModel):
    """Result payload returned by Visual Search Engine."""

    search_id: str = Field(description="Unique transaction ID for this search execution")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO query execution timestamp",
    )
    results: list[SearchResultItem] = Field(description="Ranked list of top-K visual matches")
    candidate_count: int = Field(description="Total visual memory items evaluated")
    returned_count: int = Field(description="Total matches satisfying threshold cutoff")
    metric_used: SimilarityMetric = Field(description="Distance metric used for ranking")
    model_used: str = Field(
        default="siglip-base-patch16-224", description="Embedding model used for query"
    )
    embedding_time_ms: float = Field(
        default=0.0, description="Embedding generation duration in milliseconds"
    )
    search_time_ms: float = Field(
        default=0.0, description="Vector similarity search duration in milliseconds"
    )
    total_execution_time_ms: float = Field(
        description="Total end-to-end search duration in milliseconds"
    )
    query_info: dict[str, Any] = Field(
        default_factory=dict, description="Metadata descriptor of the search query input"
    )


class VisualSearchEngine:
    """Production-grade similarity search engine operating over Visual Memory Store."""

    def __init__(
        self,
        memory_index: VisualMemoryIndex | None = None,
        vision_engine: VisionEngine | None = None,
    ):
        self._memory_index = memory_index or get_visual_memory_index()
        self._vision_engine = vision_engine or get_vision_engine()

    async def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
    ) -> SearchResponsePayload:
        """Execute visual search given an input query image file."""
        embed_start = time.perf_counter()

        # 1. Run query image through embedding pipeline
        res = await self._vision_engine.run_task(
            task_type=TaskType.RETRIEVAL,
            payload=image_bytes,
            model_name="siglip-base-patch16-224",
        )

        if not res.success or res.data is None:
            err = res.error.message if res.error else "Embedding extraction failed for query"
            raise RuntimeError(f"Visual search query embedding failed: {err}")

        embedding_res: ImageEmbeddingResult = res.data
        embedding_time_ms = (time.perf_counter() - embed_start) * 1000

        meta = embedding_res.image_metadata
        query_info = {
            "type": "image",
            "resolution": f"{meta.width}x{meta.height}",
            "format": meta.format,
            "file_size_bytes": meta.file_size_bytes,
        }

        return self.search_by_vector(
            query_vector=embedding_res.embedding,
            top_k=top_k,
            metric=metric,
            threshold=threshold,
            query_info=query_info,
            embedding_time_ms=round(embedding_time_ms, 2),
            model_used=embedding_res.model,
        )

    def search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
        query_info: dict[str, Any] | None = None,
        embedding_time_ms: float = 0.0,
        model_used: str = "siglip-base-patch16-224",
    ) -> SearchResponsePayload:
        """Execute high-speed vectorized similarity search over indexed visual memory."""
        start_time = time.perf_counter()
        search_id = f"srch_{uuid.uuid4().hex[:10]}"

        # Validate query vector
        q_arr = validate_embedding_vector(query_vector)

        # Validate Top-K parameters
        clean_top_k = max(1, min(top_k, 100))
        clean_threshold = max(0.0, min(threshold, 0.95))

        matrix, ids = self._memory_index.get_matrix_and_ids()
        candidate_count = len(ids)

        if candidate_count == 0:
            search_time_ms = (time.perf_counter() - start_time) * 1000
            total_time_ms = embedding_time_ms + search_time_ms
            return SearchResponsePayload(
                search_id=search_id,
                results=[],
                candidate_count=0,
                returned_count=0,
                metric_used=metric,
                model_used=model_used,
                embedding_time_ms=embedding_time_ms,
                search_time_ms=round(search_time_ms, 2),
                total_execution_time_ms=round(total_time_ms, 2),
                query_info=query_info or {"type": "vector", "dimension": len(q_arr)},
            )

        # Compute metric matrix scores
        if metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT):
            scores = compute_matrix_cosine_similarity(matrix, q_arr)
            distances = 1.0 - scores
        elif metric == SimilarityMetric.EUCLIDEAN:
            distances, scores = compute_matrix_euclidean_distance(matrix, q_arr)
        else:
            raise ValueError(f"Unsupported similarity metric '{metric}'")

        # Top-K ranking (sort indices by score descending)
        ranked_indices = np.argsort(-scores)

        items: list[SearchResultItem] = []
        rank_counter = 1
        for idx in ranked_indices:
            score_val = float(scores[idx])
            if score_val < clean_threshold:
                continue

            record_id = ids[idx]
            rec = self._memory_index.get_record(record_id)

            items.append(
                SearchResultItem(
                    rank=rank_counter,
                    id=rec.id,
                    similarity_score=round(score_val, 4),
                    distance=round(float(distances[idx]), 4),
                    image_metadata=rec.image_metadata,
                    tags=rec.tags,
                    indexed_at=rec.indexed_at,
                    embedding_model=model_used,
                )
            )
            rank_counter += 1

            if len(items) >= clean_top_k:
                break

        search_time_ms = (time.perf_counter() - start_time) * 1000
        total_time_ms = embedding_time_ms + search_time_ms

        return SearchResponsePayload(
            search_id=search_id,
            results=items,
            candidate_count=candidate_count,
            returned_count=len(items),
            metric_used=metric,
            model_used=model_used,
            embedding_time_ms=embedding_time_ms,
            search_time_ms=round(search_time_ms, 2),
            total_execution_time_ms=round(total_time_ms, 2),
            query_info=query_info or {"type": "vector", "dimension": len(q_arr)},
        )

    def search_unified_vectors(
        self,
        query_vector: list[float],
        assets_map: dict[str, VisualAsset],
        top_k: int = 10,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: float = 0.0,
        filter_asset_types: list[VisualAssetType] | None = None,
        filter_dataset_id: str | None = None,
        filter_video_id: str | None = None,
        filter_class_name: str | None = None,
        filter_event_type: str | None = None,
        query_model: str = "siglip-base-patch16-224",
    ) -> tuple[list[UnifiedSearchResultItem], int, float, float]:
        """Execute unified vector similarity search with metadata filtering and provenance links."""
        search_start = time.perf_counter()
        q_arr = validate_embedding_vector(query_vector)

        matrix, ids = self._memory_index.get_matrix_and_ids()
        candidate_count = len(ids)

        if candidate_count == 0:
            return [], 0, 0.0, 0.0

        # 1. Compute metric matrix similarity
        if metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT):
            scores = compute_matrix_cosine_similarity(matrix, q_arr)
            distances = 1.0 - scores
        elif metric == SimilarityMetric.EUCLIDEAN:
            distances, scores = compute_matrix_euclidean_distance(matrix, q_arr)
        else:
            raise ValueError(f"Unsupported similarity metric '{metric}'")

        search_time_ms = (time.perf_counter() - search_start) * 1000

        # 2. Ranking and filtering
        filter_start = time.perf_counter()
        ranked_indices = np.argsort(-scores)

        results: list[UnifiedSearchResultItem] = []
        rank_counter = 1

        for idx in ranked_indices:
            score_val = float(scores[idx])
            if score_val < threshold:
                continue

            rec_id = ids[idx]
            asset = assets_map.get(rec_id)
            if not asset:
                # Construct asset from memory record image_metadata if not explicitly registered
                rec = self._memory_index.get_record(rec_id)
                meta = rec.image_metadata
                a_type = VisualAssetType.IMAGE
                if "asset_type" in meta:
                    try:
                        a_type = VisualAssetType(meta["asset_type"])
                    except Exception:
                        a_type = VisualAssetType.IMAGE

                asset = VisualAsset(
                    asset_id=f"asset_{rec.id}",
                    asset_type=a_type,
                    title=meta.get("title", f"Visual Asset {rec.id[:8]}"),
                    embedding_id=rec.id,
                    embedding_model=rec.image_metadata.get("embedding_model", query_model),
                    source_video_id=meta.get("video_id"),
                    source_dataset_id=meta.get("dataset_id"),
                    source_run_id=meta.get("run_id"),
                    source_event_id=meta.get("event_id"),
                    timestamp_sec=meta.get("timestamp_sec"),
                    frame_idx=meta.get("frame_idx"),
                    track_id=meta.get("track_id"),
                    bbox=meta.get("bbox"),
                    class_name=meta.get("class_name"),
                    metadata=meta,
                    indexed_at=rec.indexed_at,
                )

            # Check Embedding Model Compatibility
            if asset.embedding_model != query_model:
                logger.warning(
                    "Skipping candidate '%s' due to embedding space mismatch (%s vs %s)",
                    asset.asset_id,
                    asset.embedding_model,
                    query_model,
                )
                continue

            # Apply Metadata Filters
            if filter_asset_types and asset.asset_type not in filter_asset_types:
                continue
            if filter_dataset_id and asset.source_dataset_id != filter_dataset_id:
                continue
            if filter_video_id and asset.source_video_id != filter_video_id:
                continue
            if filter_class_name and (not asset.class_name or asset.class_name.lower() != filter_class_name.lower()):
                continue
            if filter_event_type and asset.metadata.get("event_type") != filter_event_type:
                continue

            # Build Provenance Links
            action_link = "/search"
            if asset.asset_type == VisualAssetType.FRAME and asset.source_video_id:
                t_sec = asset.timestamp_sec or 0.0
                action_link = f"/video-lab?video={asset.source_video_id}&seek={t_sec:.1f}"
            elif asset.asset_type == VisualAssetType.OBJECT_CROP and asset.source_video_id:
                t_sec = asset.timestamp_sec or 0.0
                tr_id = asset.track_id or 1
                action_link = f"/video-lab?video={asset.source_video_id}&seek={t_sec:.1f}&track={tr_id}"
            elif asset.asset_type == VisualAssetType.EVENT_FRAME and asset.source_video_id:
                t_sec = asset.timestamp_sec or 0.0
                action_link = f"/video-lab?video={asset.source_video_id}&seek={t_sec:.1f}"
            elif asset.asset_type == VisualAssetType.DATASET_SAMPLE and asset.source_dataset_id:
                action_link = f"/datasets?id={asset.source_dataset_id}"

            evidence_notes = f"Matched {asset.asset_type.value} from {asset.source_video_id or asset.source_dataset_id or 'Visual Memory'} (similarity: {score_val:.2%})"

            source_trace = {
                "asset_id": asset.asset_id,
                "asset_type": asset.asset_type.value,
                "video_id": asset.source_video_id,
                "dataset_id": asset.source_dataset_id,
                "run_id": asset.source_run_id,
                "event_id": asset.source_event_id,
                "timestamp_sec": asset.timestamp_sec,
                "track_id": asset.track_id,
                "class_name": asset.class_name,
            }

            results.append(
                UnifiedSearchResultItem(
                    rank=rank_counter,
                    asset=asset,
                    similarity_score=round(score_val, 4),
                    distance=round(float(distances[idx]), 4),
                    source_traceability=source_trace,
                    action_link=action_link,
                    evidence_notes=evidence_notes,
                )
            )
            rank_counter += 1

            if len(results) >= top_k:
                break

        filter_time_ms = (time.perf_counter() - filter_start) * 1000
        return results, candidate_count, round(search_time_ms, 2), round(filter_time_ms, 2)

    def find_near_duplicates(
        self,
        assets_map: dict[str, VisualAsset],
        threshold: float = 0.95,
        filter_asset_type: VisualAssetType | None = None,
        filter_dataset_id: str | None = None,
    ) -> NearDuplicateResponse:
        """Discover candidate near-duplicate visual asset pairs based on pairwise cosine similarity matrix."""
        start_time = time.perf_counter()
        matrix, ids = self._memory_index.get_matrix_and_ids()
        n = len(ids)

        if n < 2:
            return NearDuplicateResponse(
                total_evaluated=n,
                duplicate_pairs_found=0,
                pairs=[],
                threshold_used=threshold,
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        # Normalize matrix rows for pairwise dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        norm_matrix = matrix / norms

        # Compute full pairwise similarity matrix S = M * M.T
        sim_matrix = np.dot(norm_matrix, norm_matrix.T)

        pairs: list[NearDuplicatePair] = []

        # Inspect upper triangle (i < j)
        for i in range(n):
            rec_id_a = ids[i]
            asset_a = assets_map.get(rec_id_a) or VisualAsset(
                asset_id=f"asset_{rec_id_a}",
                asset_type=VisualAssetType.IMAGE,
                title=f"Asset {rec_id_a[:8]}",
                embedding_id=rec_id_a,
            )

            if filter_asset_type and asset_a.asset_type != filter_asset_type:
                continue
            if filter_dataset_id and asset_a.source_dataset_id != filter_dataset_id:
                continue

            for j in range(i + 1, n):
                rec_id_b = ids[j]
                asset_b = assets_map.get(rec_id_b) or VisualAsset(
                    asset_id=f"asset_{rec_id_b}",
                    asset_type=VisualAssetType.IMAGE,
                    title=f"Asset {rec_id_b[:8]}",
                    embedding_id=rec_id_b,
                )

                if filter_asset_type and asset_b.asset_type != filter_asset_type:
                    continue
                if filter_dataset_id and asset_b.source_dataset_id != filter_dataset_id:
                    continue

                sim_val = float(sim_matrix[i, j])
                if sim_val >= threshold:
                    rec_notes = (
                        "High visual redundancy detected. Consider deduplicating or grouping in dataset balance."
                        if sim_val > 0.98
                        else "Near-identical visual appearance candidate."
                    )
                    pairs.append(
                        NearDuplicatePair(
                            asset_a=asset_a,
                            asset_b=asset_b,
                            similarity_score=round(sim_val, 4),
                            distance=round(1.0 - sim_val, 4),
                            recommendation=rec_notes,
                        )
                    )

        # Sort pairs descending by similarity
        pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        exec_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return NearDuplicateResponse(
            total_evaluated=n,
            duplicate_pairs_found=len(pairs),
            pairs=pairs,
            threshold_used=threshold,
            execution_time_ms=exec_ms,
        )


@lru_cache
def get_visual_search_engine() -> VisualSearchEngine:
    """Return a cached singleton instance of VisualSearchEngine."""
    return VisualSearchEngine()
