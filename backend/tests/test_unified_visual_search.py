"""Unit and Integration Tests for Unified Visual Search System."""

import pytest
from fastapi.testclient import TestClient

from visionforge.events.service import get_temporal_event_service
from visionforge.main import app
from visionforge.memory.index import VisualMemoryRecord
from visionforge.search.schemas import (
    UnifiedSearchRequest,
    VisualAsset,
    VisualAssetType,
)
from visionforge.search.service import (
    VisualSearchService,
    get_visual_search_service,
)
from visionforge.video.schemas import FrameSamplingConfig, Track, TrajectoryPoint, VideoInferenceRun
from visionforge.video.service import get_video_intelligence_service

client = TestClient(app)


def create_synthetic_indexed_environment(service: VisualSearchService) -> None:
    """Populate Visual Memory with known deterministic vectors for search testing."""
    mem_idx = service._memory_index

    # 1. Base query vector A
    vec_a = [0.1] * 768
    # 2. Highly similar vector B (near-duplicate, sim ~ 0.99)
    vec_b = [0.101] * 768
    # 3. Moderately similar vector C
    vec_c = [0.05] * 768
    # 4. Orthogonal / different vector D
    vec_d = [-0.1] * 768

    mem_idx.add_record(
        VisualMemoryRecord(
            id="rec_frame_01",
            embedding=vec_a,
            dimension=768,
            image_metadata={"asset_type": "FRAME", "video_id": "vid_test_01", "timestamp_sec": 2.0},
        )
    )
    service.register_asset(
        VisualAsset(
            asset_id="asset_rec_frame_01",
            asset_type=VisualAssetType.FRAME,
            title="Frame @ 2.0s",
            embedding_id="rec_frame_01",
            source_video_id="vid_test_01",
            timestamp_sec=2.0,
            metadata={"video_id": "vid_test_01"},
        )
    )

    mem_idx.add_record(
        VisualMemoryRecord(
            id="rec_frame_02_dup",
            embedding=vec_b,
            dimension=768,
            image_metadata={"asset_type": "FRAME", "video_id": "vid_test_01", "timestamp_sec": 2.1},
        )
    )
    service.register_asset(
        VisualAsset(
            asset_id="asset_rec_frame_02_dup",
            asset_type=VisualAssetType.FRAME,
            title="Frame @ 2.1s (Duplicate Candidate)",
            embedding_id="rec_frame_02_dup",
            source_video_id="vid_test_01",
            timestamp_sec=2.1,
            metadata={"video_id": "vid_test_01"},
        )
    )

    mem_idx.add_record(
        VisualMemoryRecord(
            id="rec_obj_person_01",
            embedding=vec_c,
            dimension=768,
            image_metadata={"asset_type": "OBJECT_CROP", "video_id": "vid_test_01", "class_name": "person"},
        )
    )
    service.register_asset(
        VisualAsset(
            asset_id="asset_rec_obj_person_01",
            asset_type=VisualAssetType.OBJECT_CROP,
            title="Person Track #1",
            embedding_id="rec_obj_person_01",
            source_video_id="vid_test_01",
            track_id=1,
            class_name="person",
            metadata={"class_name": "person"},
        )
    )

    mem_idx.add_record(
        VisualMemoryRecord(
            id="rec_ds_sample_01",
            embedding=vec_d,
            dimension=768,
            image_metadata={"asset_type": "DATASET_SAMPLE", "dataset_id": "ds_coco_01"},
        )
    )
    service.register_asset(
        VisualAsset(
            asset_id="asset_rec_ds_sample_01",
            asset_type=VisualAssetType.DATASET_SAMPLE,
            title="COCO Sample #402",
            embedding_id="rec_ds_sample_01",
            source_dataset_id="ds_coco_01",
            metadata={"dataset_id": "ds_coco_01"},
        )
    )


@pytest.mark.asyncio
async def test_unified_visual_search_and_ranking():
    """Verify unified search ranks similar assets and preserves source provenance."""
    service = get_visual_search_service()
    create_synthetic_indexed_environment(service)

    # 1. Search with vector A
    req = UnifiedSearchRequest(
        query_type="VECTOR",
        vector=[0.1] * 768,
        top_k=5,
        threshold=0.0,
    )
    resp = await service.search_unified(req)
    assert resp.returned_count >= 3
    # Top rank should have similarity ~ 1.0 (vector A itself)
    top_match = resp.results[0]
    assert top_match.similarity_score > 0.99
    assert top_match.asset.embedding_id in ("rec_frame_01", "rec_frame_02_dup")
    assert "source_traceability" in top_match.model_dump()


@pytest.mark.asyncio
async def test_filtering_by_asset_type_and_class():
    """Verify search properly filters candidates by asset type and class label."""
    service = get_visual_search_service()
    create_synthetic_indexed_environment(service)

    # Filter only OBJECT_CROP
    req_crop = UnifiedSearchRequest(
        query_type="VECTOR",
        vector=[0.1] * 768,
        filter_asset_types=[VisualAssetType.OBJECT_CROP],
        top_k=10,
    )
    resp_crop = await service.search_unified(req_crop)
    for item in resp_crop.results:
        assert item.asset.asset_type == VisualAssetType.OBJECT_CROP

    # Filter only DATASET_SAMPLE
    req_ds = UnifiedSearchRequest(
        query_type="VECTOR",
        vector=[0.1] * 768,
        filter_asset_types=[VisualAssetType.DATASET_SAMPLE],
        top_k=10,
    )
    resp_ds = await service.search_unified(req_ds)
    for item in resp_ds.results:
        assert item.asset.asset_type == VisualAssetType.DATASET_SAMPLE


def test_near_duplicate_candidate_discovery():
    """Verify discovery of candidate near-duplicate pairs (sim >= 0.95)."""
    service = get_visual_search_service()
    create_synthetic_indexed_environment(service)

    dups = service.find_near_duplicates(threshold=0.95)
    assert dups.total_evaluated >= 4
    assert dups.duplicate_pairs_found >= 1
    top_pair = dups.pairs[0]
    assert top_pair.similarity_score >= 0.95
    assert (
        top_pair.asset_a.embedding_id in ("rec_frame_01", "rec_frame_02_dup")
        and top_pair.asset_b.embedding_id in ("rec_frame_01", "rec_frame_02_dup")
    )


def test_video_run_asset_indexing_and_api():
    """Verify indexing of frames, objects, and event moments from a VideoInferenceRun."""
    video_svc = get_video_intelligence_service()
    get_temporal_event_service()
    get_visual_search_service()

    # Create dummy video run
    traj = [
        TrajectoryPoint(
            frame_index=0,
            timestamp_sec=0.0,
            x_center_px=100.0,
            y_center_px=100.0,
            norm_x=0.1,
            norm_y=0.1,
            width_px=50.0,
            height_px=50.0,
            bbox=[75.0, 75.0, 125.0, 125.0],
        )
    ]
    tr = Track(
        track_id=1,
        class_name="person",
        first_frame=0,
        last_frame=30,
        first_timestamp_sec=0.0,
        last_timestamp_sec=1.0,
        visibility_duration_sec=1.0,
        avg_confidence=0.9,
        min_confidence=0.9,
        max_confidence=0.9,
        total_distance_px=10.0,
        avg_speed_px_per_sec=10.0,
        status="TERMINATED",
        trajectory=traj,
        detections_count=1,
    )
    vrun = VideoInferenceRun(
        run_id="vrun_search_test_01",
        video_id="sample_traffic_01",
        model_id="yolo11s.pt",
        tracker_name="ByteTrack",
        sampling_config=FrameSamplingConfig(mode="EVERY_FRAME", sample_interval=1, total_sampled_frames=3),
        duration_sec=3.0,
        processed_frames=3,
        total_detections=1,
        total_tracks=1,
        tracks=[tr],
        analytics={
            "total_tracks": 1,
            "tracks_by_class": {"person": 1},
            "avg_track_duration_sec": 1.0,
            "longest_track_duration_sec": 1.0,
            "avg_pixel_movement_px": 10.0,
        },
        processing_fps=60.0,
        inference_latency_ms=5.0,
        tracking_latency_ms=0.5,
    )
    video_svc._runs[vrun.run_id] = vrun

    # 1. API: Index Run
    res_idx = client.post("/api/v1/search/index-run", json={"run_id": vrun.run_id})
    assert res_idx.status_code == 200
    assert res_idx.json()["data"]["indexed_assets"] >= 3

    # 2. API: List Assets
    res_assets = client.get("/api/v1/search/assets?limit=50")
    assert res_assets.status_code == 200
    assert len(res_assets.json()["data"]) >= 3

    # 3. API: Search Frame
    res_frame = client.post(
        "/api/v1/search/frame",
        json={"video_id": "sample_traffic_01", "timestamp_sec": 1.0, "top_k": 5},
    )
    assert res_frame.status_code == 200
    assert "results" in res_frame.json()["data"]

    # 4. API: Search Object
    res_obj = client.post(
        "/api/v1/search/object",
        json={"run_id": vrun.run_id, "track_id": 1, "top_k": 5},
    )
    assert res_obj.status_code == 200

    # 5. API: Duplicates Discovery
    res_dup = client.post("/api/v1/search/duplicates?threshold=0.90")
    assert res_dup.status_code == 200
    assert "duplicate_pairs_found" in res_dup.json()["data"]
