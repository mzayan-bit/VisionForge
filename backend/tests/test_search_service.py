"""Unit test suite for Visual Search Service and Search History."""

import pytest
from fastapi.testclient import TestClient

from visionforge.main import app
from visionforge.memory.index import VisualMemoryIndex, VisualMemoryRecord
from visionforge.search.engine import SimilarityMetric, VisualSearchEngine
from visionforge.search.history import SearchHistoryStore
from visionforge.search.service import VisualSearchService

client = TestClient(app)


@pytest.fixture
def populated_memory_index(tmp_path):
    """Create a temporary VisualMemoryIndex with 4 synthetic records."""
    mem = VisualMemoryIndex(storage_dir=str(tmp_path / "memory"))

    # Vector dimensions 768
    rec1 = VisualMemoryRecord(
        id="item_high_match",
        embedding=[1.0] + [0.0] * 767,
        image_metadata={"width": 224, "height": 224, "format": "JPEG"},
        tags=["target"],
    )
    rec2 = VisualMemoryRecord(
        id="item_mid_match",
        embedding=[0.7071, 0.7071] + [0.0] * 766,
        image_metadata={"width": 400, "height": 300, "format": "PNG"},
        tags=["similar"],
    )
    rec3 = VisualMemoryRecord(
        id="item_low_match",
        embedding=[0.0, 1.0] + [0.0] * 766,
        image_metadata={"width": 100, "height": 100, "format": "WEBP"},
        tags=["different"],
    )

    mem.add_record(rec1)
    mem.add_record(rec2)
    mem.add_record(rec3)
    return mem


def test_search_service_top_k_and_ranking(populated_memory_index, tmp_path):
    """Verify search service computes correct ranking order and respects Top-K limits."""
    history = SearchHistoryStore(storage_dir=str(tmp_path / "history"))
    engine = VisualSearchEngine(memory_index=populated_memory_index)
    svc = VisualSearchService(
        search_engine=engine, memory_index=populated_memory_index, history_store=history
    )

    query_vec = [1.0] + [0.0] * 767

    # Top-K = 2
    res = svc.search_by_vector(query_vector=query_vec, top_k=2, metric=SimilarityMetric.COSINE)
    assert res.candidate_count == 3
    assert res.returned_count == 2
    assert len(res.results) == 2

    # Rank 1 should be item_high_match
    assert res.results[0].rank == 1
    assert res.results[0].id == "item_high_match"
    assert abs(res.results[0].similarity_score - 1.0) < 1e-3

    # Rank 2 should be item_mid_match
    assert res.results[1].rank == 2
    assert res.results[1].id == "item_mid_match"


def test_search_service_threshold_filtering(populated_memory_index, tmp_path):
    """Verify threshold filtering excludes results below minimum score."""
    history = SearchHistoryStore(storage_dir=str(tmp_path / "history"))
    engine = VisualSearchEngine(memory_index=populated_memory_index)
    svc = VisualSearchService(
        search_engine=engine, memory_index=populated_memory_index, history_store=history
    )

    query_vec = [1.0] + [0.0] * 767

    # Threshold = 0.8 should only keep item_high_match (score 1.0)
    res = svc.search_by_vector(
        query_vector=query_vec, top_k=5, metric=SimilarityMetric.COSINE, threshold=0.8
    )
    assert res.returned_count == 1
    assert res.results[0].id == "item_high_match"


def test_search_service_query_by_record(populated_memory_index, tmp_path):
    """Verify search by existing Visual Memory record ID."""
    history = SearchHistoryStore(storage_dir=str(tmp_path / "history"))
    engine = VisualSearchEngine(memory_index=populated_memory_index)
    svc = VisualSearchService(
        search_engine=engine, memory_index=populated_memory_index, history_store=history
    )

    res = svc.search_by_record(record_id="item_high_match", top_k=3)
    assert res.returned_count == 3
    assert res.results[0].id == "item_high_match"

    # Search history logging
    history_records = svc.get_search_history()
    assert len(history_records) >= 1
    assert history_records[0].query_type == "memory_record"


def test_search_service_empty_memory(tmp_path):
    """Verify graceful handling when visual memory index is empty."""
    empty_mem = VisualMemoryIndex(storage_dir=str(tmp_path / "empty_mem"))
    history = SearchHistoryStore(storage_dir=str(tmp_path / "history"))
    engine = VisualSearchEngine(memory_index=empty_mem)
    svc = VisualSearchService(
        search_engine=engine, memory_index=empty_mem, history_store=history
    )

    query_vec = [0.1] * 768
    res = svc.search_by_vector(query_vector=query_vec, top_k=5)
    assert res.candidate_count == 0
    assert res.returned_count == 0
    assert len(res.results) == 0


def test_api_record_search_and_history_endpoints():
    """Verify REST API endpoints for search by record and search history log."""
    # History endpoint
    hist_res = client.get("/api/v1/search/history")
    assert hist_res.status_code == 200
    assert hist_res.json()["success"] is True

    # Vector search API
    vec_payload = {
        "vector": [0.1] * 768,
        "top_k": 3,
        "metric": "cosine",
        "threshold": 0.0,
    }
    search_res = client.post("/api/v1/search/vector", json=vec_payload)
    assert search_res.status_code == 200
    assert search_res.json()["success"] is True
