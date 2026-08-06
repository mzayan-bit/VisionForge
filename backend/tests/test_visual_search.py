"""Unit test suite for Visual Search Engine."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from visionforge.main import app
from visionforge.memory.index import VisualMemoryIndex, VisualMemoryRecord
from visionforge.search.engine import SimilarityMetric, VisualSearchEngine

client = TestClient(app)


def _create_sample_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_visual_search_vector_matching(tmp_path):
    """Test vector similarity search with Cosine and Euclidean metrics."""
    mem = VisualMemoryIndex(storage_dir=str(tmp_path))

    # Add 2 target vectors
    v1 = [1.0] + [0.0] * 767
    v2 = [0.0] + [1.0] + [0.0] * 766

    mem.add_record(VisualMemoryRecord(id="rec_1", embedding=v1, tags=["v1"]))
    mem.add_record(VisualMemoryRecord(id="rec_2", embedding=v2, tags=["v2"]))

    engine = VisualSearchEngine(memory_index=mem)

    # Search close to v1
    res_cosine = engine.search_by_vector(query_vector=v1, top_k=2, metric=SimilarityMetric.COSINE)
    assert res_cosine.candidate_count == 2
    assert len(res_cosine.results) == 2
    assert res_cosine.results[0].id == "rec_1"
    assert abs(res_cosine.results[0].similarity_score - 1.0) < 1e-3

    # Search Euclidean
    res_euc = engine.search_by_vector(query_vector=v1, top_k=2, metric=SimilarityMetric.EUCLIDEAN)
    assert res_euc.results[0].id == "rec_1"


def test_memory_and_search_api_endpoints():
    """Verify REST API endpoints for memory stats and visual search."""
    # Memory stats
    stats_res = client.get("/api/v1/memory/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["success"] is True

    # Index an image
    img_bytes = _create_sample_image_bytes()
    files = {"file": ("query.jpg", img_bytes, "image/jpeg")}
    index_res = client.post("/api/v1/memory/index", files=files, data={"tags": "test_tag"})
    assert index_res.status_code == 200
    rec_id = index_res.json()["data"]["id"]

    # Search by image
    search_files = {"file": ("query.jpg", img_bytes, "image/jpeg")}
    search_res = client.post("/api/v1/search/image", files=search_files, data={"top_k": "5"})
    assert search_res.status_code == 200
    data = search_res.json()["data"]
    assert len(data["results"]) >= 1
    assert data["results"][0]["id"] == rec_id

    # Clear memory
    clear_res = client.delete("/api/v1/memory/clear")
    assert clear_res.status_code == 200
