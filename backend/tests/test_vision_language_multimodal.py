"""Comprehensive Unit & Grounding Tests for Multimodal Vision-Language Layer."""

from fastapi.testclient import TestClient

from visionforge.main import app
from visionforge.vision_language.grounding import GroundingValidator
from visionforge.vision_language.interpreter import MultimodalQueryInterpreter
from visionforge.vision_language.schemas import (
    EvidenceType,
    MultimodalQueryStatus,
    MultiTurnContext,
    VisionEvidenceItem,
    VisionQueryType,
)
from visionforge.vision_language.service import VisionLanguageService

client = TestClient(app)


def test_query_classification_and_interpretation():
    """Verify natural language queries are parsed into correct VisionQueryType categories."""
    interpreter = MultimodalQueryInterpreter()

    # 1. Failure Query
    q1 = interpreter.parse_query(
        "Show helmet failures with confidence below 0.50", available_models=["yolo11s.pt"]
    )
    assert q1.query_type == VisionQueryType.FAILURE_QUERY
    assert q1.filters.get("class_name") == "helmet"
    assert q1.filters.get("max_confidence") == 0.50

    # 2. Dataset Query
    q2 = interpreter.parse_query("Show underrepresented classes in dataset")
    assert q2.query_type == VisionQueryType.DATASET_QUERY

    # 3. Model Query / Comparison
    q3 = interpreter.parse_query("Compare model v7 and v8")
    assert q3.query_type == VisionQueryType.MODEL_QUERY
    assert q3.target.get("model_a") == "v7"
    assert q3.target.get("model_b") == "v8"

    # 4. Search Query
    q4 = interpreter.parse_query("Find images visually similar to sample 1024")
    assert q4.query_type == VisionQueryType.SEARCH_QUERY
    assert q4.target.get("sample_id") == "1024"

    # 5. Temporal Event Query
    q5 = interpreter.parse_query("Which objects entered Zone A?")
    assert q5.query_type == VisionQueryType.EVENT_QUERY
    assert q5.filters.get("region_name") == "Zone A"


def test_numerical_and_entity_grounding_validation():
    """Verify GroundingValidator accepts truthful claims and rejects hallucinations."""
    validator = GroundingValidator()

    exec_result = {
        "count": 23,
        "total_count": 23,
        "tracks": [{"track_id": 42}, {"track_id": 51}],
        "events": [{"event_id": "evt_12"}],
    }
    evidence = [
        VisionEvidenceItem(
            evidence_id="evi_1",
            evidence_type=EvidenceType.TEMPORAL_EVENT,
            title="Zone Entry",
            description="Track #42 entered Zone A",
            track_id=42,
            event_id="evt_12",
            action_link="/video-lab",
        )
    ]

    # Test truthful candidate answer -> MUST PASS
    truthful_answer = "Found 23 matching events. Track #42 entered the zone at evt_12."
    is_valid, final_ans, discrepancies = validator.validate_answer(
        truthful_answer, exec_result, evidence
    )
    assert is_valid is True
    assert len(discrepancies) == 0
    assert "Found 23" in final_ans

    # Test numerical hallucination (claiming 25 instead of 23) -> MUST BE CAUGHT & REPLACED
    hallucinated_count = "Found 25 matching failures."
    is_valid, final_ans, discrepancies = validator.validate_answer(
        hallucinated_count, exec_result, evidence
    )
    assert is_valid is False
    assert len(discrepancies) >= 1
    assert any(d.claim_type == "NUMERICAL_MISMATCH" for d in discrepancies)
    assert "23" in final_ans

    # Test entity hallucination (claiming non-existent Track #99) -> MUST BE CAUGHT & REPLACED
    hallucinated_track = "Track #99 was observed moving."
    is_valid, final_ans, discrepancies = validator.validate_answer(
        hallucinated_track, exec_result, evidence
    )
    assert is_valid is False
    assert any(d.claim_type == "UNGROUNDED_TRACK_ID" for d in discrepancies)


def test_ambiguity_detection_and_clarification():
    """Verify ambiguous requests without model context produce clarification prompts."""
    interpreter = MultimodalQueryInterpreter()

    # Ambiguous generic failure question when multiple models exist
    q = interpreter.parse_query(
        "Show failures",
        available_models=["yolo11s.pt", "yolo11m.pt", "yolo11x.pt"],
    )
    assert q.status == MultimodalQueryStatus.AMBIGUOUS
    assert q.clarification_needed is not None
    assert len(q.clarification_options) == 3
    assert "yolo11s.pt" in q.clarification_options


def test_multi_turn_conversational_context(tmp_path):
    """Verify follow-up queries inherit filters and context from previous turns."""
    service = VisionLanguageService(storage_dir=tmp_path)

    # Turn 1: "Show helmet failures"
    res1 = service.ask(
        "Show helmet failures",
        context=MultiTurnContext(session_id="ses_1", selected_model="yolo11s.pt"),
    )
    assert res1.query_type == VisionQueryType.FAILURE_QUERY
    assert res1.filters.get("class_name") == "helmet"

    # Turn 2: "Only low confidence ones"
    ctx2 = MultiTurnContext(
        session_id="ses_1",
        selected_model="yolo11s.pt",
        previous_query=res1.model_dump(),
    )
    res2 = service.ask("Only low confidence ones", context=ctx2)
    assert res2.query_type == VisionQueryType.FAILURE_QUERY
    assert res2.filters.get("class_name") == "helmet"
    assert res2.filters.get("max_confidence") == 0.50


def test_unsupported_and_security_queries(tmp_path):
    """Verify rejection of arbitrary SQL, shell injection, and speculative questions."""
    service = VisionLanguageService(storage_dir=tmp_path)

    # Injection test
    res_sql = service.ask("SELECT * FROM users; DROP TABLE models;")
    assert res_sql.status == MultimodalQueryStatus.UNSUPPORTED

    # Subjective emotion test
    res_emotion = service.ask("What is the person in the image feeling?")
    assert res_emotion.status == MultimodalQueryStatus.UNSUPPORTED


def test_dataset_and_model_queries_execution(tmp_path):
    """Verify execution of dataset and model evaluation queries."""
    service = VisionLanguageService(storage_dir=tmp_path)

    # Dataset query
    q_data = service.ask("How many samples are in the dataset?")
    assert q_data.query_type == VisionQueryType.DATASET_QUERY
    assert q_data.status == MultimodalQueryStatus.SUCCESS
    assert "samples in dataset" in q_data.answer
    assert len(q_data.evidence) > 0

    # Model comparison query
    q_comp = service.ask("Compare model yolo11s.pt and yolo11m.pt")
    assert q_comp.query_type == VisionQueryType.MODEL_QUERY
    assert "mAP@50 delta" in q_comp.answer
    assert len(q_comp.evidence) > 0


def test_visual_search_and_temporal_queries_execution(tmp_path):
    """Verify visual search and temporal event questions execution."""
    service = VisionLanguageService(storage_dir=tmp_path)

    # Visual Search
    q_search = service.ask("Find images visually similar to sample 1024")
    assert q_search.query_type == VisionQueryType.SEARCH_QUERY
    assert "visually similar" in q_search.answer

    # Temporal Event
    q_event = service.ask("Which objects entered Zone A?")
    assert q_event.query_type == VisionQueryType.EVENT_QUERY
    assert q_event.grounding_verified is True


def test_query_history_and_replay(tmp_path):
    """Verify history recording and deterministic replay."""
    service = VisionLanguageService(storage_dir=tmp_path)

    q = service.ask("Show underrepresented classes")
    history = service.list_history()
    assert len(history) >= 1
    assert history[0].query_id == q.query_id

    # Replay
    replayed = service.replay_query(q.query_id)
    assert replayed.query_type == q.query_type
    assert replayed.grounding_verified is True


def test_multimodal_api_endpoints():
    """Verify REST API routes for Multimodal Vision-Language."""
    # 1. Ask query
    res = client.post(
        "/api/v1/multimodal/ask",
        json={"query": "Show underrepresented classes"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "query_id" in data
    assert data["query_type"] == "DATASET_QUERY"
    qid = data["query_id"]

    # 2. Get query detail
    res_get = client.get(f"/api/v1/multimodal/queries/{qid}")
    assert res_get.status_code == 200
    assert res_get.json()["query_id"] == qid

    # 3. Get evidence
    res_evi = client.get(f"/api/v1/multimodal/queries/{qid}/evidence")
    assert res_evi.status_code == 200
    assert isinstance(res_evi.json(), list)

    # 4. History
    res_hist = client.get("/api/v1/multimodal/history")
    assert res_hist.status_code == 200
    assert len(res_hist.json()) >= 1

    # 5. Suggested queries
    res_sug = client.get("/api/v1/multimodal/suggested?page_context=failure_gallery")
    assert res_sug.status_code == 200
    assert len(res_sug.json()) >= 2
