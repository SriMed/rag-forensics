import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from main import app
from models import RetrievedChunk, DimensionResult, HedgingMismatchMetrics, RetrievalResult

client = TestClient(app)

# Dimension keys that still use DimensionResult shape (verdict/explanation/evidence)
DIMENSION_KEYS = [
    "retrieval_score_distribution",
    "chunk_attribution",
    "confidence_calibration",
]

_STUB_HEDGING_MISMATCH = HedgingMismatchMetrics(
    overconfident_fraction=0.0,
    underconfident_fraction=0.0,
    total_claims=0,
    claim_breakdown=[],
)

_STUB_CHUNKS = [
    RetrievedChunk(chunk_id="c1", text="Sample chunk text.", score=0.9),
]

# Minimal 4-dim embeddings for the stub — sufficient for PCA (2 points, 4 dims → 2 components)
_STUB_QUERY_EMBEDDING = [0.1, 0.2, 0.3, 0.4]
_STUB_CHUNK_EMBEDDINGS = [[0.9, 0.8, 0.7, 0.6]]

_STUB_RETRIEVAL_RESULT = RetrievalResult(
    chunks=_STUB_CHUNKS,
    query_embedding=_STUB_QUERY_EMBEDDING,
    chunk_embeddings=_STUB_CHUNK_EMBEDDINGS,
)

_STUB_DIMENSION = DimensionResult(verdict="pass", explanation="ok", evidence=["Sample chunk text."])

# RAGAS score functions now return (float, list[str])
_STUB_SCORE_TUPLE = (0.85, ["Sample chunk text."])


def _patch_services(mocker):
    mocker.patch("routers.analyze.retrieve_for_example", return_value=("What is X?", _STUB_RETRIEVAL_RESULT))
    mocker.patch("routers.analyze.generate_answer", return_value="Generated answer.")
    mocker.patch("routers.analyze.score_retrieval_relevance", return_value=_STUB_SCORE_TUPLE)
    mocker.patch("routers.analyze.score_answer_faithfulness", return_value=_STUB_SCORE_TUPLE)
    mocker.patch("routers.analyze.analyze_hedging_mismatch", return_value=_STUB_HEDGING_MISMATCH)


def test_post_analyze_valid_id_returns_200(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200


def test_post_analyze_missing_example_id_returns_422():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyze_response_has_ragas_field(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200
    body = response.json()
    assert "ragas" in body, "Missing ragas field"


def test_analyze_ragas_has_continuous_scores(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    ragas = body["ragas"]
    assert "retrieval_relevance_score" in ragas
    assert "faithfulness_score" in ragas
    assert isinstance(ragas["retrieval_relevance_score"], float)
    assert isinstance(ragas["faithfulness_score"], float)
    assert 0.0 <= ragas["retrieval_relevance_score"] <= 1.0
    assert 0.0 <= ragas["faithfulness_score"] <= 1.0


def test_analyze_ragas_has_no_verdict(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    ragas = body["ragas"]
    assert "verdict" not in ragas


def test_analyze_ragas_has_evidence_fields(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    ragas = body["ragas"]
    assert "relevance_evidence" in ragas
    assert "faithfulness_evidence" in ragas
    assert isinstance(ragas["relevance_evidence"], list)
    assert isinstance(ragas["faithfulness_evidence"], list)


def test_analyze_response_no_longer_has_retrieval_relevance_dimension(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    # These moved into ragas — should not exist as top-level DimensionResult fields
    assert "retrieval_relevance" not in body
    assert "answer_faithfulness" not in body


def test_analyze_response_has_remaining_dimension_keys(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200
    body = response.json()
    for key in DIMENSION_KEYS:
        assert key in body, f"Missing dimension key: {key}"


def test_analyze_response_has_generated_answer(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert "generated_answer" in body
    assert isinstance(body["generated_answer"], str)


def test_analyze_response_has_attribution_map(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert "attribution_map" in body
    assert isinstance(body["attribution_map"], list)


def test_analyze_response_has_question_and_chunks(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert "question" in body
    assert "retrieved_chunks" in body
    assert isinstance(body["retrieved_chunks"], list)


def test_analyze_remaining_dimensions_have_verdict_explanation_evidence(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    for key in DIMENSION_KEYS:
        dim = body[key]
        assert "verdict" in dim, f"{key} missing verdict"
        assert "explanation" in dim, f"{key} missing explanation"
        assert "evidence" in dim, f"{key} missing evidence"
        assert dim["verdict"] in ("pass", "warn", "fail"), f"{key} verdict invalid"
        assert isinstance(dim["evidence"], list), f"{key} evidence not a list"


def test_analyze_response_has_hedging_mismatch(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert "hedging_mismatch" in body
    hm = body["hedging_mismatch"]
    assert "overconfident_fraction" in hm
    assert "underconfident_fraction" in hm
    assert "total_claims" in hm
    assert "claim_breakdown" in hm
    assert "verdict" not in hm  # hedging_mismatch is continuous, not DimensionResult


def test_analyze_generated_answer_comes_from_generator(mocker):
    _patch_services(mocker)
    mocker.patch("routers.analyze.generate_answer", return_value="Specific generated text.")
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert body["generated_answer"] == "Specific generated text."


def test_analyze_service_failure_returns_500(mocker):
    mocker.patch("routers.analyze.retrieve_for_example", side_effect=Exception("DB error"))
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 500
