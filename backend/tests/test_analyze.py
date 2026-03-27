import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from main import app
from models import RetrievedChunk, DimensionResult

client = TestClient(app)

DIMENSION_KEYS = [
    "retrieval_relevance",
    "answer_faithfulness",
    "retrieval_score_distribution",
    "hedging_verification_mismatch",
    "chunk_attribution",
    "confidence_calibration",
]

_STUB_CHUNKS = [
    RetrievedChunk(chunk_id="c1", text="Sample chunk text.", score=0.9),
]

_STUB_DIMENSION = DimensionResult(verdict="pass", explanation="ok", evidence=["Sample chunk text."])


def _patch_services(mocker):
    mocker.patch("routers.analyze.retrieve_for_example", return_value=("What is X?", _STUB_CHUNKS))
    mocker.patch("routers.analyze.generate_answer", return_value="Generated answer.")
    mocker.patch("routers.analyze.score_retrieval_relevance", return_value=_STUB_DIMENSION)
    mocker.patch("routers.analyze.score_answer_faithfulness", return_value=_STUB_DIMENSION)


def test_post_analyze_valid_id_returns_200(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200


def test_post_analyze_missing_example_id_returns_422():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyze_response_has_all_six_dimension_keys(mocker):
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


def test_analyze_dimension_has_verdict_explanation_evidence(mocker):
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
