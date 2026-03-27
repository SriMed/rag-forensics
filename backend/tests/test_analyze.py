import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

DIMENSION_KEYS = [
    "retrieval_relevance",
    "answer_faithfulness",
    "retrieval_score_distribution",
    "hedging_verification_mismatch",
    "chunk_attribution",
    "confidence_calibration",
]


def test_post_analyze_valid_id_returns_200():
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200


def test_post_analyze_missing_example_id_returns_422():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyze_response_has_all_six_dimension_keys():
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200
    body = response.json()
    for key in DIMENSION_KEYS:
        assert key in body, f"Missing dimension key: {key}"


def test_analyze_response_has_generated_answer():
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert "generated_answer" in body
    assert isinstance(body["generated_answer"], str)


def test_analyze_response_has_attribution_map():
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert "attribution_map" in body
    assert isinstance(body["attribution_map"], list)


def test_analyze_response_has_question_and_chunks():
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert "question" in body
    assert "retrieved_chunks" in body
    assert isinstance(body["retrieved_chunks"], list)


def test_analyze_dimension_has_verdict_explanation_evidence():
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    for key in DIMENSION_KEYS:
        dim = body[key]
        assert "verdict" in dim, f"{key} missing verdict"
        assert "explanation" in dim, f"{key} missing explanation"
        assert "evidence" in dim, f"{key} missing evidence"
        assert dim["verdict"] in ("pass", "warn", "fail"), f"{key} verdict invalid"
        assert isinstance(dim["evidence"], list), f"{key} evidence not a list"
