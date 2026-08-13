import pytest
from fastapi.testclient import TestClient
from main import app
from models import (
    HedgingMismatchMetrics,
    ChunkAttributionMetrics,
    QueryCorpusFitMetrics,
    RAGASMetricResult,
    AttributionEntry,
)

client = TestClient(app)

_VALID_REQUEST = {
    "question": "What is the refund policy?",
    "answer": "Refunds are processed in 5-7 days. Customers must request within 30 days.",
    "score_semantics": "normalized_similarity",
    "chunks": [
        {"chunk_id": "doc_1_chunk_0", "text": "Refunds are processed within 5 to 7 business days.", "score": 0.87},
        {"chunk_id": "doc_1_chunk_1", "text": "Customers must request a refund within 30 days of purchase.", "score": 0.75},
    ],
}

_STUB_HEDGING = HedgingMismatchMetrics(
    overconfident_fraction=0.0,
    underconfident_fraction=0.0,
    total_claims=0,
    claim_breakdown=[],
)

_STUB_ATTRIBUTION = ChunkAttributionMetrics(
    unattributed_fraction=0.0,
    mean_attribution_score=0.9,
    weak_match_fraction=0.0,
    attribution_map=[
        AttributionEntry(
            sentence="Refunds are processed in 5-7 days.",
            chunk_id="doc_1_chunk_0",
            similarity_score=0.9,
            attribution_strength="strong",
        ),
        AttributionEntry(
            sentence="Customers must request within 30 days.",
            chunk_id="doc_1_chunk_1",
            similarity_score=0.8,
            attribution_strength="strong",
        ),
    ],
)

_STUB_QUERY_FIT = QueryCorpusFitMetrics(
    triggered=False,
    observed_fit=None,
    suggested_questions=[],
    mean_question_similarity=None,
)

_STUB_SCORE_TUPLE = (RAGASMetricResult(score=0.85, status="ok"), ["evidence text"])


def _patch_services(mocker):
    embedding_model = mocker.MagicMock()
    embedding_model.encode.side_effect = [
        [[1.0, 0.0]],
        [[1.0, 0.0], [0.8, 0.2]],
    ]
    mocker.patch("services.retriever.get_embedding_model", return_value=embedding_model)
    mocker.patch("routers.analyze.score_context_utilization", return_value=_STUB_SCORE_TUPLE)
    mocker.patch("routers.analyze.score_answer_faithfulness", return_value=_STUB_SCORE_TUPLE)
    mocker.patch("routers.analyze.analyze_hedging_mismatch", return_value=_STUB_HEDGING)
    mocker.patch("routers.analyze.analyze_chunk_attribution", return_value=_STUB_ATTRIBUTION)
    mocker.patch("routers.analyze.analyze_query_corpus_fit", return_value=_STUB_QUERY_FIT)
    mocker.patch("routers.analyze.render_recommendation", return_value="No changes indicated.")


def test_custom_valid_request_returns_200(mocker):
    _patch_services(mocker)
    response = client.post("/analyze/custom", json=_VALID_REQUEST)
    assert response.status_code == 200


def test_custom_response_is_analyze_response_shape(mocker):
    _patch_services(mocker)
    response = client.post("/analyze/custom", json=_VALID_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert "question" in body
    assert "generated_answer" in body
    assert "retrieved_chunks" in body
    assert "ragas" in body
    assert "hedging_mismatch" in body
    assert "chunk_attribution" in body
    assert "retrieval_distribution" in body
    assert "embedding_space" in body
    assert "query_corpus_fit" in body
    assert "verdict_reasoning" in body
    assert "recommendation" in body
    assert "rule_id" not in body


def test_custom_empty_chunks_returns_422(mocker):
    _patch_services(mocker)
    payload = {**_VALID_REQUEST, "chunks": []}
    response = client.post("/analyze/custom", json=payload)
    assert response.status_code == 422


def test_custom_missing_question_returns_422(mocker):
    _patch_services(mocker)
    payload = {k: v for k, v in _VALID_REQUEST.items() if k != "question"}
    response = client.post("/analyze/custom", json=payload)
    assert response.status_code == 422


def test_custom_missing_answer_returns_422(mocker):
    _patch_services(mocker)
    payload = {k: v for k, v in _VALID_REQUEST.items() if k != "answer"}
    response = client.post("/analyze/custom", json=payload)
    assert response.status_code == 422


def test_custom_missing_score_semantics_returns_422(mocker):
    _patch_services(mocker)
    payload = {k: v for k, v in _VALID_REQUEST.items() if k != "score_semantics"}
    response = client.post("/analyze/custom", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_custom_rejects_scores_outside_normalized_range(mocker, score):
    _patch_services(mocker)
    payload = {**_VALID_REQUEST, "chunks": [{**_VALID_REQUEST["chunks"][0], "score": score}]}
    response = client.post("/analyze/custom", json=payload)
    assert response.status_code == 422


def test_custom_chunk_missing_score_returns_400(mocker):
    _patch_services(mocker)
    payload = {
        **_VALID_REQUEST,
        "chunks": [{"chunk_id": "c1", "text": "Some text."}],
    }
    response = client.post("/analyze/custom", json=payload)
    assert response.status_code in (400, 422)


def test_custom_all_six_forensics_dimensions_present(mocker):
    _patch_services(mocker)
    response = client.post("/analyze/custom", json=_VALID_REQUEST)
    body = response.json()
    assert "ragas" in body
    assert "hedging_mismatch" in body
    assert "chunk_attribution" in body
    assert "retrieval_distribution" in body
    assert "embedding_space" in body
    assert "query_corpus_fit" in body


def test_custom_attribution_map_length_matches_sentences(mocker):
    _patch_services(mocker)
    response = client.post("/analyze/custom", json=_VALID_REQUEST)
    body = response.json()
    # _STUB_ATTRIBUTION has 2 entries (one per sentence in the answer)
    assert len(body["chunk_attribution"]["attribution_map"]) == 2


def test_custom_retriever_not_called(mocker):
    _patch_services(mocker)
    mock_retriever = mocker.patch("routers.analyze.retrieve_for_example")
    client.post("/analyze/custom", json=_VALID_REQUEST)
    mock_retriever.assert_not_called()


def test_custom_answer_echoed_back(mocker):
    _patch_services(mocker)
    response = client.post("/analyze/custom", json=_VALID_REQUEST)
    body = response.json()
    assert body["generated_answer"] == _VALID_REQUEST["answer"]


def test_custom_question_echoed_back(mocker):
    _patch_services(mocker)
    response = client.post("/analyze/custom", json=_VALID_REQUEST)
    body = response.json()
    assert body["question"] == _VALID_REQUEST["question"]


def test_custom_chunks_reflected_in_retrieved_chunks(mocker):
    _patch_services(mocker)
    response = client.post("/analyze/custom", json=_VALID_REQUEST)
    body = response.json()
    expected_texts = [c["text"] for c in _VALID_REQUEST["chunks"]]
    assert body["retrieved_chunks"] == expected_texts
