import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from main import app
from models import RetrievedChunk, HedgingMismatchMetrics, ChunkAttributionMetrics, QueryCorpusFitMetrics, RAGASMetricResult, RetrievalResult

client = TestClient(app)

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

_STUB_CHUNK_ATTRIBUTION = ChunkAttributionMetrics(
    unattributed_fraction=0.0,
    mean_attribution_score=0.9,
    weak_match_fraction=0.0,
    attribution_map=[],
)

_STUB_SCORE_TUPLE = (RAGASMetricResult(score=0.85, status="ok"), ["Sample chunk text."])

_STUB_QUERY_CORPUS_FIT = QueryCorpusFitMetrics(
    triggered=False,
    observed_fit=None,
    suggested_questions=[],
    mean_question_similarity=None,
)


def _patch_services(mocker):
    mocker.patch("routers.analyze.retrieve_for_example", return_value=("What is X?", _STUB_RETRIEVAL_RESULT))
    mocker.patch("routers.analyze.generate_answer", return_value="Generated answer.")
    mocker.patch("routers.analyze.score_context_utilization", return_value=_STUB_SCORE_TUPLE)
    mocker.patch("routers.analyze.score_answer_faithfulness", return_value=_STUB_SCORE_TUPLE)
    mocker.patch("routers.analyze.analyze_hedging_mismatch", return_value=_STUB_HEDGING_MISMATCH)
    mocker.patch("routers.analyze.analyze_chunk_attribution", return_value=_STUB_CHUNK_ATTRIBUTION)
    mocker.patch("routers.analyze.analyze_query_corpus_fit", return_value=_STUB_QUERY_CORPUS_FIT)
    mocker.patch("routers.analyze.render_recommendation", return_value="No changes indicated.")


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
    assert ragas["context_utilization"] == {"score": 0.85, "status": "ok", "error": None}
    assert ragas["faithfulness"] == {"score": 0.85, "status": "ok", "error": None}


def test_analyze_ragas_exposes_unavailable_state(mocker):
    _patch_services(mocker)
    unavailable = RAGASMetricResult(
        score=None, status="unavailable", error="evaluation_failed"
    )
    mocker.patch(
        "routers.analyze.score_context_utilization",
        return_value=(unavailable, ["Sample chunk text."]),
    )
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200
    assert response.json()["ragas"]["context_utilization"] == {
        "score": None,
        "status": "unavailable",
        "error": "evaluation_failed",
    }


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
    assert "utilization_context_excerpts" in ragas
    assert "faithfulness_context_excerpts" in ragas
    assert isinstance(ragas["utilization_context_excerpts"], list)
    assert isinstance(ragas["faithfulness_context_excerpts"], list)


def test_analyze_response_no_longer_has_context_utilization_dimension(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    # These moved into ragas — should not exist as top-level DimensionResult fields
    assert "context_utilization" not in body
    assert "answer_faithfulness" not in body


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
    assert "chunk_attribution" in body
    assert "attribution_map" in body["chunk_attribution"]
    assert isinstance(body["chunk_attribution"]["attribution_map"], list)


def test_analyze_response_has_question_and_chunks(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    assert "question" in body
    assert "retrieved_chunks" in body
    assert isinstance(body["retrieved_chunks"], list)
    assert body["retrieved_chunk_details"][0]["completeness"] == "unknown"
    assert body["retrieved_chunk_details"][0]["completeness_source"] == "unavailable"




def test_analyze_response_chunk_attribution_is_continuous_metrics(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    body = response.json()
    ca = body["chunk_attribution"]
    assert "unattributed_fraction" in ca
    assert "mean_attribution_score" in ca
    assert "weak_match_fraction" in ca
    assert "attribution_map" in ca
    assert "verdict" not in ca  # ChunkAttributionMetrics is continuous, not DimensionResult


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


def test_analyze_response_has_query_corpus_fit(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200
    body = response.json()
    assert "query_corpus_fit" in body
    qcf = body["query_corpus_fit"]
    assert "triggered" in qcf
    assert "observed_fit" in qcf
    assert "suggested_questions" in qcf
    assert "mean_question_similarity" in qcf
    assert isinstance(qcf["triggered"], bool)
    assert isinstance(qcf["suggested_questions"], list)


def test_analyze_response_has_recommendation(mocker):
    _patch_services(mocker)
    response = client.post("/analyze", json={"example_id": "techqa-001"})
    assert response.status_code == 200
    body = response.json()
    assert "recommendation" in body
    assert isinstance(body["recommendation"], str)


def test_analyze_response_exposes_structured_verdict_reasoning(mocker):
    _patch_services(mocker)
    body = client.post("/analyze", json={"example_id": "techqa-001"}).json()
    reasoning = body["verdict_reasoning"]
    assert reasoning["observations"]
    assert len(reasoning["hypotheses"]) == 2
    assert reasoning["test"]["component"]
    assert len(reasoning["test"]["interpretations"]) == 2
    assert "rule_id" not in body
