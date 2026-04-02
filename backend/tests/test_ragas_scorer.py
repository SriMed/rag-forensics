"""Tests for services/ragas_scorer.py — mocks ragas.evaluate entirely.

Both scoring functions return (float, list[str]) — no verdict, no DimensionResult.
"""
import pytest
from unittest.mock import MagicMock
from models import RetrievedChunk

CHUNKS = [
    RetrievedChunk(chunk_id="c1", text="The mitochondria is the powerhouse of the cell. It produces ATP.", score=0.9),
    RetrievedChunk(chunk_id="c2", text="Cells require energy to function and survive.", score=0.8),
    RetrievedChunk(chunk_id="c3", text="ATP is the primary energy currency in biology.", score=0.7),
]


def _mock_evaluate(mocker, metric_name: str, score: float):
    mock_result = {metric_name: [score]}
    mocker.patch("services.ragas_scorer.evaluate", return_value=mock_result)


# --- _extract_evidence ---

def test_extract_evidence_returns_list():
    from services.ragas_scorer import _extract_evidence
    result = _extract_evidence(CHUNKS)
    assert isinstance(result, list)


def test_extract_evidence_between_1_and_3_items():
    from services.ragas_scorer import _extract_evidence
    result = _extract_evidence(CHUNKS)
    assert 1 <= len(result) <= 3


def test_extract_evidence_items_are_strings():
    from services.ragas_scorer import _extract_evidence
    result = _extract_evidence(CHUNKS)
    for item in result:
        assert isinstance(item, str)


def test_extract_evidence_items_appear_verbatim_in_chunks():
    from services.ragas_scorer import _extract_evidence
    all_chunk_text = " ".join(c.text for c in CHUNKS)
    result = _extract_evidence(CHUNKS)
    for item in result:
        assert item in all_chunk_text, f"Evidence '{item}' not found verbatim in any chunk"


def test_extract_evidence_uses_top_chunks():
    from services.ragas_scorer import _extract_evidence
    single = [RetrievedChunk(chunk_id="x", text="Only chunk text here.", score=1.0)]
    result = _extract_evidence(single)
    assert len(result) == 1
    assert result[0] in single[0].text


# --- score_retrieval_relevance ---

def test_score_retrieval_relevance_returns_tuple(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    result = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_score_retrieval_relevance_score_is_float(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    score, _ = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert isinstance(score, float)


def test_score_retrieval_relevance_score_between_0_and_1(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    score, _ = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert 0.0 <= score <= 1.0


def test_score_retrieval_relevance_no_verdict(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    result = score_retrieval_relevance("What is ATP?", CHUNKS)
    # result must be a plain tuple, not a DimensionResult or any object with .verdict
    assert not hasattr(result, "verdict")


def test_score_retrieval_relevance_has_evidence(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    _, evidence = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert 1 <= len(evidence) <= 3


def test_score_retrieval_relevance_evidence_verbatim_in_chunks(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    _, evidence = score_retrieval_relevance("What is ATP?", CHUNKS)
    all_text = " ".join(c.text for c in CHUNKS)
    for item in evidence:
        assert item in all_text


def test_score_retrieval_relevance_score_low_value(mocker):
    _mock_evaluate(mocker, "context_precision", 0.2)
    from services.ragas_scorer import score_retrieval_relevance
    score, _ = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert score == pytest.approx(0.2)


# --- score_answer_faithfulness ---

def test_score_answer_faithfulness_returns_tuple(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.9)
    from services.ragas_scorer import score_answer_faithfulness
    result = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_score_answer_faithfulness_score_is_float(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.9)
    from services.ragas_scorer import score_answer_faithfulness
    score, _ = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert isinstance(score, float)


def test_score_answer_faithfulness_score_between_0_and_1(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.9)
    from services.ragas_scorer import score_answer_faithfulness
    score, _ = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert 0.0 <= score <= 1.0


def test_score_answer_faithfulness_no_verdict(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.9)
    from services.ragas_scorer import score_answer_faithfulness
    result = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert not hasattr(result, "verdict")


def test_score_answer_faithfulness_has_evidence(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.8)
    from services.ragas_scorer import score_answer_faithfulness
    _, evidence = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert 1 <= len(evidence) <= 3


def test_score_answer_faithfulness_evidence_verbatim_in_chunks(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.8)
    from services.ragas_scorer import score_answer_faithfulness
    _, evidence = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    all_text = " ".join(c.text for c in CHUNKS)
    for item in evidence:
        assert item in all_text


def test_score_answer_faithfulness_score_low_value(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.1)
    from services.ragas_scorer import score_answer_faithfulness
    score, _ = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert score == pytest.approx(0.1)
