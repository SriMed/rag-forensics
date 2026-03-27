"""Tests for services/ragas_scorer.py — mocks ragas.evaluate entirely."""
import pytest
from unittest.mock import MagicMock
from models import RetrievedChunk, DimensionResult

CHUNKS = [
    RetrievedChunk(chunk_id="c1", text="The mitochondria is the powerhouse of the cell. It produces ATP.", score=0.9),
    RetrievedChunk(chunk_id="c2", text="Cells require energy to function and survive.", score=0.8),
    RetrievedChunk(chunk_id="c3", text="ATP is the primary energy currency in biology.", score=0.7),
]


def _mock_evaluate(mocker, metric_name: str, score: float):
    mock_result = {metric_name: [score]}
    mocker.patch("services.ragas_scorer.evaluate", return_value=mock_result)


# --- _verdict_from_score ---

def test_verdict_pass_at_0_75():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(0.75) == "pass"


def test_verdict_pass_above_threshold():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(0.8) == "pass"


def test_verdict_pass_at_1_0():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(1.0) == "pass"


def test_verdict_warn_at_0_5():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(0.5) == "warn"


def test_verdict_warn_at_0_6():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(0.6) == "warn"


def test_verdict_warn_just_below_pass():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(0.74) == "warn"


def test_verdict_fail_at_0_3():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(0.3) == "fail"


def test_verdict_fail_at_0_0():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(0.0) == "fail"


def test_verdict_fail_just_below_warn():
    from services.ragas_scorer import _verdict_from_score
    assert _verdict_from_score(0.49) == "fail"


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
    # Only pass one chunk — should return exactly 1 evidence item
    single = [RetrievedChunk(chunk_id="x", text="Only chunk text here.", score=1.0)]
    result = _extract_evidence(single)
    assert len(result) == 1
    assert result[0] in single[0].text


# --- score_retrieval_relevance ---

def test_score_retrieval_relevance_returns_dimension_result(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    result = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert isinstance(result, DimensionResult)


def test_score_retrieval_relevance_pass_verdict(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    result = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert result.verdict == "pass"


def test_score_retrieval_relevance_warn_verdict(mocker):
    _mock_evaluate(mocker, "context_precision", 0.6)
    from services.ragas_scorer import score_retrieval_relevance
    result = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert result.verdict == "warn"


def test_score_retrieval_relevance_fail_verdict(mocker):
    _mock_evaluate(mocker, "context_precision", 0.3)
    from services.ragas_scorer import score_retrieval_relevance
    result = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert result.verdict == "fail"


def test_score_retrieval_relevance_has_evidence(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    result = score_retrieval_relevance("What is ATP?", CHUNKS)
    assert 1 <= len(result.evidence) <= 3


def test_score_retrieval_relevance_evidence_verbatim_in_chunks(mocker):
    _mock_evaluate(mocker, "context_precision", 0.8)
    from services.ragas_scorer import score_retrieval_relevance
    result = score_retrieval_relevance("What is ATP?", CHUNKS)
    all_text = " ".join(c.text for c in CHUNKS)
    for item in result.evidence:
        assert item in all_text


# --- score_answer_faithfulness ---

def test_score_answer_faithfulness_returns_dimension_result(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.9)
    from services.ragas_scorer import score_answer_faithfulness
    result = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert isinstance(result, DimensionResult)


def test_score_answer_faithfulness_pass_verdict(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.8)
    from services.ragas_scorer import score_answer_faithfulness
    result = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert result.verdict == "pass"


def test_score_answer_faithfulness_warn_verdict(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.6)
    from services.ragas_scorer import score_answer_faithfulness
    result = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert result.verdict == "warn"


def test_score_answer_faithfulness_fail_verdict(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.3)
    from services.ragas_scorer import score_answer_faithfulness
    result = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert result.verdict == "fail"


def test_score_answer_faithfulness_has_evidence(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.8)
    from services.ragas_scorer import score_answer_faithfulness
    result = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    assert 1 <= len(result.evidence) <= 3


def test_score_answer_faithfulness_evidence_verbatim_in_chunks(mocker):
    _mock_evaluate(mocker, "faithfulness", 0.8)
    from services.ragas_scorer import score_answer_faithfulness
    result = score_answer_faithfulness("ATP is made in mitochondria.", CHUNKS, "What makes ATP?")
    all_text = " ".join(c.text for c in CHUNKS)
    for item in result.evidence:
        assert item in all_text
