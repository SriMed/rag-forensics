"""Tests for query-corpus fit forensics module — written before implementation (TDD)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=f"c{i}", text=f"chunk text {i}", score=round(0.9 - i * 0.1, 1))
        for i in range(n)
    ]


def _unit(dim: int = 8, seed: int = 42) -> np.ndarray:
    v = np.random.default_rng(seed).random(dim)
    return v / np.linalg.norm(v)


def _orthogonal(base: np.ndarray, seed: int = 99) -> np.ndarray:
    """Return a unit vector orthogonal to base."""
    rng = np.random.default_rng(seed)
    v = rng.random(len(base))
    v = v - np.dot(v, base) * base
    return v / np.linalg.norm(v)


def _partial(base: np.ndarray, target_sim: float, seed: int = 77) -> np.ndarray:
    """Return a unit vector with cosine similarity exactly target_sim to base."""
    rng = np.random.default_rng(seed)
    perp = rng.random(len(base))
    perp = perp - np.dot(perp, base) * base
    perp = perp / np.linalg.norm(perp)
    beta = np.sqrt(max(0.0, 1.0 - target_sim ** 2))
    v = target_sim * base + beta * perp
    return v / np.linalg.norm(v)


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_claude_mock(mocker, response) -> MagicMock:
    """Patch anthropic.Anthropic in query_corpus_fit. response is str or Exception instance."""
    mock_client = MagicMock()
    if isinstance(response, BaseException):
        mock_client.messages.create.side_effect = response
    else:
        mock_client.messages.create.return_value = _mock_response(response)
    mock_cls = MagicMock(return_value=mock_client)
    mocker.patch("services.forensics.query_corpus_fit.anthropic.Anthropic", mock_cls)
    return mock_client


def _make_embed_mock(embeddings: list[np.ndarray]) -> MagicMock:
    """Return a mock embedding model. encode([q]) returns (1, dim) array per call."""
    model = MagicMock()
    model.encode.side_effect = [e.reshape(1, -1) for e in embeddings]
    return model


# All-clear signal values — no trigger conditions met
_NO_TRIGGER = dict(
    query_isolation=0.5,
    retrieval_relevance_score=0.8,
    score_entropy=0.5,
    faithfulness_score=0.8,
)


# ---------------------------------------------------------------------------
# Test 1 — no trigger conditions met → triggered=False, empty fields, no Claude call
# ---------------------------------------------------------------------------

def test_no_trigger_returns_false(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    mock_client = _make_claude_mock(mocker, "[]")
    query_emb = _unit(seed=1)
    chunks = _chunks(2)
    chunk_embs = [_unit(seed=10 + i) for i in range(2)]

    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=query_emb,
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        **_NO_TRIGGER,
    )

    assert result.triggered is False
    assert result.suggested_questions == []
    assert result.mismatch_type is None
    assert result.mean_question_similarity is None
    mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2 — query_isolation > 1.2 → triggered=True
# ---------------------------------------------------------------------------

def test_query_isolation_triggers(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    query_emb = _unit(seed=1)
    q_emb = _unit(seed=2)
    chunks = _chunks(1)
    chunk_embs = [_unit(seed=10)]

    _make_claude_mock(mocker, '["What is Y?"]')
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock([q_emb])
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.3,
            retrieval_relevance_score=0.8,
            score_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.triggered is True


# ---------------------------------------------------------------------------
# Test 3 — retrieval_relevance_score < 0.5 → triggered=True
# ---------------------------------------------------------------------------

def test_retrieval_relevance_triggers(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    query_emb = _unit(seed=1)
    q_emb = _unit(seed=2)
    chunks = _chunks(1)
    chunk_embs = [_unit(seed=10)]

    _make_claude_mock(mocker, '["What is Y?"]')
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock([q_emb])
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=0.5,
            retrieval_relevance_score=0.4,
            score_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.triggered is True


# ---------------------------------------------------------------------------
# Test 4 — score_entropy > 1.5 AND faithfulness_score < 0.5 → triggered=True
# ---------------------------------------------------------------------------

def test_entropy_and_faithfulness_trigger(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    query_emb = _unit(seed=1)
    q_emb = _unit(seed=2)
    chunks = _chunks(1)
    chunk_embs = [_unit(seed=10)]

    _make_claude_mock(mocker, '["What is Y?"]')
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock([q_emb])
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=0.5,
            retrieval_relevance_score=0.8,
            score_entropy=1.6,
            faithfulness_score=0.4,
        )

    assert result.triggered is True


# ---------------------------------------------------------------------------
# Test 5 — score_entropy > 1.5 alone (faithfulness OK) → triggered=False
# ---------------------------------------------------------------------------

def test_entropy_alone_does_not_trigger(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    mock_client = _make_claude_mock(mocker, "[]")
    query_emb = _unit(seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(seed=10)]

    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=query_emb,
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        query_isolation=0.5,
        retrieval_relevance_score=0.8,
        score_entropy=1.6,
        faithfulness_score=0.8,
    )

    assert result.triggered is False
    mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6 — faithfulness_score < 0.5 alone (entropy OK) → triggered=False
# ---------------------------------------------------------------------------

def test_faithfulness_alone_does_not_trigger(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    mock_client = _make_claude_mock(mocker, "[]")
    query_emb = _unit(seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(seed=10)]

    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=query_emb,
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        query_isolation=0.5,
        retrieval_relevance_score=0.8,
        score_entropy=0.5,
        faithfulness_score=0.4,
    )

    assert result.triggered is False
    mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 7 — triggered + Claude returns 3 questions → 3 SuggestedQuestion objects
# ---------------------------------------------------------------------------

def test_triggered_returns_expected_question_count(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    query_emb = _unit(seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(seed=10)]

    _make_claude_mock(mocker, '["Q1?", "Q2?", "Q3?"]')
    q_embs = [_unit(seed=20 + i) for i in range(3)]
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock(q_embs)
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.5,
            retrieval_relevance_score=0.8,
            score_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.triggered is True
    assert len(result.suggested_questions) == 3


# ---------------------------------------------------------------------------
# Test 8 — each SuggestedQuestion has question, source_chunk_ids, relevance_to_original
# ---------------------------------------------------------------------------

def test_suggested_question_fields(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    query_emb = _unit(seed=1)
    chunks = _chunks(2)
    chunk_embs = [_unit(seed=10 + i) for i in range(2)]

    _make_claude_mock(mocker, '["Tell me about A?"]')
    q_emb = _unit(seed=20)
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock([q_emb])
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.5,
            retrieval_relevance_score=0.8,
            score_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert len(result.suggested_questions) == 1
    sq = result.suggested_questions[0]
    assert isinstance(sq.question, str) and sq.question
    assert isinstance(sq.source_chunk_ids, list) and len(sq.source_chunk_ids) > 0
    assert isinstance(sq.relevance_to_original, float)


# ---------------------------------------------------------------------------
# Test 9 — source_chunk_ids contains only IDs from input chunks
# ---------------------------------------------------------------------------

def test_source_chunk_ids_from_input(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    query_emb = _unit(seed=1)
    chunks = _chunks(3)
    valid_ids = {c.chunk_id for c in chunks}
    chunk_embs = [_unit(seed=10 + i) for i in range(3)]

    _make_claude_mock(mocker, '["Q1?", "Q2?"]')
    q_embs = [_unit(seed=20 + i) for i in range(2)]
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock(q_embs)
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.5,
            retrieval_relevance_score=0.8,
            score_entropy=0.5,
            faithfulness_score=0.8,
        )

    for sq in result.suggested_questions:
        for cid in sq.source_chunk_ids:
            assert cid in valid_ids


# ---------------------------------------------------------------------------
# Test 10 — high mean similarity (> 0.6) → mismatch_type="query_mismatch"
# ---------------------------------------------------------------------------

def test_high_similarity_query_mismatch(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    dim = 8
    query_emb = _unit(dim=dim, seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(dim=dim, seed=10)]

    # question embeddings identical to query_emb → cosine sim = 1.0
    q_embs = [query_emb.copy() for _ in range(3)]

    _make_claude_mock(mocker, '["Q1?", "Q2?", "Q3?"]')
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock(q_embs)
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.5,
            retrieval_relevance_score=0.8,
            score_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.mean_question_similarity > 0.6
    assert result.mismatch_type == "query_mismatch"


# ---------------------------------------------------------------------------
# Test 11 — low mean similarity (< 0.3) → mismatch_type="coverage_gap"
# ---------------------------------------------------------------------------

def test_low_similarity_coverage_gap(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    dim = 8
    query_emb = _unit(dim=dim, seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(dim=dim, seed=10)]

    # question embeddings orthogonal to query_emb → cosine sim ≈ 0.0
    q_embs = [_orthogonal(query_emb, seed=99 + i) for i in range(3)]

    _make_claude_mock(mocker, '["Q1?", "Q2?", "Q3?"]')
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock(q_embs)
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.5,
            retrieval_relevance_score=0.8,
            score_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.mean_question_similarity < 0.3
    assert result.mismatch_type == "coverage_gap"


# ---------------------------------------------------------------------------
# Test 12 — mid-range similarity (0.3–0.6) → mismatch_type="ambiguous"
# ---------------------------------------------------------------------------

def test_mid_similarity_ambiguous(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    dim = 8
    query_emb = _unit(dim=dim, seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(dim=dim, seed=10)]

    # question embeddings at cosine sim ≈ 0.45 to query_emb
    q_embs = [_partial(query_emb, 0.45, seed=77 + i) for i in range(3)]

    _make_claude_mock(mocker, '["Q1?", "Q2?", "Q3?"]')
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock(q_embs)
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.5,
            retrieval_relevance_score=0.8,
            score_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert 0.3 <= result.mean_question_similarity <= 0.6
    assert result.mismatch_type == "ambiguous"


# ---------------------------------------------------------------------------
# Test 13 — Claude API exception → triggered=True, empty questions, no crash
# ---------------------------------------------------------------------------

def test_claude_exception_fallback(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    _make_claude_mock(mocker, Exception("API error"))
    query_emb = _unit(seed=1)
    chunks = _chunks(2)
    chunk_embs = [_unit(seed=10 + i) for i in range(2)]

    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=query_emb,
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        query_isolation=1.5,
        retrieval_relevance_score=0.8,
        score_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is True
    assert result.suggested_questions == []
    assert result.mismatch_type is None
    assert result.mean_question_similarity is None


# ---------------------------------------------------------------------------
# Test 14 — Claude returns invalid JSON → same fallback as API exception
# ---------------------------------------------------------------------------

def test_invalid_json_fallback(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    _make_claude_mock(mocker, "not valid json at all")
    query_emb = _unit(seed=1)
    chunks = _chunks(2)
    chunk_embs = [_unit(seed=10 + i) for i in range(2)]

    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=query_emb,
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        query_isolation=0.5,
        retrieval_relevance_score=0.4,
        score_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is True
    assert result.suggested_questions == []
    assert result.mismatch_type is None
    assert result.mean_question_similarity is None


def test_wrong_json_structure_fallback(mocker):
    """Claude returns valid JSON but not a list of strings (e.g. a dict) → same fallback."""
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    _make_claude_mock(mocker, '{"questions": ["Q1", "Q2"]}')
    query_emb = _unit(seed=1)
    chunks = _chunks(2)
    chunk_embs = [_unit(seed=10 + i) for i in range(2)]

    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=query_emb,
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        query_isolation=1.5,
        retrieval_relevance_score=0.8,
        score_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is True
    assert result.suggested_questions == []
    assert result.mismatch_type is None
    assert result.mean_question_similarity is None


# ---------------------------------------------------------------------------
# Test 15 — query_isolation exactly 1.2 → triggered=False (strict >)
# ---------------------------------------------------------------------------

def test_query_isolation_boundary_not_triggered(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    mock_client = _make_claude_mock(mocker, "[]")
    query_emb = _unit(seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(seed=10)]

    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=query_emb,
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        query_isolation=1.2,
        retrieval_relevance_score=0.8,
        score_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is False
    mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 16 — retrieval_relevance_score exactly 0.5 → triggered=False (strict <)
# ---------------------------------------------------------------------------

def test_retrieval_relevance_boundary_not_triggered(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    mock_client = _make_claude_mock(mocker, "[]")
    query_emb = _unit(seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(seed=10)]

    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=query_emb,
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        query_isolation=0.5,
        retrieval_relevance_score=0.5,
        score_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is False
    mock_client.messages.create.assert_not_called()
