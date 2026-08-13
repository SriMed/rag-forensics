"""Tests for query-corpus fit forensics module — written before implementation (TDD)."""
import json
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
        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            generated = [
                {"question": question, "source_chunk_ids": ["c0"]}
                for question in parsed
            ]
            validated = [
                {"question_index": index, "directly_answerable": True, "specific": True, "supporting_chunk_ids": ["c0"]}
                for index in range(len(generated))
            ]
            mock_client.messages.create.side_effect = [
                _mock_response(json.dumps(generated)),
                _mock_response(json.dumps(validated)),
            ]
        else:
            mock_client.messages.create.return_value = _mock_response(response)
    mock_cls = MagicMock(return_value=mock_client)
    mocker.patch("services.forensics.query_corpus_fit.anthropic.Anthropic", mock_cls)
    return mock_client


def _make_embed_mock(embeddings: list[np.ndarray]) -> MagicMock:
    """Return a mock embedding model. encode(texts) returns (len(texts), dim) array in one call."""
    model = MagicMock()
    model.encode.return_value = np.vstack([e.reshape(1, -1) for e in embeddings])
    return model


def _make_structured_claude_mock(mocker, generated: list[dict], judgments: list[dict]) -> MagicMock:
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_response(json.dumps(generated)),
        _mock_response(json.dumps(judgments)),
    ]
    mocker.patch(
        "services.forensics.query_corpus_fit.anthropic.Anthropic",
        MagicMock(return_value=mock_client),
    )
    return mock_client


# All-clear signal values — no trigger conditions met
_NO_TRIGGER = dict(
    query_isolation=0.5,
    context_utilization_score=0.8,
    normalized_entropy=0.5,
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
    assert result.observed_fit is None
    assert result.mean_question_similarity is None
    assert result.status == "not_run"
    assert result.error is None
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
            context_utilization_score=0.8,
            normalized_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.triggered is True


# ---------------------------------------------------------------------------
# Test 3 — context_utilization_score < 0.5 → triggered=True
# ---------------------------------------------------------------------------

def test_context_utilization_triggers(mocker):
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
            context_utilization_score=0.4,
            normalized_entropy=0.5,
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
            context_utilization_score=0.8,
            normalized_entropy=0.95,
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
        context_utilization_score=0.8,
        normalized_entropy=0.95,
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
        context_utilization_score=0.8,
        normalized_entropy=0.5,
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
            context_utilization_score=0.8,
            normalized_entropy=0.5,
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

    _make_claude_mock(mocker, '["Tell me about A?", "Tell me about B?", "Tell me about C?"]')
    q_embs = [_unit(seed=20 + i) for i in range(3)]
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock(q_embs)
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.5,
            context_utilization_score=0.8,
            normalized_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert len(result.suggested_questions) == 3
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
            context_utilization_score=0.8,
            normalized_entropy=0.5,
            faithfulness_score=0.8,
        )

    for sq in result.suggested_questions:
        for cid in sq.source_chunk_ids:
            assert cid in valid_ids


# ---------------------------------------------------------------------------
# Test 10 — high mean similarity (> 0.6) → observed_fit="retrieved_context_near_miss"
# ---------------------------------------------------------------------------

def test_high_similarity_query_mismatch(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    dim = 8
    query_emb = _unit(dim=dim, seed=1)
    chunks = _chunks(1)
    chunk_embs = [_unit(dim=dim, seed=10)]

    # Distinct question embeddings remain strongly similar to the original query.
    q_embs = [_partial(query_emb, 0.7, seed=90 + i) for i in range(3)]

    _make_claude_mock(mocker, '["Q1?", "Q2?", "Q3?"]')
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock(q_embs)
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=query_emb,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            query_isolation=1.5,
            context_utilization_score=0.8,
            normalized_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.mean_question_similarity > 0.6
    assert result.observed_fit == "retrieved_context_near_miss"


# ---------------------------------------------------------------------------
# Test 11 — low mean similarity (< 0.3) → observed_fit="retrieved_context_topic_gap"
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
            context_utilization_score=0.8,
            normalized_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.mean_question_similarity < 0.3
    assert result.observed_fit == "retrieved_context_topic_gap"


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
            context_utilization_score=0.8,
            normalized_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert 0.3 <= result.mean_question_similarity <= 0.6
    assert result.observed_fit == "ambiguous"


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
        context_utilization_score=0.8,
        normalized_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is True
    assert result.suggested_questions == []
    assert result.observed_fit is None
    assert result.mean_question_similarity is None
    assert result.status == "error"
    assert result.error == "question_generation_failed"


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
        context_utilization_score=0.4,
        normalized_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is True
    assert result.suggested_questions == []
    assert result.observed_fit is None
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
        context_utilization_score=0.8,
        normalized_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is True
    assert result.suggested_questions == []
    assert result.observed_fit is None
    assert result.mean_question_similarity is None


def test_embedding_failure_returns_explicit_error(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    _make_claude_mock(mocker, '["Q1?", "Q2?", "Q3?"]')
    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.side_effect = RuntimeError("model unavailable")
        result = analyze_query_corpus_fit(
            question="What is X?",
            query_embedding=_unit(seed=1),
            chunks=_chunks(2),
            chunk_embeddings=[_unit(seed=10), _unit(seed=11)],
            query_isolation=1.5,
            context_utilization_score=0.8,
            normalized_entropy=0.5,
            faithfulness_score=0.8,
        )

    assert result.status == "error"
    assert result.error == "fit_computation_failed"
    assert result.observed_fit is None


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
        context_utilization_score=0.8,
        normalized_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is False
    mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 16 — context_utilization_score exactly 0.5 → triggered=False (strict <)
# ---------------------------------------------------------------------------

def test_context_utilization_boundary_not_triggered(mocker):
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
        context_utilization_score=0.5,
        normalized_entropy=0.5,
        faithfulness_score=0.8,
    )

    assert result.triggered is False
    mock_client.messages.create.assert_not_called()


def test_unavailable_ragas_scores_do_not_trigger(mocker):
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    mock_client = _make_claude_mock(mocker, "[]")
    result = analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=_unit(seed=1),
        chunks=_chunks(1),
        chunk_embeddings=[_unit(seed=10)],
        query_isolation=0.5,
        context_utilization_score=None,
        normalized_entropy=0.95,
        faithfulness_score=None,
    )

    assert result.triggered is False
    mock_client.messages.create.assert_not_called()


def test_unsupported_question_is_rejected_and_fit_is_unavailable(mocker):
    generated = [
        {"question": f"Question {i}?", "source_chunk_ids": ["c0"]}
        for i in range(3)
    ]
    judgments = [
        {"question_index": 0, "directly_answerable": True, "specific": True, "supporting_chunk_ids": ["c0"]},
        {"question_index": 1, "directly_answerable": True, "specific": True, "supporting_chunk_ids": ["c0"]},
        {"question_index": 2, "directly_answerable": False, "specific": True, "supporting_chunk_ids": []},
    ]
    _make_structured_claude_mock(mocker, generated, judgments)

    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock([_unit(seed=20), _unit(seed=21)])
        result = analyze_query_corpus_fit_for_test()

    assert result.status == "error"
    assert result.error == "insufficient_valid_questions"
    assert result.observed_fit is None
    assert len(result.suggested_questions) == 2
    assert result.rejected_questions[0].reason == "unsupported"


def test_semantic_duplicate_is_rejected(mocker):
    generated = [
        {"question": f"Question {i}?", "source_chunk_ids": ["c0"]}
        for i in range(4)
    ]
    judgments = [
        {"question_index": i, "directly_answerable": True, "specific": True, "supporting_chunk_ids": ["c0"]}
        for i in range(4)
    ]
    _make_structured_claude_mock(mocker, generated, judgments)
    distinct = [_unit(seed=30), _unit(seed=31), _unit(seed=32)]
    embeddings = [distinct[0], distinct[0], distinct[1], distinct[2]]

    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock(embeddings)
        result = analyze_query_corpus_fit_for_test()

    assert result.status == "ok"
    assert len(result.suggested_questions) == 3
    assert [item.reason for item in result.rejected_questions] == ["semantic_duplicate"]


def test_unknown_supporting_chunk_is_rejected_before_validation(mocker):
    generated = [
        {"question": "Injected instruction question?", "source_chunk_ids": ["not-a-real-chunk"]},
        {"question": "Question 1?", "source_chunk_ids": ["c0"]},
        {"question": "Question 2?", "source_chunk_ids": ["c0"]},
        {"question": "Question 3?", "source_chunk_ids": ["c0"]},
    ]
    judgments = [
        {"question_index": i, "directly_answerable": True, "specific": True, "supporting_chunk_ids": ["c0"]}
        for i in range(3)
    ]
    mock_client = _make_structured_claude_mock(mocker, generated, judgments)

    with patch("services.forensics.query_corpus_fit.get_embedding_model") as mock_get:
        mock_get.return_value = _make_embed_mock([_unit(seed=40 + i) for i in range(3)])
        result = analyze_query_corpus_fit_for_test()

    assert result.status == "ok"
    assert result.rejected_questions[0].reason == "invalid_source_chunk"
    validation_prompt = mock_client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert "Injected instruction question?" not in validation_prompt


def analyze_query_corpus_fit_for_test():
    from services.forensics.query_corpus_fit import analyze_query_corpus_fit

    return analyze_query_corpus_fit(
        question="What is X?",
        query_embedding=_unit(seed=1),
        chunks=_chunks(1),
        chunk_embeddings=[_unit(seed=10)],
        query_isolation=1.5,
        context_utilization_score=0.8,
        normalized_entropy=0.5,
        faithfulness_score=0.8,
    )
