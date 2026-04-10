"""Tests for chunk attribution forensics module — written before implementation (TDD)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from models import RetrievedChunk


def make_chunks(n: int) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk_id=f"chunk_{i}", text=f"text {i}", score=0.9) for i in range(n)]


def make_embedding(dim: int = 8) -> np.ndarray:
    v = np.random.default_rng(42).random(dim)
    return v / np.linalg.norm(v)


def make_parallel_embeddings(n: int, dim: int = 8) -> list[np.ndarray]:
    """Return n embeddings all pointing in the same direction — cosine sim = 1.0."""
    base = make_embedding(dim)
    return [base.copy() for _ in range(n)]


def make_orthogonal_embedding(dim: int = 8) -> np.ndarray:
    """Return an embedding orthogonal to make_embedding() — cosine sim ≈ 0."""
    rng = np.random.default_rng(99)
    base = make_embedding(dim)
    v = rng.random(dim)
    v = v - np.dot(v, base) * base
    v = v / np.linalg.norm(v)
    return v


def _mock_model(sentence_embeddings: np.ndarray):
    """Return a mock embedding model whose .encode() returns the given array."""
    model = MagicMock()
    model.encode.return_value = sentence_embeddings
    return model


# ---------------------------------------------------------------------------
# Test 1 — all sentences strongly attributed (similarity > 0.75)
# ---------------------------------------------------------------------------

def test_all_strong_attribution():
    from services.forensics.chunk_attribution import analyze_chunk_attribution

    chunks = make_chunks(2)
    chunk_embs = make_parallel_embeddings(2)
    sentence_emb = make_parallel_embeddings(1)[0].reshape(1, -1)

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_get.return_value = _mock_model(sentence_emb)
        result = analyze_chunk_attribution(
            answer="The sky is blue.",
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    assert result.unattributed_fraction == 0.0
    assert all(e.attribution_strength == "strong" for e in result.attribution_map)


# ---------------------------------------------------------------------------
# Test 2 — one sentence unattributed (similarity < 0.4)
# ---------------------------------------------------------------------------

def test_one_sentence_unattributed():
    from services.forensics.chunk_attribution import analyze_chunk_attribution

    chunks = make_chunks(2)
    chunk_embs = make_parallel_embeddings(2)
    ortho = make_orthogonal_embedding().reshape(1, -1)

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_get.return_value = _mock_model(ortho)
        result = analyze_chunk_attribution(
            answer="One sentence only.",
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    assert result.unattributed_fraction == pytest.approx(1.0)
    assert result.attribution_map[0].chunk_id is None
    assert result.attribution_map[0].attribution_strength == "unattributed"


# ---------------------------------------------------------------------------
# Test 3 — all sentences weakly matched (0.4 < sim < 0.75)
# ---------------------------------------------------------------------------

def test_all_weak_attribution():
    from services.forensics.chunk_attribution import analyze_chunk_attribution

    rng = np.random.default_rng(7)
    dim = 8

    # Build a chunk embedding
    base = rng.random(dim)
    base /= np.linalg.norm(base)

    # Build a sentence embedding that gives cosine sim ≈ 0.55 with base
    # We do: s = alpha*base + beta*perp, choosing alpha/beta so dot(s_norm, base) ≈ 0.55
    perp = rng.random(dim)
    perp -= np.dot(perp, base) * base
    perp /= np.linalg.norm(perp)

    alpha, beta = 0.55, np.sqrt(1 - 0.55**2)
    s = alpha * base + beta * perp
    s /= np.linalg.norm(s)

    chunks = make_chunks(1)
    chunk_embs = [base.tolist()]

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_get.return_value = _mock_model(s.reshape(1, -1))
        result = analyze_chunk_attribution(
            answer="One sentence only.",
            chunks=chunks,
            chunk_embeddings=chunk_embs,
        )

    assert result.weak_match_fraction == pytest.approx(1.0)
    assert result.unattributed_fraction == pytest.approx(0.0)
    assert result.attribution_map[0].attribution_strength == "weak"


# ---------------------------------------------------------------------------
# Test 4 — single-sentence answer does not crash
# ---------------------------------------------------------------------------

def test_single_sentence_no_crash():
    from services.forensics.chunk_attribution import analyze_chunk_attribution

    chunks = make_chunks(1)
    chunk_embs = make_parallel_embeddings(1)
    sent_emb = make_parallel_embeddings(1)[0].reshape(1, -1)

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_get.return_value = _mock_model(sent_emb)
        result = analyze_chunk_attribution(
            answer="Single sentence.",
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    assert len(result.attribution_map) == 1


# ---------------------------------------------------------------------------
# Test 5 — empty answer returns zeroed metrics and empty map
# ---------------------------------------------------------------------------

def test_empty_answer_returns_zeros():
    from services.forensics.chunk_attribution import analyze_chunk_attribution

    chunks = make_chunks(2)
    chunk_embs = make_parallel_embeddings(2)

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        result = analyze_chunk_attribution(
            answer="",
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    assert result.unattributed_fraction == 0.0
    assert result.mean_attribution_score == 0.0
    assert result.weak_match_fraction == 0.0
    assert result.attribution_map == []
    mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6 — attribution_map length equals sentence count
# ---------------------------------------------------------------------------

def test_attribution_map_length_equals_sentence_count():
    from services.forensics.chunk_attribution import analyze_chunk_attribution
    import nltk

    answer = "The sky is blue. Water is wet. Fire is hot."
    chunks = make_chunks(2)
    chunk_embs = make_parallel_embeddings(2)

    # Determine expected sentence count via nltk
    nltk.download("punkt_tab", quiet=True)
    expected = len(nltk.sent_tokenize(answer))

    sent_embs = make_parallel_embeddings(expected)
    sent_matrix = np.stack(sent_embs)

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_get.return_value = _mock_model(sent_matrix)
        result = analyze_chunk_attribution(
            answer=answer,
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    assert len(result.attribution_map) == expected


# ---------------------------------------------------------------------------
# Test 7 — chunk_id is None for all unattributed entries
# ---------------------------------------------------------------------------

def test_chunk_id_none_for_unattributed():
    from services.forensics.chunk_attribution import analyze_chunk_attribution

    chunks = make_chunks(2)
    chunk_embs = make_parallel_embeddings(2)

    # All sentences orthogonal → all unattributed
    answer = "First sentence. Second sentence."
    import nltk
    nltk.download("punkt_tab", quiet=True)
    n = len(nltk.sent_tokenize(answer))
    ortho_base = make_orthogonal_embedding()
    sent_matrix = np.stack([ortho_base] * n)

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_get.return_value = _mock_model(sent_matrix)
        result = analyze_chunk_attribution(
            answer=answer,
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    for entry in result.attribution_map:
        if entry.attribution_strength == "unattributed":
            assert entry.chunk_id is None


# ---------------------------------------------------------------------------
# Test 8 — mean_attribution_score equals mean of best similarity scores
# ---------------------------------------------------------------------------

def test_mean_attribution_score_is_mean_of_best_scores():
    from services.forensics.chunk_attribution import analyze_chunk_attribution

    chunks = make_chunks(2)
    chunk_embs = make_parallel_embeddings(2)  # identical → same sim for both

    answer = "First sentence. Second sentence."
    import nltk
    nltk.download("punkt_tab", quiet=True)
    n = len(nltk.sent_tokenize(answer))
    parallel = make_parallel_embeddings(n)
    sent_matrix = np.stack(parallel)

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_get.return_value = _mock_model(sent_matrix)
        result = analyze_chunk_attribution(
            answer=answer,
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    expected_mean = float(np.mean([e.similarity_score for e in result.attribution_map]))
    assert result.mean_attribution_score == pytest.approx(expected_mean, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 9 — embedding model called only for sentences, not chunks
# ---------------------------------------------------------------------------

def test_embedding_model_called_only_for_sentences():
    from services.forensics.chunk_attribution import analyze_chunk_attribution

    chunks = make_chunks(2)
    chunk_embs = make_parallel_embeddings(2)
    sent_emb = make_parallel_embeddings(1)[0].reshape(1, -1)

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_model = _mock_model(sent_emb)
        mock_get.return_value = mock_model
        analyze_chunk_attribution(
            answer="Single sentence.",
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    # encode must be called exactly once (for sentences), never with chunk texts
    mock_model.encode.assert_called_once()


# ---------------------------------------------------------------------------
# Test 10 — fractions sum to 1.0
# ---------------------------------------------------------------------------

def test_fractions_sum_to_one():
    from services.forensics.chunk_attribution import analyze_chunk_attribution
    import nltk

    answer = "The sky is blue. Water is wet. Fire is hot."
    nltk.download("punkt_tab", quiet=True)
    n = len(nltk.sent_tokenize(answer))

    chunks = make_chunks(2)
    chunk_embs = make_parallel_embeddings(2)
    sent_matrix = np.stack(make_parallel_embeddings(n))

    with patch("services.forensics.chunk_attribution.get_embedding_model") as mock_get:
        mock_get.return_value = _mock_model(sent_matrix)
        result = analyze_chunk_attribution(
            answer=answer,
            chunks=chunks,
            chunk_embeddings=[e.tolist() for e in chunk_embs],
        )

    strong_fraction = sum(
        1 for e in result.attribution_map if e.attribution_strength == "strong"
    ) / n
    total = result.unattributed_fraction + result.weak_match_fraction + strong_fraction
    assert total == pytest.approx(1.0, abs=1e-6)