"""Tests for embedding space analysis module (Issue 5b / #14).

All embeddings are synthetic numpy arrays — no real model calls.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from models import EmbeddingSpaceMetrics, EmbeddingPoint
from services.forensics.embedding_analysis import analyze_embedding_space

DIM = 64
RNG = np.random.default_rng(42)


def _rand_unit(shape):
    """Return a random unit vector (or matrix of unit rows)."""
    v = RNG.standard_normal(shape).astype(np.float32)
    if v.ndim == 1:
        return v / np.linalg.norm(v)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


# ---------------------------------------------------------------------------
# Test 1 — query at centroid → low centroid_distance, query_isolation < 1.0
# ---------------------------------------------------------------------------

def test_query_at_centroid_has_low_centroid_distance():
    chunks = _rand_unit((5, DIM))
    centroid = chunks.mean(axis=0)
    query = centroid / np.linalg.norm(centroid)  # unit vector toward centroid
    result = analyze_embedding_space(query, list(chunks), [f"c{i}" for i in range(5)])
    assert result.centroid_distance < 0.1


def test_query_at_centroid_has_isolation_below_one():
    chunks = _rand_unit((5, DIM))
    centroid = chunks.mean(axis=0)
    query = centroid / np.linalg.norm(centroid)
    result = analyze_embedding_space(query, list(chunks), [f"c{i}" for i in range(5)])
    assert result.query_isolation < 1.0


# ---------------------------------------------------------------------------
# Test 2 — query far from all chunks → high centroid_distance, isolation > 1.0
# ---------------------------------------------------------------------------

def test_query_far_from_chunks_has_high_centroid_distance():
    # Place chunks in positive half of first axis, query in negative half
    chunks = np.zeros((5, DIM), dtype=np.float32)
    chunks[:, 0] = 1.0  # all chunks point in +x direction
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = -1.0  # query points in -x direction (maximally far)
    result = analyze_embedding_space(query, list(chunks), [f"c{i}" for i in range(5)])
    assert result.centroid_distance > 0.5


def test_query_far_from_chunks_has_isolation_above_one():
    chunks = np.zeros((5, DIM), dtype=np.float32)
    # Give each chunk a slight perturbation so they're not identical
    for i in range(5):
        chunks[i, 0] = 1.0
        chunks[i, i + 1] = 0.05
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = -1.0
    result = analyze_embedding_space(query, list(chunks), [f"c{i}" for i in range(5)])
    assert result.query_isolation > 1.0


# ---------------------------------------------------------------------------
# Test 3 — all chunks identical → chunk_spread ≈ 0.0
# ---------------------------------------------------------------------------

def test_identical_chunks_give_zero_spread():
    base = _rand_unit(DIM)
    chunks = [base.copy() for _ in range(5)]
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, chunks, [f"c{i}" for i in range(5)])
    assert result.chunk_spread == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 4 — diverse chunks → chunk_spread > 0
# ---------------------------------------------------------------------------

def test_diverse_chunks_give_positive_spread():
    chunks = _rand_unit((5, DIM))
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, list(chunks), [f"c{i}" for i in range(5)])
    assert result.chunk_spread > 0.0


# ---------------------------------------------------------------------------
# Test 5 — projection length equals len(chunk_ids) + 1
# ---------------------------------------------------------------------------

def test_projection_length_equals_chunks_plus_one():
    n = 4
    chunks = _rand_unit((n, DIM))
    query = _rand_unit(DIM)
    chunk_ids = [f"chunk_{i}" for i in range(n)]
    result = analyze_embedding_space(query, list(chunks), chunk_ids)
    assert len(result.projection) == n + 1


# ---------------------------------------------------------------------------
# Test 6 — exactly one point has is_query=True
# ---------------------------------------------------------------------------

def test_exactly_one_query_point():
    chunks = _rand_unit((5, DIM))
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, list(chunks), [f"c{i}" for i in range(5)])
    query_points = [p for p in result.projection if p.is_query]
    assert len(query_points) == 1
    assert query_points[0].label == "query"


# ---------------------------------------------------------------------------
# Test 7 — single chunk input → chunk_spread=0.0, no crash
# ---------------------------------------------------------------------------

def test_single_chunk_spread_is_zero():
    chunk = _rand_unit(DIM)
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, [chunk], ["c0"])
    assert result.chunk_spread == pytest.approx(0.0)


def test_single_chunk_no_crash():
    chunk = _rand_unit(DIM)
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, [chunk], ["c0"])
    assert isinstance(result, EmbeddingSpaceMetrics)


def test_single_chunk_projection_has_two_points():
    chunk = _rand_unit(DIM)
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, [chunk], ["c0"])
    assert len(result.projection) == 2


# ---------------------------------------------------------------------------
# Test 8 — centroid_distance >= 0 always
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_chunks", [1, 2, 5, 10])
def test_centroid_distance_nonnegative(n_chunks):
    chunks = _rand_unit((n_chunks, DIM))
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, list(chunks), [f"c{i}" for i in range(n_chunks)])
    assert result.centroid_distance >= 0.0


# ---------------------------------------------------------------------------
# Test 9 — query_isolation >= 0 always
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_chunks", [1, 2, 5, 10])
def test_query_isolation_nonnegative(n_chunks):
    chunks = _rand_unit((n_chunks, DIM))
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, list(chunks), [f"c{i}" for i in range(n_chunks)])
    assert result.query_isolation >= 0.0


# ---------------------------------------------------------------------------
# Test 10 — function makes no embedding model calls
# ---------------------------------------------------------------------------

def test_no_embedding_model_called():
    """Function accepts pre-computed embeddings — no embedding library should be imported."""
    import services.forensics.embedding_analysis as mod
    import sys

    # Known embedding-model libraries that must NOT be imported by this module
    forbidden_modules = {
        "sentence_transformers",
        "openai",
        "anthropic",
        "transformers",
        "torch",
    }
    mod_globals = vars(mod)
    for name, obj in mod_globals.items():
        module_name = getattr(obj, "__name__", None) or getattr(obj, "__module__", None) or ""
        assert not any(forbidden in module_name for forbidden in forbidden_modules), (
            f"Embedding model library detected in module attribute '{name}': {module_name}"
        )
    # Also confirm none of the forbidden packages were imported by the module
    for forbidden in forbidden_modules:
        assert forbidden not in sys.modules or not any(
            getattr(v, "__module__", "").startswith(forbidden)
            for v in mod_globals.values()
            if callable(v)
        ), f"Forbidden embedding library '{forbidden}' is in use"


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------

def test_return_type_is_embedding_space_metrics():
    chunks = _rand_unit((3, DIM))
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, list(chunks), ["a", "b", "c"])
    assert isinstance(result, EmbeddingSpaceMetrics)


def test_projection_points_are_embedding_point_instances():
    chunks = _rand_unit((3, DIM))
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, list(chunks), ["a", "b", "c"])
    for p in result.projection:
        assert isinstance(p, EmbeddingPoint)


def test_projection_chunk_labels_match_chunk_ids():
    chunk_ids = ["doc1_chunk0", "doc2_chunk1", "doc3_chunk2"]
    chunks = _rand_unit((3, DIM))
    query = _rand_unit(DIM)
    result = analyze_embedding_space(query, list(chunks), chunk_ids)
    non_query_labels = [p.label for p in result.projection if not p.is_query]
    assert non_query_labels == chunk_ids
