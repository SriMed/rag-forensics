import numpy as np
import pytest
from models import RetrievedChunk, RetrievalDistributionMetrics
from services.forensics.retrieval_distribution import analyze_retrieval_distribution


def _chunks(scores: list[float]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=f"c{i}", text=f"chunk {i}", score=s)
        for i, s in enumerate(scores)
    ]


# ---------------------------------------------------------------------------
# score_gap
# ---------------------------------------------------------------------------

def test_score_gap_is_top_minus_second():
    result = analyze_retrieval_distribution(_chunks([0.91, 0.63, 0.61, 0.60, 0.58]))
    assert abs(result.score_gap - 0.28) < 1e-6


# ---------------------------------------------------------------------------
# score_entropy
# ---------------------------------------------------------------------------

def test_equal_scores_give_high_entropy():
    equal = analyze_retrieval_distribution(_chunks([0.70, 0.70, 0.70, 0.70, 0.70]))
    steep = analyze_retrieval_distribution(_chunks([0.95, 0.50, 0.20, 0.10, 0.05]))
    assert equal.score_entropy > steep.score_entropy


def test_steep_drop_gives_low_entropy():
    result = analyze_retrieval_distribution(_chunks([0.95, 0.50, 0.20, 0.10, 0.05]))
    equal = analyze_retrieval_distribution(_chunks([0.70, 0.70, 0.70, 0.70, 0.70]))
    assert result.score_entropy < equal.score_entropy


def test_score_entropy_is_always_nonnegative():
    for scores in [
        [0.91, 0.63, 0.61, 0.60, 0.58],
        [0.70, 0.70, 0.70, 0.70, 0.70],
        [0.95, 0.50, 0.20, 0.10, 0.05],
        [1.0],
        [0.8, 0.2],
    ]:
        result = analyze_retrieval_distribution(_chunks(scores))
        assert result.score_entropy >= 0.0, f"Negative entropy for scores {scores}"


# ---------------------------------------------------------------------------
# decay_rate
# ---------------------------------------------------------------------------

def test_steep_drop_gives_high_decay_rate():
    steep = analyze_retrieval_distribution(_chunks([0.95, 0.50, 0.20, 0.10, 0.05]))
    equal = analyze_retrieval_distribution(_chunks([0.70, 0.70, 0.70, 0.70, 0.70]))
    assert steep.decay_rate > equal.decay_rate


def test_flat_distribution_decay_rate_falls_back_to_zero():
    result = analyze_retrieval_distribution(_chunks([0.70, 0.70, 0.70, 0.70, 0.70]))
    assert result.decay_rate == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# tail_mass
# ---------------------------------------------------------------------------

def test_tail_mass_is_between_zero_and_one():
    for scores in [
        [0.91, 0.63, 0.61, 0.60, 0.58],
        [0.70, 0.70, 0.70, 0.70, 0.70],
        [0.95, 0.50, 0.20, 0.10, 0.05],
        [0.8, 0.2],
        [1.0],
    ]:
        result = analyze_retrieval_distribution(_chunks(scores))
        assert 0.0 <= result.tail_mass <= 1.0, f"tail_mass out of range for {scores}"


# ---------------------------------------------------------------------------
# edge cases: single and two chunks
# ---------------------------------------------------------------------------

def test_single_chunk_score_gap_is_zero():
    result = analyze_retrieval_distribution(_chunks([0.85]))
    assert result.score_gap == pytest.approx(0.0)


def test_single_chunk_tail_mass_is_zero():
    result = analyze_retrieval_distribution(_chunks([0.85]))
    assert result.tail_mass == pytest.approx(0.0)


def test_single_chunk_no_crash():
    result = analyze_retrieval_distribution(_chunks([0.85]))
    assert isinstance(result, RetrievalDistributionMetrics)


def test_two_chunks_tail_mass_is_zero():
    result = analyze_retrieval_distribution(_chunks([0.8, 0.2]))
    assert result.tail_mass == pytest.approx(0.0)


def test_two_chunks_no_crash():
    result = analyze_retrieval_distribution(_chunks([0.8, 0.2]))
    assert isinstance(result, RetrievalDistributionMetrics)


# ---------------------------------------------------------------------------
# n_chunks and top_score
# ---------------------------------------------------------------------------

def test_n_chunks_equals_input_length():
    chunks = _chunks([0.91, 0.63, 0.61, 0.60, 0.58])
    result = analyze_retrieval_distribution(chunks)
    assert result.n_chunks == 5


def test_top_score_equals_highest_score():
    result = analyze_retrieval_distribution(_chunks([0.63, 0.91, 0.61, 0.60, 0.58]))
    assert result.top_score == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# type checks
# ---------------------------------------------------------------------------

def test_all_float_metrics_are_floats():
    result = analyze_retrieval_distribution(_chunks([0.91, 0.63, 0.61, 0.60, 0.58]))
    assert isinstance(result.score_gap, float)
    assert isinstance(result.score_entropy, float)
    assert isinstance(result.decay_rate, float)
    assert isinstance(result.tail_mass, float)
    assert isinstance(result.top_score, float)


def test_n_chunks_is_int():
    result = analyze_retrieval_distribution(_chunks([0.91, 0.63, 0.61, 0.60, 0.58]))
    assert isinstance(result.n_chunks, int)
