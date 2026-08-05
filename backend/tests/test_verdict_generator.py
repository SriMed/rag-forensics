"""Tests for verdict_generator.py — ranked signals approach.

rank_signals() is pure Python — no mocks needed.
render_recommendation() mocks services.verdict_generator.anthropic.Anthropic.
"""
import random
import pytest
from unittest.mock import MagicMock, patch

from models import (
    ChunkAttributionMetrics,
    EmbeddingSpaceMetrics,
    HedgingMismatchMetrics,
    QueryCorpusFitMetrics,
    RetrievalDistributionMetrics,
)
from services.verdict_generator import RankedSignal, rank_signals, render_recommendation
from signal_weights import SignalWeights


# ---------------------------------------------------------------------------
# Helpers — build minimal metric objects with sane defaults
# ---------------------------------------------------------------------------

def _distribution(**kwargs) -> RetrievalDistributionMetrics:
    defaults = dict(score_gap=0.05, score_entropy=0.5, decay_rate=0.3,
                    tail_mass=0.1, top_score=0.9, n_chunks=5,
                    normalized_entropy=0.5 / 1.609437912)
    defaults.update(kwargs)
    if "score_entropy" in kwargs and "normalized_entropy" not in kwargs:
        defaults["normalized_entropy"] = kwargs["score_entropy"] / 1.609437912
    return RetrievalDistributionMetrics(**defaults)


def _embedding(**kwargs) -> EmbeddingSpaceMetrics:
    defaults = dict(centroid_distance=0.2, chunk_spread=0.1,
                    query_isolation=0.8, projection=[])
    defaults.update(kwargs)
    return EmbeddingSpaceMetrics(**defaults)


def _attribution(**kwargs) -> ChunkAttributionMetrics:
    defaults = dict(unattributed_fraction=0.0, mean_attribution_score=0.9,
                    weak_match_fraction=0.0, attribution_map=[])
    defaults.update(kwargs)
    return ChunkAttributionMetrics(**defaults)


def _hedging(**kwargs) -> HedgingMismatchMetrics:
    defaults = dict(overconfident_fraction=0.0, underconfident_fraction=0.0,
                    total_claims=5, claim_breakdown=[])
    defaults.update(kwargs)
    return HedgingMismatchMetrics(**defaults)


def _query_fit(**kwargs) -> QueryCorpusFitMetrics:
    defaults = dict(triggered=False, observed_fit=None,
                    suggested_questions=[], mean_question_similarity=None)
    defaults.update(kwargs)
    return QueryCorpusFitMetrics(**defaults)


def _rank(**kwargs) -> list[RankedSignal]:
    """Call rank_signals with clean defaults, overriding via kwargs passed to each helper."""
    return rank_signals(
        distribution=_distribution(**{k: v for k, v in kwargs.items() if k in RetrievalDistributionMetrics.model_fields}),
        embedding=_embedding(**{k: v for k, v in kwargs.items() if k in EmbeddingSpaceMetrics.model_fields}),
        faithfulness_score=kwargs.get("faithfulness_score", 0.9),
        context_utilization_score=kwargs.get("context_utilization_score", 0.9),
        attribution=_attribution(**{k: v for k, v in kwargs.items() if k in ChunkAttributionMetrics.model_fields}),
        hedging_mismatch=_hedging(**{k: v for k, v in kwargs.items() if k in HedgingMismatchMetrics.model_fields}),
        query_fit=_query_fit(**{k: v for k, v in kwargs.items() if k in QueryCorpusFitMetrics.model_fields}),
        weights=kwargs.get("weights"),
    )


# ---------------------------------------------------------------------------
# rank_signals — sorting and structure
# ---------------------------------------------------------------------------

def test_rank_signals_returns_list_of_ranked_signals():
    signals = _rank()
    assert isinstance(signals, list)
    assert all(isinstance(s, RankedSignal) for s in signals)


def test_analysis_errors_are_ranked_as_unavailable_not_healthy():
    signals = rank_signals(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(status="error", error="claim_extraction_failed"),
        query_fit=_query_fit(status="error", error="fit_computation_failed"),
    )
    names = {signal.name for signal in signals}
    assert "hedging_analysis_unavailable" in names
    assert "retrieved_context_fit_unavailable" in names


def test_unavailable_ragas_metrics_are_ranked_without_numeric_concerns():
    signals = _rank(faithfulness_score=None, context_utilization_score=None)
    names = {signal.name for signal in signals}
    assert "faithfulness_unavailable" in names
    assert "context_utilization_unavailable" in names
    assert "low_faithfulness" not in names
    assert "low_context_utilization" not in names


def test_rank_signals_sorted_descending_by_concern():
    signals = rank_signals(
        distribution=_distribution(score_entropy=1.8, tail_mass=0.5),
        embedding=_embedding(query_isolation=1.5),
        faithfulness_score=0.4,
        context_utilization_score=0.3,
        attribution=_attribution(unattributed_fraction=0.6, weak_match_fraction=0.2),
        hedging_mismatch=_hedging(overconfident_fraction=0.5),
        query_fit=_query_fit(),
    )
    scores = [s.priority_score for s in signals]
    assert scores == sorted(scores, reverse=True)


def test_rank_signals_high_unattributed_is_top_signal():
    signals = rank_signals(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(unattributed_fraction=0.8),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert signals[0].name == "unattributed_content"
    assert signals[0].priority_score == pytest.approx(0.8)


def test_rank_signals_corpus_coverage_gap_is_high_concern():
    signals = rank_signals(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(triggered=True, observed_fit="retrieved_context_topic_gap"),
    )
    gap_signal = next(s for s in signals if s.name == "retrieved_context_topic_gap")
    assert gap_signal.priority_score == pytest.approx(0.9)


def test_rank_signals_query_mismatch_present_when_triggered():
    signals = rank_signals(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(triggered=True, observed_fit="retrieved_context_near_miss"),
    )
    names = [s.name for s in signals]
    assert "retrieved_context_near_miss" in names


def test_rank_signals_overconfidence_appears():
    signals = rank_signals(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(overconfident_fraction=0.5),
        query_fit=_query_fit(),
    )
    names = [s.name for s in signals]
    assert "overconfidence" in names
    overconfidence = next(s for s in signals if s.name == "overconfidence")
    assert overconfidence.priority_score == pytest.approx(0.5)


def test_rank_signals_does_not_infer_underconfidence_from_binary_entailment():
    signals = rank_signals(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(overconfident_fraction=0.4, underconfident_fraction=0.4),
        query_fit=_query_fit(),
    )
    names = [s.name for s in signals]
    assert "overconfidence" in names
    assert "underconfidence" not in names


def test_rank_signals_clean_metrics_top_concern_is_low():
    signals = rank_signals(
        distribution=_distribution(score_entropy=0.2, tail_mass=0.05),
        embedding=_embedding(query_isolation=0.5),
        faithfulness_score=0.95,
        context_utilization_score=0.95,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert signals[0].priority_score < 0.2


def test_rank_signals_untriggered_query_fit_adds_no_fit_signal():
    signals = rank_signals(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(triggered=False),
    )
    names = [s.name for s in signals]
    assert "retrieved_context_topic_gap" not in names
    assert "retrieved_context_near_miss" not in names


def test_rank_signals_decay_rate_none_does_not_crash():
    """decay_rate=None (fit failure) must not raise."""
    signals = rank_signals(
        distribution=_distribution(decay_rate=None),
        embedding=_embedding(),
        faithfulness_score=0.7,
        context_utilization_score=0.7,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert isinstance(signals, list)


def test_rank_signals_fuzz_never_crashes():
    """Random valid inputs should never raise and result should be sorted."""
    random.seed(0)
    for _ in range(50):
        signals = rank_signals(
            distribution=_distribution(
                score_gap=random.uniform(0, 1),
                score_entropy=random.uniform(0, 3),
                decay_rate=random.choice([None, random.uniform(0, 1)]),
                tail_mass=random.uniform(0, 1),
            ),
            embedding=_embedding(
                chunk_spread=random.uniform(0, 1),
                query_isolation=random.uniform(0, 2),
            ),
            faithfulness_score=random.uniform(0, 1),
            context_utilization_score=random.uniform(0, 1),
            attribution=_attribution(
                unattributed_fraction=random.uniform(0, 1),
                weak_match_fraction=random.uniform(0, 1),
            ),
            hedging_mismatch=_hedging(
                overconfident_fraction=random.uniform(0, 1),
                underconfident_fraction=random.uniform(0, 1),
            ),
            query_fit=_query_fit(),
        )
        scores = [s.priority_score for s in signals]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s.priority_score <= 1.0 for s in signals)


# ---------------------------------------------------------------------------
# render_recommendation
# ---------------------------------------------------------------------------

def test_render_recommendation_returns_string():
    signals = [RankedSignal(name="low_faithfulness", priority_score=0.6, description="Low faithfulness.", reliability="model_judged")]
    attr = _attribution()
    hed = _hedging()

    fake_message = MagicMock()
    fake_message.content = [MagicMock(text="Investigate your prompt template — faithfulness is low.")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_message

    with patch("services.verdict_generator.anthropic.Anthropic", return_value=fake_client):
        result = render_recommendation(signals, 0.4, 0.8, attr, hed)

    assert isinstance(result, str)
    assert len(result) > 0


def test_render_recommendation_claude_failure_falls_back_to_top_signal():
    signals = [RankedSignal(name="overconfidence", priority_score=0.7, description="Claims are overconfident.", reliability="model_judged")]
    attr = _attribution()
    hed = _hedging()

    with patch("services.verdict_generator.anthropic.Anthropic", side_effect=Exception("API down")):
        result = render_recommendation(signals, 0.5, 0.5, attr, hed)

    assert result == "Claims are overconfident."


def test_render_recommendation_empty_signals_returns_no_issues():
    result = render_recommendation([], 0.95, 0.95, _attribution(), _hedging())
    assert "no significant issues" in result.lower()


# ---------------------------------------------------------------------------
# SignalWeights — custom weights parameter
# ---------------------------------------------------------------------------

def test_rank_signals_accepts_custom_weights():
    """rank_signals must accept a SignalWeights instance without raising."""
    w = SignalWeights()
    signals = _rank(weights=w)
    assert isinstance(signals, list)


def test_rank_signals_entropy_uses_top_k_invariant_normalized_value():
    """Entropy priority should use the metric's normalized, top-k-invariant value."""
    default_signals = rank_signals(
        distribution=_distribution(score_entropy=1.0),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    changed_raw_signals = rank_signals(
        distribution=_distribution(score_entropy=0.2, normalized_entropy=1.0 / 1.609437912),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    default_entropy = next(s for s in default_signals if s.name == "ambiguous_retrieval")
    changed_raw_entropy = next(s for s in changed_raw_signals if s.name == "ambiguous_retrieval")
    assert changed_raw_entropy.priority_score == pytest.approx(default_entropy.priority_score)


def test_underconfidence_compatibility_field_does_not_affect_ranking():
    signals = rank_signals(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(underconfident_fraction=0.5),
        query_fit=_query_fit(),
    )
    assert all(signal.name != "underconfidence" for signal in signals)


def test_rank_signals_tail_mass_threshold_suppresses_below_threshold():
    """tail_mass below tail_mass_threshold must not generate a noisy_context signal."""
    signals = rank_signals(
        distribution=_distribution(tail_mass=0.37),  # just below default threshold 0.38
        embedding=_embedding(),
        faithfulness_score=0.9,
        context_utilization_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    names = [s.name for s in signals]
    assert "noisy_context" not in names


def test_rank_signals_omitting_weights_uses_defaults():
    """Omitting weights parameter is equivalent to passing DEFAULT_WEIGHTS."""
    from signal_weights import DEFAULT_WEIGHTS
    without = rank_signals(
        distribution=_distribution(score_entropy=1.0, tail_mass=0.5),
        embedding=_embedding(query_isolation=5.0),
        faithfulness_score=0.7,
        context_utilization_score=0.7,
        attribution=_attribution(unattributed_fraction=0.3),
        hedging_mismatch=_hedging(overconfident_fraction=0.2),
        query_fit=_query_fit(),
    )
    with_defaults = rank_signals(
        distribution=_distribution(score_entropy=1.0, tail_mass=0.5),
        embedding=_embedding(query_isolation=5.0),
        faithfulness_score=0.7,
        context_utilization_score=0.7,
        attribution=_attribution(unattributed_fraction=0.3),
        hedging_mismatch=_hedging(overconfident_fraction=0.2),
        query_fit=_query_fit(),
        weights=DEFAULT_WEIGHTS,
    )
    assert [(s.name, s.priority_score) for s in without] == [
        (s.name, s.priority_score) for s in with_defaults
    ]


def test_render_recommendation_under_word_limit():
    signals = [RankedSignal(name="low_faithfulness", priority_score=0.6, description="Low faithfulness.", reliability="model_judged")]
    attr = _attribution()
    hed = _hedging()

    fake_text = "Retrieval is decisive but the answer draws on content outside the retrieved chunks — increase chunk size to give the model more grounding material."
    fake_message = MagicMock()
    fake_message.content = [MagicMock(text=fake_text)]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_message

    with patch("services.verdict_generator.anthropic.Anthropic", return_value=fake_client):
        result = render_recommendation(signals, 0.4, 0.8, attr, hed)

    assert len(result.split()) <= 100
