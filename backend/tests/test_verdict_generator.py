"""Tests for verdict_generator.py (Issue #9).

All 15 tests per the issue spec. Tests are written before implementation (TDD).
match_rule() is pure Python — no mocks needed.
render_recommendation() mocks services.verdict_generator.anthropic.Anthropic.
"""
import pytest
from unittest.mock import MagicMock, patch

from models import (
    ChunkAttributionMetrics,
    EmbeddingSpaceMetrics,
    HedgingMismatchMetrics,
    QueryCorpusFitMetrics,
    RetrievalDistributionMetrics,
)
from services.verdict_generator import match_rule, render_recommendation
from prompts.recommendation_rules import get_rule


# ---------------------------------------------------------------------------
# Helpers — build minimal metric objects with sane defaults
# ---------------------------------------------------------------------------

def _distribution(**kwargs) -> RetrievalDistributionMetrics:
    defaults = dict(score_gap=0.05, score_entropy=0.5, decay_rate=0.3,
                    tail_mass=0.1, top_score=0.9, n_chunks=5)
    defaults.update(kwargs)
    return RetrievalDistributionMetrics(**defaults)


def _embedding(**kwargs) -> EmbeddingSpaceMetrics:
    defaults = dict(centroid_distance=0.2, chunk_spread=0.1,
                    query_isolation=0.8, projection=[])
    defaults.update(kwargs)
    return EmbeddingSpaceMetrics(**defaults)


def _attribution(**kwargs) -> ChunkAttributionMetrics:
    defaults = dict(unattributed_fraction=0.05, mean_attribution_score=0.8,
                    weak_match_fraction=0.1, attribution_map=[])
    defaults.update(kwargs)
    return ChunkAttributionMetrics(**defaults)


def _hedging(**kwargs) -> HedgingMismatchMetrics:
    defaults = dict(overconfident_fraction=0.0, underconfident_fraction=0.0,
                    total_claims=5, claim_breakdown=[])
    defaults.update(kwargs)
    return HedgingMismatchMetrics(**defaults)


def _query_fit(**kwargs) -> QueryCorpusFitMetrics:
    defaults = dict(triggered=False, mismatch_type=None,
                    suggested_questions=[], mean_question_similarity=None)
    defaults.update(kwargs)
    return QueryCorpusFitMetrics(**defaults)


# ---------------------------------------------------------------------------
# Rule-matching tests (pure, no mocks)
# ---------------------------------------------------------------------------

def test_r01_high_entropy_low_faithfulness():
    """High entropy + low faithfulness → R01."""
    rule = match_rule(
        distribution=_distribution(score_entropy=1.6),
        embedding=_embedding(),
        faithfulness_score=0.4,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R01"


def test_r01_high_entropy_high_chunk_spread():
    """High entropy + high chunk_spread → R01."""
    rule = match_rule(
        distribution=_distribution(score_entropy=1.6),
        embedding=_embedding(chunk_spread=0.4),
        faithfulness_score=0.7,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R01"


def test_r02_high_gap_high_unattributed():
    """Decisive retrieval + high unattributed fraction → R02."""
    rule = match_rule(
        distribution=_distribution(score_gap=0.3),
        embedding=_embedding(),
        faithfulness_score=0.7,
        attribution=_attribution(unattributed_fraction=0.3),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R02"


def test_r03a_decisive_retrieval_wrong_content():
    """High score_gap + low faithfulness + low unattributed_fraction → R03."""
    rule = match_rule(
        distribution=_distribution(score_gap=0.3),
        embedding=_embedding(),
        faithfulness_score=0.4,
        attribution=_attribution(unattributed_fraction=0.1),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R03"


def test_r03b_high_gap_high_query_isolation():
    """High score_gap + high query_isolation + low faithfulness → R03."""
    rule = match_rule(
        distribution=_distribution(score_gap=0.3),
        embedding=_embedding(query_isolation=1.3),
        faithfulness_score=0.5,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R03"


def test_r04_low_decay_overconfident():
    """Low decay_rate + overconfident generation → R04."""
    rule = match_rule(
        distribution=_distribution(decay_rate=0.05),
        embedding=_embedding(),
        faithfulness_score=0.7,
        attribution=_attribution(),
        hedging_mismatch=_hedging(overconfident_fraction=0.3),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R04"


def test_r05_high_tail_mass_high_weak_match():
    """High tail_mass + high weak_match_fraction → R05."""
    rule = match_rule(
        distribution=_distribution(tail_mass=0.5),
        embedding=_embedding(),
        faithfulness_score=0.7,
        attribution=_attribution(weak_match_fraction=0.6),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R05"


def test_r06_strong_retrieval_underconfident():
    """High score_gap + high faithfulness + underconfident generation → R06."""
    rule = match_rule(
        distribution=_distribution(score_gap=0.25),
        embedding=_embedding(),
        faithfulness_score=0.85,
        attribution=_attribution(),
        hedging_mismatch=_hedging(underconfident_fraction=0.3),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R06"


def test_r07_healthy_pipeline():
    """All signals clean → R07 (healthy pipeline fallback)."""
    rule = match_rule(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.9,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(),
    )
    assert rule.rule_id == "R07"


def test_r08_query_mismatch():
    """query_fit triggered + mismatch_type='query_mismatch' → R08."""
    rule = match_rule(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.5,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(triggered=True, mismatch_type="query_mismatch"),
    )
    assert rule.rule_id == "R08"


def test_r09_coverage_gap():
    """query_fit triggered + mismatch_type='coverage_gap' → R09."""
    rule = match_rule(
        distribution=_distribution(),
        embedding=_embedding(),
        faithfulness_score=0.5,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(triggered=True, mismatch_type="coverage_gap"),
    )
    assert rule.rule_id == "R09"


def test_r08_takes_priority_over_r01():
    """R08/R09 checked before R01: triggered query_fit + high entropy → R08, not R01."""
    rule = match_rule(
        distribution=_distribution(score_entropy=1.6),
        embedding=_embedding(),
        faithfulness_score=0.4,
        attribution=_attribution(),
        hedging_mismatch=_hedging(),
        query_fit=_query_fit(triggered=True, mismatch_type="query_mismatch"),
    )
    assert rule.rule_id == "R08"


# ---------------------------------------------------------------------------
# render_recommendation tests
# ---------------------------------------------------------------------------

def test_render_recommendation_under_50_words():
    """render_recommendation() returns a string under 50 words."""
    rule = get_rule("R07")
    dist = _distribution()
    emb = _embedding()
    attr = _attribution()
    hed = _hedging()

    fake_message = MagicMock()
    fake_message.content = [MagicMock(text="Your pipeline is healthy — decisive retrieval and faithful generation with calibrated confidence.")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_message

    with patch("services.verdict_generator.anthropic.Anthropic", return_value=fake_client):
        result = render_recommendation(rule, dist, emb, 0.9, attr, hed)

    assert isinstance(result, str)
    assert len(result.split()) <= 50


def test_render_recommendation_claude_failure_falls_back_to_action():
    """Claude API failure → returns rule.action string, no exception raised."""
    rule = get_rule("R01")
    dist = _distribution()
    emb = _embedding()
    attr = _attribution()
    hed = _hedging()

    with patch("services.verdict_generator.anthropic.Anthropic", side_effect=Exception("API down")):
        result = render_recommendation(rule, dist, emb, 0.4, attr, hed)

    assert result == rule.action


def test_match_rule_never_returns_none():
    """No valid input combination causes match_rule() to return None."""
    import random
    random.seed(42)

    for _ in range(50):
        dist = _distribution(
            score_gap=random.uniform(0, 1),
            score_entropy=random.uniform(0, 3),
            decay_rate=random.uniform(0, 1),
            tail_mass=random.uniform(0, 1),
        )
        emb = _embedding(
            chunk_spread=random.uniform(0, 1),
            query_isolation=random.uniform(0, 2),
        )
        attr = _attribution(
            unattributed_fraction=random.uniform(0, 1),
            weak_match_fraction=random.uniform(0, 1),
        )
        hed = _hedging(
            overconfident_fraction=random.uniform(0, 1),
            underconfident_fraction=random.uniform(0, 1),
        )
        rule = match_rule(
            distribution=dist,
            embedding=emb,
            faithfulness_score=random.uniform(0, 1),
            attribution=attr,
            hedging_mismatch=hed,
            query_fit=_query_fit(),
        )
        assert rule is not None
        assert rule.rule_id in {"R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09"}
