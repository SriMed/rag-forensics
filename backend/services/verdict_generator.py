"""Verdict generation and recommendation rendering (Issue #9).

Two-stage architecture:
  Stage 1 — match_rule(): deterministic decision tree, no LLM.
  Stage 2 — render_recommendation(): Claude renders the matched rule as one sentence.
"""
import logging

import anthropic

from models import (
    ChunkAttributionMetrics,
    EmbeddingSpaceMetrics,
    HedgingMismatchMetrics,
    QueryCorpusFitMetrics,
    RecommendationRule,
    RetrievalDistributionMetrics,
)
from prompts.recommendation_rules import get_rule
from prompts.verdict_prompts import RECOMMENDATION_RENDER_PROMPT

logger = logging.getLogger(__name__)


def match_rule(
    distribution: RetrievalDistributionMetrics,
    embedding: EmbeddingSpaceMetrics,
    faithfulness_score: float,
    attribution: ChunkAttributionMetrics,
    hedging_mismatch: HedgingMismatchMetrics,
    query_fit: QueryCorpusFitMetrics,
) -> RecommendationRule:
    overconfident = hedging_mismatch.overconfident_fraction > 0.2
    underconfident = hedging_mismatch.underconfident_fraction > 0.2

    # R08/R09: query-corpus fit intercepts first — these signal the query or corpus
    # is the problem, not the retrieval/generation pipeline itself.
    if query_fit.triggered and query_fit.mismatch_type == "query_mismatch":
        return get_rule("R08")

    if query_fit.triggered and query_fit.mismatch_type == "coverage_gap":
        return get_rule("R09")

    # R01: ambiguous retrieval — entropy > 1.5 ≈ log2(3), near-uniform over 3 chunks,
    # indicating the retriever couldn't rank meaningfully.
    if distribution.score_entropy > 1.5 and faithfulness_score < 0.6:
        return get_rule("R01")
    if distribution.score_entropy > 1.5 and embedding.chunk_spread > 0.3:
        # chunk_spread > 0.3: chunks from semantically distant regions, reinforcing ambiguity
        return get_rule("R01")

    # R02: retriever was decisive (gap > 0.2) but answer quotes content not in chunks —
    # unattributed_fraction > 0.25 means >1 in 4 answer sentences have no chunk match.
    if distribution.score_gap > 0.2 and attribution.unattributed_fraction > 0.25:
        return get_rule("R02")

    # R03: decisive retrieval but the content is semantically off.
    # faithfulness < 0.5 with low unattributed: the answer uses the chunks but they're wrong.
    if distribution.score_gap > 0.2 and faithfulness_score < 0.5 and attribution.unattributed_fraction < 0.25:
        return get_rule("R03")
    # query_isolation > 1.2: query sits further from chunks than chunks sit from each other —
    # retriever was confident but in the wrong part of the embedding space.
    if distribution.score_gap > 0.2 and embedding.query_isolation > 1.2 and faithfulness_score < 0.6:
        return get_rule("R03")

    # R04: flat score distribution (decay_rate < 0.1 means scores barely drop off) combined
    # with overconfident claims — model is asserting hard facts without selective retrieval.
    if distribution.decay_rate < 0.1 and overconfident:
        return get_rule("R04")

    # R05: tail_mass > 0.4 means 40%+ of score weight is in low-relevance chunks;
    # weak_match_fraction > 0.5 confirms those chunks are muddying the answer.
    if distribution.tail_mass > 0.4 and attribution.weak_match_fraction > 0.5:
        return get_rule("R05")

    # R06: retrieval was decisive and answer is grounded (faithfulness > 0.75) but the
    # model is hedging unnecessarily — a prompt calibration issue.
    if distribution.score_gap > 0.15 and faithfulness_score > 0.75 and underconfident:
        return get_rule("R06")

    # R07: fallback — no abnormal signals detected.
    return get_rule("R07")


def render_recommendation(
    rule: RecommendationRule,
    distribution: RetrievalDistributionMetrics,
    embedding: EmbeddingSpaceMetrics,
    faithfulness_score: float,
    attribution: ChunkAttributionMetrics,
    hedging_mismatch: HedgingMismatchMetrics,
) -> str:
    prompt = RECOMMENDATION_RENDER_PROMPT.format(
        score_entropy=distribution.score_entropy,
        score_gap=distribution.score_gap,
        decay_rate=distribution.decay_rate,
        tail_mass=distribution.tail_mass,
        centroid_distance=embedding.centroid_distance,
        chunk_spread=embedding.chunk_spread,
        query_isolation=embedding.query_isolation,
        faithfulness_score=faithfulness_score,
        unattributed_fraction=attribution.unattributed_fraction,
        weak_match_fraction=attribution.weak_match_fraction,
        overconfident_fraction=hedging_mismatch.overconfident_fraction,
        underconfident_fraction=hedging_mismatch.underconfident_fraction,
        root_cause=rule.root_cause,
        pipeline_component=rule.pipeline_component,
        action=rule.action,
        render_hint=rule.render_hint,
    )
    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        logger.warning("render_recommendation: Claude call failed, falling back to rule.action")
        return rule.action
