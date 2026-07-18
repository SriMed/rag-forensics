"""Verdict generation — ranked signals approach.

Two-stage pipeline:
  Stage 1 — rank_signals(): deterministic scoring of each forensic dimension, no LLM.
  Stage 2 — render_recommendation(): Claude synthesizes the top-ranked signals into
             a 2–3 sentence diagnostic for the developer.
"""
import logging
from dataclasses import dataclass

import anthropic

from config import CLAUDE_SONNET
from models import (
    ChunkAttributionMetrics,
    EmbeddingSpaceMetrics,
    HedgingMismatchMetrics,
    QueryCorpusFitMetrics,
    RetrievalDistributionMetrics,
)
from prompts.verdict_prompts import RANKED_SIGNALS_PROMPT

logger = logging.getLogger(__name__)

_TOP_N = 3


@dataclass
class RankedSignal:
    name: str
    concern_score: float  # 0.0–1.0, higher = more concerning
    description: str


def rank_signals(
    distribution: RetrievalDistributionMetrics,
    embedding: EmbeddingSpaceMetrics,
    faithfulness_score: float,
    retrieval_relevance_score: float,
    attribution: ChunkAttributionMetrics,
    hedging_mismatch: HedgingMismatchMetrics,
    query_fit: QueryCorpusFitMetrics,
) -> list[RankedSignal]:
    """Score each forensic signal by concern level and return sorted descending."""
    signals: list[RankedSignal] = []

    # Unattributed content — fraction is already 0–1
    if attribution.unattributed_fraction > 0:
        signals.append(RankedSignal(
            name="unattributed_content",
            concern_score=attribution.unattributed_fraction,
            description=f"{attribution.unattributed_fraction:.0%} of answer sentences have no clear source in retrieved chunks",
        ))

    # Overconfident claims
    if hedging_mismatch.overconfident_fraction > 0:
        signals.append(RankedSignal(
            name="overconfidence",
            concern_score=hedging_mismatch.overconfident_fraction,
            description=f"{hedging_mismatch.overconfident_fraction:.0%} of claims stated definitively but unsupported by retrieved chunks",
        ))

    # Underconfident claims (weighted lower — less actionable than overconfidence)
    if hedging_mismatch.underconfident_fraction > 0:
        signals.append(RankedSignal(
            name="underconfidence",
            concern_score=hedging_mismatch.underconfident_fraction * 0.7,
            description=f"{hedging_mismatch.underconfident_fraction:.0%} of claims hedged despite strong chunk support",
        ))

    # Low faithfulness — inverted so higher concern = lower score
    faithfulness_concern = max(0.0, 1.0 - faithfulness_score)
    signals.append(RankedSignal(
        name="low_faithfulness",
        concern_score=faithfulness_concern,
        description=f"Faithfulness score {faithfulness_score:.2f} — answer is not fully grounded in retrieved content",
    ))

    # Low retrieval relevance
    relevance_concern = max(0.0, 1.0 - retrieval_relevance_score)
    signals.append(RankedSignal(
        name="low_retrieval_relevance",
        concern_score=relevance_concern,
        description=f"Retrieval relevance score {retrieval_relevance_score:.2f} — retrieved chunks do not match the question well",
    ))

    # Ambiguous retrieval — normalize score_entropy (practical range 0–2.5) to 0–1
    entropy_concern = min(distribution.score_entropy / 2.5, 1.0)
    if entropy_concern > 0.1:
        signals.append(RankedSignal(
            name="ambiguous_retrieval",
            concern_score=entropy_concern,
            description=f"Score entropy {distribution.score_entropy:.2f} — retrieved chunks have similarly uncertain relevance scores",
        ))

    # Weak chunk matches (weighted lower than unattributed — partial support is better than none)
    if attribution.weak_match_fraction > 0:
        signals.append(RankedSignal(
            name="weak_chunk_matches",
            concern_score=attribution.weak_match_fraction * 0.6,
            description=f"{attribution.weak_match_fraction:.0%} of answer sentences have only weak chunk support",
        ))

    # Query geometrically isolated from chunks — concern rises above isolation ratio 1.0
    if embedding.query_isolation > 1.0:
        isolation_concern = min((embedding.query_isolation - 1.0) / 1.5, 1.0)
        signals.append(RankedSignal(
            name="query_isolation",
            concern_score=isolation_concern,
            description=f"Query is geometrically distant from retrieved chunks (isolation ratio {embedding.query_isolation:.2f})",
        ))

    # Query-corpus fit — high-severity signals when triggered
    if query_fit.triggered and query_fit.mismatch_type == "coverage_gap":
        signals.append(RankedSignal(
            name="corpus_coverage_gap",
            concern_score=0.9,
            description="Retrieved chunks answer different questions than the one asked — corpus likely lacks coverage for this query",
        ))
    elif query_fit.triggered and query_fit.mismatch_type == "query_mismatch":
        signals.append(RankedSignal(
            name="query_phrasing_mismatch",
            concern_score=0.75,
            description="Corpus contains relevant content but the query did not retrieve it effectively — consider rephrasing the query",
        ))

    # Noisy context from high tail mass
    if distribution.tail_mass > 0.2:
        tail_concern = min(distribution.tail_mass / 0.6, 1.0) * 0.65
        signals.append(RankedSignal(
            name="noisy_context",
            concern_score=tail_concern,
            description=f"{distribution.tail_mass:.0%} of retrieval score mass in low-ranked chunks — noisy context reaching the generator",
        ))

    return sorted(signals, key=lambda s: s.concern_score, reverse=True)


def render_recommendation(
    signals: list[RankedSignal],
    faithfulness_score: float,
    retrieval_relevance_score: float,
    attribution: ChunkAttributionMetrics,
    hedging_mismatch: HedgingMismatchMetrics,
) -> str:
    top = signals[:_TOP_N]
    if not top:
        return "No significant issues detected — retrieval and generation signals are within normal range."

    signals_text = "\n".join(
        f"{i + 1}. {s.description} (concern score: {s.concern_score:.2f})"
        for i, s in enumerate(top)
    )
    prompt = RANKED_SIGNALS_PROMPT.format(
        signals_text=signals_text,
        faithfulness_score=faithfulness_score,
        retrieval_relevance_score=retrieval_relevance_score,
        unattributed_fraction=attribution.unattributed_fraction,
        overconfident_fraction=hedging_mismatch.overconfident_fraction,
    )
    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=CLAUDE_SONNET,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        logger.warning("render_recommendation: Claude call failed, falling back to top signal")
        return top[0].description
