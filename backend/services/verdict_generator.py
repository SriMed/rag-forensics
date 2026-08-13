"""Verdict generation — deterministic reasoning with bounded rendering.

Two-stage pipeline:
  Stage 1 — rank_signals(): deterministic scoring of each forensic dimension, no LLM.
  Stage 2 — build_verdict_reasoning(): deterministic observations, hypotheses, and test.
  Stage 3 — render_recommendation(): Claude may only word that inspectable structure.
"""
import json
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
from signal_weights import DEFAULT_WEIGHTS, SignalWeights

logger = logging.getLogger(__name__)

_TOP_N = 3


@dataclass
class RankedSignal:
    name: str
    priority_score: float  # heuristic 0.0–1.0 index; not a probability or calibrated severity
    description: str
    reliability: str


@dataclass(frozen=True)
class VerdictObservation:
    signal_name: str
    description: str
    reliability: str


@dataclass(frozen=True)
class VerdictHypothesis:
    hypothesis_id: str
    statement: str


@dataclass(frozen=True)
class OutcomeInterpretation:
    outcome: str
    supports_hypothesis_ids: list[str]


@dataclass(frozen=True)
class DiscriminatingTest:
    component: str
    action: str
    interpretations: list[OutcomeInterpretation]


@dataclass(frozen=True)
class VerdictReasoning:
    observations: list[VerdictObservation]
    hypotheses: list[VerdictHypothesis]
    test: DiscriminatingTest | None


def rank_signals(
    distribution: RetrievalDistributionMetrics,
    embedding: EmbeddingSpaceMetrics,
    faithfulness_score: float | None,
    context_utilization_score: float | None,
    attribution: ChunkAttributionMetrics,
    hedging_mismatch: HedgingMismatchMetrics,
    query_fit: QueryCorpusFitMetrics,
    weights: SignalWeights | None = None,
) -> list[RankedSignal]:
    """Compute heuristic diagnostic priorities and return them sorted descending."""
    w = weights if weights is not None else DEFAULT_WEIGHTS
    signals: list[RankedSignal] = []

    if faithfulness_score is None:
        signals.append(RankedSignal(
            name="faithfulness_unavailable",
            priority_score=0.5,
            description="Faithfulness evaluation is unavailable; do not interpret the missing score as healthy",
            reliability="unvalidated",
        ))

    if context_utilization_score is None:
        signals.append(RankedSignal(
            name="context_utilization_unavailable",
            priority_score=0.5,
            description="Context-utilization evaluation is unavailable; do not interpret the missing score as healthy",
            reliability="unvalidated",
        ))

    if hedging_mismatch.status == "error":
        signals.append(RankedSignal(
            name="hedging_analysis_unavailable",
            priority_score=0.5,
            description=f"Hedging analysis is unavailable ({hedging_mismatch.error}); do not interpret zero fractions as healthy",
            reliability="unvalidated",
        ))
    elif hedging_mismatch.unavailable_claim_count > 0:
        signals.append(RankedSignal(
            name="hedging_judgments_unavailable",
            priority_score=0.5,
            description=(
                f"Entailment judgments are unavailable for "
                f"{hedging_mismatch.unavailable_claim_count} of "
                f"{hedging_mismatch.total_claims} claims; mismatch fractions exclude them"
            ),
            reliability="unvalidated",
        ))

    if query_fit.status == "error":
        signals.append(RankedSignal(
            name="retrieved_context_fit_unavailable",
            priority_score=0.5,
            description=f"Retrieved-context fit analysis is unavailable ({query_fit.error})",
            reliability="unvalidated",
        ))

    # Unattributed content — fraction is already 0–1
    if attribution.unattributed_fraction > 0:
        signals.append(RankedSignal(
            name="unattributed_content",
            priority_score=attribution.unattributed_fraction,
            description=f"{attribution.unattributed_fraction:.0%} of answer sentences have no semantically close source candidate in retrieved chunks",
            reliability="unvalidated",
        ))

    # Overconfident claims
    if hedging_mismatch.overconfident_fraction > 0:
        signals.append(RankedSignal(
            name="overconfidence",
            priority_score=hedging_mismatch.overconfident_fraction,
            description=f"{hedging_mismatch.overconfident_fraction:.0%} of claims stated definitively but unsupported by retrieved chunks",
            reliability="model_judged",
        ))

    # Low faithfulness — inverted so higher concern = lower score
    if faithfulness_score is not None:
        faithfulness_concern = max(0.0, 1.0 - faithfulness_score)
        signals.append(RankedSignal(
            name="low_faithfulness",
            priority_score=faithfulness_concern,
            description=f"Faithfulness score {faithfulness_score:.2f} — answer is not fully grounded in retrieved content",
            reliability="model_judged",
        ))

    # Low answer-conditioned context utilization
    if context_utilization_score is not None:
        utilization_concern = max(0.0, 1.0 - context_utilization_score)
        signals.append(RankedSignal(
            name="low_context_utilization",
            priority_score=utilization_concern,
            description=f"Context utilization score {context_utilization_score:.2f} — retrieved chunks were not consistently useful for producing the answer",
            reliability="model_judged",
        ))

    # Ambiguous retrieval — normalize score_entropy by empirical p95 to 0–1
    entropy_concern = min(max(distribution.normalized_entropy, 0.0), 1.0)
    if entropy_concern > w.entropy_min_concern:
        signals.append(RankedSignal(
            name="ambiguous_retrieval",
            priority_score=entropy_concern,
            description=f"Normalized score entropy {distribution.normalized_entropy:.2f} — retrieval scores are similarly distributed; interpret with absolute relevance",
            reliability="partially_calibrated",
        ))

    # Weak chunk matches (weighted lower than unattributed — partial support is better than none)
    if attribution.weak_match_fraction > 0:
        signals.append(RankedSignal(
            name="weak_chunk_matches",
            priority_score=attribution.weak_match_fraction * w.weak_match_weight,
            description=f"{attribution.weak_match_fraction:.0%} of answer sentences have only weak chunk support",
            reliability="unvalidated",
        ))

    # Query geometrically isolated from chunks — concern rises above isolation threshold
    if embedding.query_isolation > w.isolation_threshold:
        isolation_concern = min(
            (embedding.query_isolation - w.isolation_threshold) / w.isolation_excess_range, 1.0
        )
        signals.append(RankedSignal(
            name="query_isolation",
            priority_score=isolation_concern,
            description=f"Query is geometrically distant from retrieved chunks (isolation ratio {embedding.query_isolation:.2f})",
            reliability="partially_calibrated",
        ))

    # Query-corpus fit — high-severity signals when triggered
    if query_fit.triggered and query_fit.observed_fit == "retrieved_context_topic_gap":
        signals.append(RankedSignal(
            name="retrieved_context_topic_gap",
            priority_score=0.9,
            description="Retrieved chunks answer semantically distant questions; this does not establish whether the full corpus lacks coverage",
            reliability="unvalidated",
        ))
    elif query_fit.triggered and query_fit.observed_fit == "retrieved_context_near_miss":
        signals.append(RankedSignal(
            name="retrieved_context_near_miss",
            priority_score=0.75,
            description="Retrieved chunks answer questions near the one asked; test a query rewrite before attributing the failure to phrasing",
            reliability="unvalidated",
        ))

    # Noisy context from high tail mass (only above corpus mean to reduce noise)
    if distribution.tail_mass > w.tail_mass_threshold:
        tail_concern = min(distribution.tail_mass / w.tail_mass_p95, 1.0) * w.tail_mass_weight
        signals.append(RankedSignal(
            name="noisy_context",
            priority_score=tail_concern,
            description=f"{distribution.tail_mass:.0%} of retrieval score mass in low-ranked chunks — noisy context reaching the generator",
            reliability="partially_calibrated",
        ))

    return sorted(signals, key=lambda s: s.priority_score, reverse=True)


_RETRIEVAL_SIGNALS = {
    "ambiguous_retrieval", "query_isolation", "retrieved_context_topic_gap",
    "retrieved_context_near_miss", "noisy_context", "low_context_utilization",
}
_GENERATION_SIGNALS = {
    "unattributed_content", "weak_chunk_matches", "overconfidence", "low_faithfulness",
}
_UNAVAILABLE_SIGNALS = {
    "faithfulness_unavailable", "context_utilization_unavailable",
    "hedging_analysis_unavailable", "hedging_judgments_unavailable",
    "retrieved_context_fit_unavailable",
}


def build_verdict_reasoning(signals: list[RankedSignal]) -> VerdictReasoning:
    """Build inspectable hypotheses and a test without asking a model to infer causes."""
    top = signals[:_TOP_N]
    observations = [
        VerdictObservation(s.name, s.description, s.reliability) for s in top
    ]
    if not observations:
        return VerdictReasoning(observations=[], hypotheses=[], test=None)

    names = {s.name for s in top}
    has_retrieval = bool(names & _RETRIEVAL_SIGNALS)
    has_generation = bool(names & _GENERATION_SIGNALS)
    has_unavailable = bool(names & _UNAVAILABLE_SIGNALS)

    if has_unavailable:
        hypotheses = [
            VerdictHypothesis(
                "H1",
                "The missing evaluation may be hiding a diagnostic concern that the available signals cannot resolve.",
            ),
            VerdictHypothesis(
                "H2",
                "The available pipeline behavior may be acceptable, while the apparent concern comes from evaluation unavailability.",
            ),
        ]
        test = DiscriminatingTest(
            component="unavailable evaluation component",
            action=(
                "Restore the unavailable evaluation, rerun it on the same answer and chunks, and compare "
                "the completed judgment with the available observations."
            ),
            interpretations=[
                OutcomeInterpretation("The restored evaluation reports a concern consistent with the available observations.", ["H1"]),
                OutcomeInterpretation("The restored evaluation reports no concern and the available observations remain unchanged.", ["H2"]),
            ],
        )
    elif has_retrieval and has_generation:
        hypotheses = [
            VerdictHypothesis("H1", "Retrieved evidence may be a poor fit for the question."),
            VerdictHypothesis("H2", "Answer generation may be introducing or overstating content despite the retrieved evidence."),
        ]
        test = DiscriminatingTest(
            component="retrieval-to-generation boundary",
            action=(
                "Hold the question and generator configuration constant, replace the retrieved chunks with "
                "manually verified relevant chunks, and rerun the answer and forensic checks."
            ),
            interpretations=[
                OutcomeInterpretation("Grounding observations improve with verified chunks.", ["H1"]),
                OutcomeInterpretation("Grounding observations remain concerning with verified chunks.", ["H2"]),
            ],
        )
    elif has_retrieval:
        hypotheses = [
            VerdictHypothesis("H1", "Retrieval may be returning evidence that is a poor fit for the question."),
            VerdictHypothesis("H2", "The retrieval signal may reflect its heuristic or partially calibrated measurement rather than poor evidence."),
        ]
        test = DiscriminatingTest(
            component="retriever",
            action="Compare the retrieved chunks with manually verified relevant chunks for the same question.",
            interpretations=[
                OutcomeInterpretation("Verified chunks are materially more relevant than the retrieved chunks.", ["H1"]),
                OutcomeInterpretation("Retrieved and verified chunks are similarly relevant.", ["H2"]),
            ],
        )
    else:
        hypotheses = [
            VerdictHypothesis("H1", "Answer generation may be introducing content not supported by the retrieved chunks."),
            VerdictHypothesis("H2", "The grounding concern may reflect the unvalidated or model-judged measurement rather than generation behavior."),
        ]
        test = DiscriminatingTest(
            component="answer generator",
            action="Review the flagged answer units against the retrieved chunks, then rerun with a strict evidence-only generation prompt.",
            interpretations=[
                OutcomeInterpretation("The strict run removes the flagged unsupported content.", ["H1"]),
                OutcomeInterpretation("Human review finds support or the strict run leaves the measurements unchanged.", ["H2"]),
            ],
        )

    return VerdictReasoning(observations=observations, hypotheses=hypotheses, test=test)


def reasoning_payload(reasoning: VerdictReasoning) -> dict:
    return {
        "observations": [o.__dict__ for o in reasoning.observations],
        "hypotheses": [h.__dict__ for h in reasoning.hypotheses],
        "test": None if reasoning.test is None else {
            "component": reasoning.test.component,
            "action": reasoning.test.action,
            "interpretations": [i.__dict__ for i in reasoning.test.interpretations],
        },
    }


def format_verdict_reasoning(reasoning: VerdictReasoning) -> str:
    """Deterministic, complete fallback for rendering failures."""
    if not reasoning.observations:
        return "No significant issues detected — retrieval and generation signals are within normal range."
    observations = "; ".join(
        f"{o.description} [reliability: {o.reliability}]" for o in reasoning.observations
    )
    hypotheses = "; ".join(f"{h.hypothesis_id}: {h.statement}" for h in reasoning.hypotheses)
    test = reasoning.test
    assert test is not None
    outcomes = "; ".join(
        f"{i.outcome} -> {', '.join(i.supports_hypothesis_ids)}" for i in test.interpretations
    )
    return f"Observations: {observations} Hypotheses: {hypotheses} Test {test.component}: {test.action} Outcomes: {outcomes}"


def render_recommendation(
    reasoning: VerdictReasoning,
) -> str:
    if not reasoning.observations:
        return format_verdict_reasoning(reasoning)
    prompt = RANKED_SIGNALS_PROMPT.format(
        reasoning_json=json.dumps(reasoning_payload(reasoning), indent=2),
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
        logger.warning("render_recommendation: Claude call failed, falling back to structured reasoning")
        return format_verdict_reasoning(reasoning)
