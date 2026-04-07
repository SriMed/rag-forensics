"""Hedging-verification mismatch detector (Issue #6).

Cross-references linguistic confidence of each claim in the generated answer
against whether that claim is verifiable from retrieved chunks. Produces
continuous overconfident_fraction and underconfident_fraction signals that
feed into the recommendation layer (Issue #9).
"""
import json
import logging
import re
from typing import Literal

import anthropic

from models import ClaimEntry, HedgingMismatchMetrics, RetrievedChunk
from prompts.hedging_prompts import CLAIM_EXTRACTION_PROMPT, ENTAILMENT_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lexicon for confidence classification (deterministic, no LLM)
# ---------------------------------------------------------------------------

# Checked first — explicit acknowledgement of ignorance.
_UNCERTAIN_MARKERS: list[str] = [
    "i'm not sure",
    "im not sure",
    "i am not sure",
    "it's unclear",
    "its unclear",
    "it is unclear",
    "this is unknown",
    "not certain",
    "it's not clear",
    "its not clear",
    "i cannot say",
    "i can't say",
    "i cant say",
]

# Checked second — epistemic softening without full uncertainty.
# Single-word entries are matched with word boundaries; multi-word with substring.
_HEDGED_SINGLE: list[str] = [
    "may", "might", "could",           # modal verbs
    "approximately", "roughly",         # approximators
    "generally", "typically", "usually", "often",  # frequency hedges
    "probably", "possibly", "perhaps",  # probability adverbs
    "reportedly", "allegedly",          # attribution shields
]

_HEDGED_MULTI: list[str] = [
    "around ",      # approximator (trailing space avoids "around the corner")
    "about ",       # approximator (trailing space avoids "about that")
    "according to", # attribution shield
    "i think",      # first-person softener
    "i believe",    # first-person softener
    "i suspect",    # first-person softener
]


def classify_confidence(claim: str) -> Literal["definitive", "hedged", "uncertain"]:
    """Classify the linguistic confidence of a claim using a deterministic lexicon.

    Priority: uncertain > hedged > definitive.
    No LLM calls — pure string matching.
    """
    lower = claim.lower()

    # Check uncertain markers first (multi-word substring match)
    for marker in _UNCERTAIN_MARKERS:
        if marker in lower:
            return "uncertain"

    # Check hedged single-word markers (word boundary)
    for word in _HEDGED_SINGLE:
        if re.search(r"\b" + re.escape(word) + r"\b", lower):
            return "hedged"

    # Check hedged multi-word markers (substring)
    for phrase in _HEDGED_MULTI:
        if phrase in lower:
            return "hedged"

    return "definitive"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(entries: list[ClaimEntry]) -> HedgingMismatchMetrics:
    n = len(entries)
    if n == 0:
        return HedgingMismatchMetrics(
            overconfident_fraction=0.0,
            underconfident_fraction=0.0,
            total_claims=0,
            claim_breakdown=[],
        )
    overconfident = sum(1 for e in entries if e.mismatch_type == "overconfident")
    underconfident = sum(1 for e in entries if e.mismatch_type == "underconfident")
    return HedgingMismatchMetrics(
        overconfident_fraction=overconfident / n,
        underconfident_fraction=underconfident / n,
        total_claims=n,
        claim_breakdown=entries,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_ZEROED = HedgingMismatchMetrics(
    overconfident_fraction=0.0,
    underconfident_fraction=0.0,
    total_claims=0,
    claim_breakdown=[],
)

# Number of top chunks to run entailment against per claim.
_ENTAILMENT_TOP_K = 3


def analyze_hedging_mismatch(
    answer: str,
    chunks: list[RetrievedChunk],
) -> HedgingMismatchMetrics:
    """Extract claims, classify confidence, check entailment, compute mismatch metrics.

    Returns zeroed metrics on any top-level failure (e.g. claim extraction fails).
    Per-claim entailment failures fall back to not_supported for that chunk only.
    """
    client = anthropic.Anthropic()

    # Step 1 — extract claims via LLM
    try:
        extraction_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": CLAIM_EXTRACTION_PROMPT.format(answer=answer),
                }
            ],
        )
        claims_list: list[str] = json.loads(extraction_response.content[0].text)
    except Exception:
        logger.warning("Claim extraction failed; returning zeroed metrics")
        return _ZEROED

    if not claims_list:
        return _ZEROED

    # Chunks are already sorted by retrieval score (descending); take top-k as pre-filter.
    top_chunks = chunks[:_ENTAILMENT_TOP_K]

    # Steps 2 & 3 — classify confidence (lexicon) + check entailment (LLM)
    entries: list[ClaimEntry] = []
    for claim_str in claims_list:
        confidence = classify_confidence(claim_str)

        supported = False
        source_chunk_id: str | None = None

        for chunk in top_chunks:
            try:
                entailment_response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=32,
                    messages=[
                        {
                            "role": "user",
                            "content": ENTAILMENT_PROMPT.format(
                                chunk_text=chunk.text, claim=claim_str
                            ),
                        }
                    ],
                )
                verdict = entailment_response.content[0].text.strip().lower()
                if verdict == "supported":
                    supported = True
                    source_chunk_id = chunk.chunk_id
                    break  # short-circuit on first supporting chunk
            except Exception:
                logger.warning(
                    "Entailment check failed for claim '%s' on chunk '%s'; treating as not_supported",
                    claim_str,
                    chunk.chunk_id,
                )
                # Continue to next chunk — per-claim failure is isolated

        if confidence == "definitive" and not supported:
            mismatch_type: Literal["overconfident", "underconfident", "matched"] = "overconfident"
        elif confidence in ("hedged", "uncertain") and supported:
            mismatch_type = "underconfident"
        else:
            mismatch_type = "matched"

        entries.append(
            ClaimEntry(
                claim=claim_str,
                confidence_class=confidence,
                supported=supported,
                mismatch_type=mismatch_type,
                source_chunk_id=source_chunk_id,
            )
        )

    return _compute_metrics(entries)
