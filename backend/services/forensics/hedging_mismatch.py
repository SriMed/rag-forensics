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
from pydantic import TypeAdapter, ValidationError

from config import CLAUDE_HAIKU
from models import (
    ClaimEntry,
    ClaimExtractionError,
    EntailmentCheck,
    EntailmentVerdict,
    HedgingMismatchMetrics,
    RetrievedChunk,
)
from prompts.hedging_prompts import CLAIM_EXTRACTION_PROMPT, ENTAILMENT_PROMPT

logger = logging.getLogger(__name__)

_CLAIMS_ADAPTER = TypeAdapter(list[str])
_CLAIMS_OUTPUT_CONFIG = {
    "format": {
        "type": "json_schema",
        "schema": _CLAIMS_ADAPTER.json_schema(),
    }
}

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
    "evidence suggests",
    "the evidence suggests",
    "the data indicate",
    "the data indicates",
    "appears to",
    "seems to",
    "is consistent with",
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
    evaluated = [entry for entry in entries if entry.supported is not None]
    n = len(evaluated)
    if not entries:
        return HedgingMismatchMetrics(
            overconfident_fraction=0.0,
            underconfident_fraction=0.0,
            total_claims=0,
            claim_breakdown=[],
        )
    overconfident = sum(1 for e in evaluated if e.mismatch_type == "overconfident")
    underconfident = sum(1 for e in evaluated if e.mismatch_type == "underconfident")
    return HedgingMismatchMetrics(
        overconfident_fraction=overconfident / n if n else 0.0,
        underconfident_fraction=underconfident / n if n else 0.0,
        total_claims=len(entries),
        claim_breakdown=entries,
        evaluated_chunk_count=sum(
            check.status == "evaluated"
            for entry in entries
            for check in entry.entailment_checks
        ),
        evaluated_claim_count=n,
        unavailable_claim_count=len(entries) - n,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_ZEROED = HedgingMismatchMetrics(
    overconfident_fraction=0.0,
    underconfident_fraction=0.0,
    total_claims=0,
    claim_breakdown=[],
    status="ok",
    evaluated_chunk_count=0,
)

# Number of top chunks to run entailment against per claim.
# Only the top-K chunks are checked per claim to bound LLM call count.
# Claims grounded in lower-ranked chunks will be classified as overconfident.
_ENTAILMENT_TOP_K = 3


def _extraction_error(error: ClaimExtractionError) -> HedgingMismatchMetrics:
    return HedgingMismatchMetrics(
        overconfident_fraction=0.0,
        underconfident_fraction=0.0,
        total_claims=0,
        claim_breakdown=[],
        status="error",
        error=error,
        evaluated_chunk_count=0,
    )


def _parse_claims(raw: str) -> list[str]:
    """Decode and strictly validate the claim-extraction payload."""
    decoded = json.loads(raw)
    return _CLAIMS_ADAPTER.validate_python(decoded, strict=True)


def analyze_hedging_mismatch(
    answer: str,
    chunks: list[RetrievedChunk],
) -> HedgingMismatchMetrics:
    """Extract claims, classify confidence, check entailment, compute mismatch metrics.

    Returns an explicit error status on top-level failure (e.g. claim extraction fails).
    Invalid or failed per-chunk judgments remain unavailable rather than becoming negative verdicts.
    """
    client = anthropic.Anthropic()

    # Step 1 — extract claims via LLM
    try:
        extraction_response = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=1024,
            output_config=_CLAIMS_OUTPUT_CONFIG,
            messages=[
                {
                    "role": "user",
                    "content": CLAIM_EXTRACTION_PROMPT.format(answer=answer),
                }
            ],
        )
        raw = extraction_response.content[0].text.strip()
        claims_list = _parse_claims(raw)
    except json.JSONDecodeError:
        logger.warning("Claim extraction returned invalid JSON")
        return _extraction_error("claim_extraction_parse_failed")
    except ValidationError:
        logger.warning("Claim extraction returned JSON that violates the claim schema")
        return _extraction_error("claim_extraction_schema_failed")
    except Exception:
        logger.warning("Claim extraction request failed; returning explicit error status")
        return _extraction_error("claim_extraction_failed")

    if not claims_list:
        return _ZEROED

    # Chunks are already sorted by retrieval score (descending); take top-k as pre-filter.
    top_chunks = chunks[:_ENTAILMENT_TOP_K]

    # Steps 2 & 3 — classify confidence (lexicon) + check entailment (LLM)
    entries: list[ClaimEntry] = []
    for claim_str in claims_list:
        confidence = classify_confidence(claim_str)

        supported: bool | None = None
        source_chunk_id: str | None = None
        checks: list[EntailmentCheck] = []

        for chunk in top_chunks:
            try:
                entailment_response = client.messages.create(
                    model=CLAUDE_HAIKU,
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
                raw_verdict = entailment_response.content[0].text
                normalized = raw_verdict.strip()
                try:
                    verdict = EntailmentVerdict(normalized)
                except ValueError:
                    checks.append(EntailmentCheck(
                        chunk_id=chunk.chunk_id,
                        status="invalid_format",
                        raw_output=raw_verdict,
                    ))
                    logger.warning(
                        "Invalid entailment response for claim '%s' on chunk '%s': %r",
                        claim_str,
                        chunk.chunk_id,
                        raw_verdict,
                    )
                    continue
                checks.append(EntailmentCheck(
                    chunk_id=chunk.chunk_id,
                    status="evaluated",
                    verdict=verdict,
                    raw_output=raw_verdict,
                ))
                if verdict == EntailmentVerdict.SUPPORTED:
                    supported = True
                    source_chunk_id = chunk.chunk_id
                    break  # short-circuit on first supporting chunk
                supported = False
            except Exception:
                checks.append(EntailmentCheck(chunk_id=chunk.chunk_id, status="error"))
                logger.warning(
                    "Entailment check failed for claim '%s' on chunk '%s'",
                    claim_str,
                    chunk.chunk_id,
                )
                # Continue to next chunk — per-claim failure is isolated

        if supported is None:
            mismatch_type = None
        elif confidence == "definitive" and not supported:
            mismatch_type: Literal["overconfident", "underconfident", "matched"] = "overconfident"
        else:
            # Binary entailment cannot determine whether hedging is unnecessarily weak.
            # That requires comparing the source's epistemic strength with the claim's.
            mismatch_type = "matched"

        entries.append(
            ClaimEntry(
                claim=claim_str,
                confidence_class=confidence,
                supported=supported,
                mismatch_type=mismatch_type,
                source_chunk_id=source_chunk_id,
                entailment_checks=checks,
            )
        )

    return _compute_metrics(entries)
