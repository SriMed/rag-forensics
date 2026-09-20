"""Retrieved-context fit analysis — conditional forensics module (API name: `query_corpus_fit`).

Generates questions the retrieved chunks would answer well. Only runs when forensics
signals suggest the retrieved passages may not match the query. Makes no LLM calls when untriggered.
"""
import json
import logging
from typing import Literal

import numpy as np

from config import CLAUDE_HAIKU
from models import QueryCorpusFitMetrics, RejectedSuggestedQuestion, RetrievedChunk, SuggestedQuestion
from prompts.query_fit_prompts import build_question_generation_prompt, build_question_validation_prompt
from services.failure_detail import failure_detail
from services.llm import LLMError, complete, strip_code_fence
from services.retriever import get_embedding_model

logger = logging.getLogger(__name__)

TriggerReason = Literal["query_isolation", "context_utilization", "entropy_faithfulness"]
ObservedFit = Literal["retrieved_context_near_miss", "retrieved_context_topic_gap", "ambiguous"]

_MIN_VALID_QUESTIONS = 3
_DUPLICATE_SIMILARITY_THRESHOLD = 0.90

_UNTRIGGERED = QueryCorpusFitMetrics(
    triggered=False,
    observed_fit=None,
    suggested_questions=[],
    mean_question_similarity=None,
    status="not_run",
)


def _should_trigger(
    query_isolation: float,
    context_utilization_score: float | None,
    normalized_entropy: float,
    faithfulness_score: float | None,
) -> TriggerReason | None:
    """Return the name of the first trigger condition that fired, or None."""
    if query_isolation > 1.2:
        return "query_isolation"
    if context_utilization_score is not None and context_utilization_score < 0.5:
        return "context_utilization"
    if normalized_entropy > 0.9 and faithfulness_score is not None and faithfulness_score < 0.5:
        return "entropy_faithfulness"
    return None


def _error_metrics(
    trigger_reason: TriggerReason,
    error: str,
    *,
    suggested: list[SuggestedQuestion] | None = None,
    rejected: list[RejectedSuggestedQuestion] | None = None,
) -> QueryCorpusFitMetrics:
    return QueryCorpusFitMetrics(
        triggered=True,
        trigger_reason=trigger_reason,
        observed_fit=None,
        suggested_questions=suggested or [],
        rejected_questions=rejected or [],
        mean_question_similarity=None,
        status="error",
        error=error,
    )


def _is_candidate(item: object) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("question"), str)
        and bool(item["question"].strip())
        and isinstance(item.get("source_chunk_ids"), list)
        and all(isinstance(cid, str) for cid in item["source_chunk_ids"])
    )


def _is_judgment(judgment: object, index: int) -> bool:
    return (
        isinstance(judgment, dict)
        and judgment.get("question_index") == index
        and isinstance(judgment.get("directly_answerable"), bool)
        and isinstance(judgment.get("specific"), bool)
        and isinstance(judgment.get("supporting_chunk_ids"), list)
        and all(isinstance(cid, str) for cid in judgment["supporting_chunk_ids"])
    )


def _generate_candidates(question: str, chunk_texts: str) -> list[dict]:
    """Ask the model for questions the chunks answer. Raises LLMError or ValueError on bad output."""
    raw = complete(build_question_generation_prompt(chunk_texts, question), model=CLAUDE_HAIKU, max_tokens=512)
    parsed = json.loads(strip_code_fence(raw))
    if not isinstance(parsed, list) or not all(_is_candidate(item) for item in parsed):
        raise ValueError("Expected question objects with source_chunk_ids")
    return [
        {"question": item["question"].strip(), "source_chunk_ids": item["source_chunk_ids"]}
        for item in parsed
    ]


def _keep_valid_sources(
    candidates: list[dict],
    chunks: list[RetrievedChunk],
    rejected: list[RejectedSuggestedQuestion],
) -> list[dict]:
    """Drop candidates citing unknown chunk ids (recorded in `rejected`); dedupe each source list."""
    valid_chunk_ids = {chunk.chunk_id for chunk in chunks}
    kept: list[dict] = []
    for candidate in candidates:
        source_ids = list(dict.fromkeys(candidate["source_chunk_ids"]))
        if not source_ids or any(cid not in valid_chunk_ids for cid in source_ids):
            rejected.append(RejectedSuggestedQuestion(
                question=candidate["question"], source_chunk_ids=source_ids, reason="invalid_source_chunk",
            ))
        else:
            kept.append({**candidate, "source_chunk_ids": source_ids})
    return kept


def _judge_candidates(
    chunk_texts: str,
    candidates: list[dict],
    rejected: list[RejectedSuggestedQuestion],
) -> list[dict]:
    """Have the model judge answerability; rejections are appended to `rejected` as they are found.

    Raises LLMError or ValueError if the judgments are malformed.
    """
    raw = complete(
        build_question_validation_prompt(chunk_texts, json.dumps(candidates, ensure_ascii=False)),
        model=CLAUDE_HAIKU,
        max_tokens=512,
    )
    judgments = json.loads(strip_code_fence(raw))
    if not isinstance(judgments, list) or len(judgments) != len(candidates):
        raise ValueError("Validator returned wrong result count")
    answerable: list[dict] = []
    for index, (candidate, judgment) in enumerate(zip(candidates, judgments)):
        if not _is_judgment(judgment, index):
            raise ValueError("Validator returned an invalid judgment")
        supported_ids = list(dict.fromkeys(judgment["supporting_chunk_ids"]))
        allowed_ids = set(candidate["source_chunk_ids"])
        if (
            judgment["directly_answerable"]
            and judgment["specific"]
            and supported_ids
            and set(supported_ids) <= allowed_ids
        ):
            answerable.append({**candidate, "source_chunk_ids": supported_ids})
        else:
            rejected.append(RejectedSuggestedQuestion(
                question=candidate["question"],
                source_chunk_ids=candidate["source_chunk_ids"],
                reason="unsupported" if not judgment["directly_answerable"] else "not_specific",
            ))
    return answerable


def _select_distinct_questions(
    answerable: list[dict],
    query_embedding: np.ndarray,
    rejected: list[RejectedSuggestedQuestion],
) -> list[SuggestedQuestion]:
    """Embed the answerable questions, drop semantic duplicates, and score relevance to the original."""
    qry_unit = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    question_embeddings = get_embedding_model().encode([candidate["question"] for candidate in answerable])

    suggested: list[SuggestedQuestion] = []
    accepted_units: list[np.ndarray] = []
    for candidate, q_emb in zip(answerable, question_embeddings):
        q_unit = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        if any(float(np.dot(q_unit, accepted)) >= _DUPLICATE_SIMILARITY_THRESHOLD for accepted in accepted_units):
            rejected.append(RejectedSuggestedQuestion(
                question=candidate["question"],
                source_chunk_ids=candidate["source_chunk_ids"],
                reason="semantic_duplicate",
            ))
            continue
        suggested.append(SuggestedQuestion(
            question=candidate["question"],
            source_chunk_ids=candidate["source_chunk_ids"],
            relevance_to_original=float(np.dot(q_unit, qry_unit)),
        ))
        accepted_units.append(q_unit)
    return suggested


def _classify_fit(mean_similarity: float) -> ObservedFit:
    if mean_similarity > 0.6:
        return "retrieved_context_near_miss"
    if mean_similarity < 0.3:
        return "retrieved_context_topic_gap"
    return "ambiguous"


def analyze_query_corpus_fit(
    question: str,
    query_embedding: np.ndarray,
    chunks: list[RetrievedChunk],
    chunk_embeddings: list[np.ndarray],
    query_isolation: float,
    context_utilization_score: float | None,
    normalized_entropy: float,
    faithfulness_score: float | None,
) -> QueryCorpusFitMetrics:
    """Generate questions the retrieved chunks answer well; classify observed retrieved-context fit.

    Returns triggered=False immediately (no LLM calls) when signals don't indicate
    a retrieved-context mismatch. On LLM failure returns triggered=True with empty questions.
    """
    trigger_reason = _should_trigger(query_isolation, context_utilization_score, normalized_entropy, faithfulness_score)
    if trigger_reason is None:
        return _UNTRIGGERED

    chunk_texts = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)

    try:
        candidates = _generate_candidates(question, chunk_texts)
    except (LLMError, ValueError):
        logger.warning("Question generation failed; returning triggered with empty questions")
        return _error_metrics(trigger_reason, "question_generation_failed")
    if not candidates:
        return _error_metrics(trigger_reason, "question_generation_returned_no_questions")

    rejected: list[RejectedSuggestedQuestion] = []
    source_valid = _keep_valid_sources(candidates, chunks, rejected)

    # RuntimeError/OSError: the local embedding model failed to load or run.
    try:
        answerable = _judge_candidates(chunk_texts, source_valid, rejected)
        if not answerable:
            return _error_metrics(trigger_reason, "insufficient_valid_questions", rejected=rejected)
        suggested = _select_distinct_questions(answerable, query_embedding, rejected)
        if len(suggested) < _MIN_VALID_QUESTIONS:
            return _error_metrics(
                trigger_reason, "insufficient_valid_questions", suggested=suggested, rejected=rejected
            )
        mean_sim = float(np.mean([sq.relevance_to_original for sq in suggested]))
    except (LLMError, ValueError, RuntimeError, OSError) as exc:
        logger.warning("Retrieved-context fit computation failed: %s", failure_detail(exc))
        return _error_metrics(trigger_reason, "fit_computation_failed", rejected=rejected)

    return QueryCorpusFitMetrics(
        triggered=True,
        trigger_reason=trigger_reason,
        observed_fit=_classify_fit(mean_sim),
        suggested_questions=suggested,
        rejected_questions=rejected,
        mean_question_similarity=mean_sim,
    )
