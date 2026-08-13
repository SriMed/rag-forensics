"""Query-corpus fit analysis — conditional forensics module (Issue #8).

Generates questions the retrieved chunks would answer well. Only runs when forensics
signals indicate a query-corpus mismatch. Makes no LLM calls when untriggered.
"""
import json
import logging

import anthropic
import numpy as np

from config import CLAUDE_HAIKU
from models import QueryCorpusFitMetrics, RejectedSuggestedQuestion, RetrievedChunk, SuggestedQuestion
from prompts.query_fit_prompts import build_question_generation_prompt, build_question_validation_prompt
from services.retriever import get_embedding_model

logger = logging.getLogger(__name__)

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
) -> str | None:
    """Return the name of the first trigger condition that fired, or None."""
    if query_isolation > 1.2:
        return "query_isolation"
    if context_utilization_score is not None and context_utilization_score < 0.5:
        return "context_utilization"
    if normalized_entropy > 0.9 and faithfulness_score is not None and faithfulness_score < 0.5:
        return "entropy_faithfulness"
    return None


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
    a query-corpus mismatch. On LLM failure returns triggered=True with empty questions.
    """
    trigger_reason = _should_trigger(query_isolation, context_utilization_score, normalized_entropy, faithfulness_score)
    if trigger_reason is None:
        return _UNTRIGGERED

    chunk_texts = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": build_question_generation_prompt(chunk_texts, question),
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("question"), str)
            and item["question"].strip()
            and isinstance(item.get("source_chunk_ids"), list)
            and all(isinstance(cid, str) for cid in item["source_chunk_ids"])
            for item in parsed
        ):
            raise ValueError("Expected question objects with source_chunk_ids")
        candidates = [
            {"question": item["question"].strip(), "source_chunk_ids": item["source_chunk_ids"]}
            for item in parsed
        ]
    except Exception:
        logger.warning("Question generation failed; returning triggered with empty questions")
        return QueryCorpusFitMetrics(
            triggered=True,
            trigger_reason=trigger_reason,
            observed_fit=None,
            suggested_questions=[],
            mean_question_similarity=None,
            status="error",
            error="question_generation_failed",
        )

    if not candidates:
        return QueryCorpusFitMetrics(
            triggered=True,
            trigger_reason=trigger_reason,
            observed_fit=None,
            suggested_questions=[],
            mean_question_similarity=None,
            status="error",
            error="question_generation_returned_no_questions",
        )

    valid_chunk_ids = {chunk.chunk_id for chunk in chunks}
    rejected: list[RejectedSuggestedQuestion] = []
    source_valid_candidates: list[dict] = []
    for candidate in candidates:
        source_ids = list(dict.fromkeys(candidate["source_chunk_ids"]))
        if not source_ids or any(cid not in valid_chunk_ids for cid in source_ids):
            rejected.append(RejectedSuggestedQuestion(
                question=candidate["question"], source_chunk_ids=source_ids, reason="invalid_source_chunk",
            ))
        else:
            source_valid_candidates.append({**candidate, "source_chunk_ids": source_ids})

    try:
        validation_response = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": build_question_validation_prompt(
                    chunk_texts, json.dumps(source_valid_candidates, ensure_ascii=False)
                ),
            }],
        )
        validation_raw = validation_response.content[0].text.strip()
        if validation_raw.startswith("```"):
            validation_raw = validation_raw.split("\n", 1)[1]
            validation_raw = validation_raw.rsplit("```", 1)[0].strip()
        judgments = json.loads(validation_raw)
        if not isinstance(judgments, list) or len(judgments) != len(source_valid_candidates):
            raise ValueError("Validator returned wrong result count")
        answerable: list[dict] = []
        for index, (candidate, judgment) in enumerate(zip(source_valid_candidates, judgments)):
            if not (
                isinstance(judgment, dict)
                and judgment.get("question_index") == index
                and isinstance(judgment.get("directly_answerable"), bool)
                and isinstance(judgment.get("specific"), bool)
                and isinstance(judgment.get("supporting_chunk_ids"), list)
                and all(isinstance(cid, str) for cid in judgment["supporting_chunk_ids"])
            ):
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

        qry_norm = np.linalg.norm(query_embedding)
        qry_unit = query_embedding / (qry_norm + 1e-10)

        question_strings = [candidate["question"] for candidate in answerable]
        if not question_strings:
            return QueryCorpusFitMetrics(
                triggered=True,
                trigger_reason=trigger_reason,
                observed_fit=None,
                suggested_questions=[],
                rejected_questions=rejected,
                mean_question_similarity=None,
                status="error",
                error="insufficient_valid_questions",
            )

        embed_model = get_embedding_model()
        question_embeddings = embed_model.encode(question_strings)

        suggested: list[SuggestedQuestion] = []
        accepted_units: list[np.ndarray] = []
        for i, candidate in enumerate(answerable):
            q_text = candidate["question"]
            q_emb = question_embeddings[i]
            q_norm = np.linalg.norm(q_emb)
            q_unit = q_emb / (q_norm + 1e-10)

            relevance = float(np.dot(q_unit, qry_unit))
            if any(float(np.dot(q_unit, accepted)) >= _DUPLICATE_SIMILARITY_THRESHOLD for accepted in accepted_units):
                rejected.append(RejectedSuggestedQuestion(
                    question=q_text,
                    source_chunk_ids=candidate["source_chunk_ids"],
                    reason="semantic_duplicate",
                ))
                continue

            suggested.append(SuggestedQuestion(
                question=q_text,
                source_chunk_ids=candidate["source_chunk_ids"],
                relevance_to_original=relevance,
            ))
            accepted_units.append(q_unit)

        if len(suggested) < _MIN_VALID_QUESTIONS:
            return QueryCorpusFitMetrics(
                triggered=True,
                trigger_reason=trigger_reason,
                observed_fit=None,
                suggested_questions=suggested,
                rejected_questions=rejected,
                mean_question_similarity=None,
                status="error",
                error="insufficient_valid_questions",
            )

        mean_sim = float(np.mean([sq.relevance_to_original for sq in suggested]))

        if mean_sim > 0.6:
            observed_fit = "retrieved_context_near_miss"
        elif mean_sim < 0.3:
            observed_fit = "retrieved_context_topic_gap"
        else:
            observed_fit = "ambiguous"
    except Exception:
        logger.warning("Retrieved-context fit computation failed", exc_info=True)
        return QueryCorpusFitMetrics(
            triggered=True,
            trigger_reason=trigger_reason,
            observed_fit=None,
            suggested_questions=[],
            rejected_questions=rejected,
            mean_question_similarity=None,
            status="error",
            error="fit_computation_failed",
        )

    return QueryCorpusFitMetrics(
        triggered=True,
        trigger_reason=trigger_reason,
        observed_fit=observed_fit,
        suggested_questions=suggested,
        rejected_questions=rejected,
        mean_question_similarity=mean_sim,
    )
