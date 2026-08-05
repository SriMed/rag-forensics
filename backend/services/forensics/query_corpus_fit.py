"""Query-corpus fit analysis — conditional forensics module (Issue #8).

Generates questions the retrieved chunks would answer well. Only runs when forensics
signals indicate a query-corpus mismatch. Makes no LLM calls when untriggered.
"""
import json
import logging

import anthropic
import numpy as np

from config import CLAUDE_HAIKU
from models import QueryCorpusFitMetrics, RetrievedChunk, SuggestedQuestion
from prompts.query_fit_prompts import build_question_generation_prompt
from services.retriever import get_embedding_model

logger = logging.getLogger(__name__)

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
        if not isinstance(parsed, list) or not all(isinstance(q, str) for q in parsed):
            raise ValueError(f"Expected JSON array of strings, got {type(parsed).__name__}")
        question_strings: list[str] = parsed
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

    if not question_strings:
        return QueryCorpusFitMetrics(
            triggered=True,
            trigger_reason=trigger_reason,
            observed_fit=None,
            suggested_questions=[],
            mean_question_similarity=None,
            status="error",
            error="question_generation_returned_no_questions",
        )

    try:
        qry_norm = np.linalg.norm(query_embedding)
        qry_unit = query_embedding / (qry_norm + 1e-10)

        chunk_matrix = np.array(chunk_embeddings)
        chunk_norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
        chunk_units = chunk_matrix / (chunk_norms + 1e-10)

        embed_model = get_embedding_model()
        question_embeddings = embed_model.encode(question_strings)

        suggested: list[SuggestedQuestion] = []
        for i, q_text in enumerate(question_strings):
            q_emb = question_embeddings[i]
            q_norm = np.linalg.norm(q_emb)
            q_unit = q_emb / (q_norm + 1e-10)

            relevance = float(np.dot(q_unit, qry_unit))
            sims = chunk_units @ q_unit
            top_idx = int(np.argmax(sims))

            suggested.append(SuggestedQuestion(
                question=q_text,
                source_chunk_ids=[chunks[top_idx].chunk_id],
                relevance_to_original=relevance,
            ))

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
            mean_question_similarity=None,
            status="error",
            error="fit_computation_failed",
        )

    return QueryCorpusFitMetrics(
        triggered=True,
        trigger_reason=trigger_reason,
        observed_fit=observed_fit,
        suggested_questions=suggested,
        mean_question_similarity=mean_sim,
    )
