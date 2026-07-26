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
    mismatch_type=None,
    suggested_questions=[],
    mean_question_similarity=None,
)


def _should_trigger(
    query_isolation: float,
    retrieval_relevance_score: float,
    score_entropy: float,
    faithfulness_score: float,
) -> str | None:
    """Return the name of the first trigger condition that fired, or None."""
    if query_isolation > 1.2:
        return "query_isolation"
    if retrieval_relevance_score < 0.5:
        return "retrieval_relevance"
    if score_entropy > 1.5 and faithfulness_score < 0.5:
        return "entropy_faithfulness"
    return None


def analyze_query_corpus_fit(
    question: str,
    query_embedding: np.ndarray,
    chunks: list[RetrievedChunk],
    chunk_embeddings: list[np.ndarray],
    query_isolation: float,
    retrieval_relevance_score: float,
    score_entropy: float,
    faithfulness_score: float,
) -> QueryCorpusFitMetrics:
    """Generate questions the retrieved chunks answer well; classify mismatch type.

    Returns triggered=False immediately (no LLM calls) when signals don't indicate
    a query-corpus mismatch. On LLM failure returns triggered=True with empty questions.
    """
    trigger_reason = _should_trigger(query_isolation, retrieval_relevance_score, score_entropy, faithfulness_score)
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
            mismatch_type=None,
            suggested_questions=[],
            mean_question_similarity=None,
        )

    if not question_strings:
        return QueryCorpusFitMetrics(
            triggered=True,
            trigger_reason=trigger_reason,
            mismatch_type=None,
            suggested_questions=[],
            mean_question_similarity=None,
        )

    qry_norm = np.linalg.norm(query_embedding)
    qry_unit = query_embedding / (qry_norm + 1e-10)

    chunk_matrix = np.array(chunk_embeddings)
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    chunk_units = chunk_matrix / (chunk_norms + 1e-10)

    embed_model = get_embedding_model()
    # Batch-encode all suggested questions in a single model call.
    question_embeddings = embed_model.encode(question_strings)  # shape (n_questions, dim)

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
        mismatch_type = "query_mismatch"
    elif mean_sim < 0.3:
        mismatch_type = "coverage_gap"
    else:
        mismatch_type = "ambiguous"

    return QueryCorpusFitMetrics(
        triggered=True,
        trigger_reason=trigger_reason,
        mismatch_type=mismatch_type,
        suggested_questions=suggested,
        mean_question_similarity=mean_sim,
    )
