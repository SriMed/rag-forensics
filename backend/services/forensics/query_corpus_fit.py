"""Query-corpus fit analysis — conditional forensics module (Issue #8).

Generates questions the retrieved chunks would answer well. Only runs when forensics
signals indicate a query-corpus mismatch. Makes no LLM calls when untriggered.
"""
import json
import logging

import anthropic
import numpy as np

from models import QueryCorpusFitMetrics, RetrievedChunk, SuggestedQuestion
from prompts.query_fit_prompts import QUESTION_GENERATION_PROMPT
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
) -> bool:
    if query_isolation > 1.2:
        return True
    if retrieval_relevance_score < 0.5:
        return True
    if score_entropy > 1.5 and faithfulness_score < 0.5:
        return True
    return False


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
    if not _should_trigger(query_isolation, retrieval_relevance_score, score_entropy, faithfulness_score):
        return _UNTRIGGERED

    # Build chunk text block for the prompt
    chunk_texts = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": QUESTION_GENERATION_PROMPT.format(
                    chunk_texts=chunk_texts,
                    original_question=question,
                ),
            }],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        question_strings: list[str] = json.loads(raw)
    except Exception:
        logger.warning("Question generation failed; returning triggered with empty questions")
        return QueryCorpusFitMetrics(
            triggered=True,
            mismatch_type=None,
            suggested_questions=[],
            mean_question_similarity=None,
        )

    if not question_strings:
        return QueryCorpusFitMetrics(
            triggered=True,
            mismatch_type=None,
            suggested_questions=[],
            mean_question_similarity=None,
        )

    # Pre-normalize query and chunk embeddings for cosine similarity via dot product
    qry_norm = np.linalg.norm(query_embedding)
    qry_unit = query_embedding / (qry_norm + 1e-10)

    chunk_matrix = np.array(chunk_embeddings)  # shape (n_chunks, dim)
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    chunk_units = chunk_matrix / (chunk_norms + 1e-10)  # shape (n_chunks, dim)

    embed_model = get_embedding_model()
    suggested: list[SuggestedQuestion] = []

    for q_text in question_strings:
        q_emb = embed_model.encode([q_text])[0]  # shape (dim,)
        q_norm = np.linalg.norm(q_emb)
        q_unit = q_emb / (q_norm + 1e-10)

        # relevance_to_original: cosine sim between this question and the original query
        relevance = float(np.dot(q_unit, qry_unit))

        # source_chunk_ids: top-1 chunk by cosine sim to this question embedding
        sims = chunk_units @ q_unit  # shape (n_chunks,)
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
        mismatch_type=mismatch_type,
        suggested_questions=suggested,
        mean_question_similarity=mean_sim,
    )
