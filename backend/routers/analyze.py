import logging
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, HTTPException

from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CustomAnalyzeRequest,
    RAGASMetrics,
    RetrievedChunk,
    VerdictReasoning,
    VerdictSignal,
)
from services.forensics.chunk_attribution import analyze_chunk_attribution
from services.forensics.embedding_analysis import analyze_embedding_space
from services.forensics.hedging_mismatch import analyze_hedging_mismatch
from services.forensics.query_corpus_fit import analyze_query_corpus_fit
from services.forensics.retrieval_distribution import analyze_retrieval_distribution
from services.generator import generate_answer
from services.ragas_scorer import score_answer_faithfulness, score_context_utilization
from services.retriever import get_embedding_model, retrieve_for_example
from services.verdict_generator import build_verdict_reasoning, rank_signals, render_recommendation

logger = logging.getLogger(__name__)
router = APIRouter()

_ANALYSIS_FAILED_DETAIL = "Analysis failed. See the server logs for details."


def build_analysis(
    question: str,
    answer: str,
    chunks: Sequence[RetrievedChunk],
    query_embedding: np.ndarray,
    chunk_embeddings: list[np.ndarray],
) -> AnalyzeResponse:
    """Run every forensics module on one (question, answer, chunks) record and assemble the response."""
    chunks = list(chunks)
    utilization, utilization_context_excerpts = score_context_utilization(question, answer, chunks)
    faithfulness, faithfulness_context_excerpts = score_answer_faithfulness(answer, chunks, question)

    embedding_space = analyze_embedding_space(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunk_ids=[c.chunk_id for c in chunks],
    )
    retrieval_distribution = analyze_retrieval_distribution(chunks)
    hedging = analyze_hedging_mismatch(answer, chunks)
    chunk_attribution = analyze_chunk_attribution(answer, chunks, [e.tolist() for e in chunk_embeddings])
    query_corpus_fit = analyze_query_corpus_fit(
        question=question,
        query_embedding=query_embedding,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        query_isolation=embedding_space.query_isolation,
        context_utilization_score=utilization.score,
        normalized_entropy=retrieval_distribution.normalized_entropy,
        faithfulness_score=faithfulness.score,
    )
    signals = rank_signals(
        distribution=retrieval_distribution,
        embedding=embedding_space,
        faithfulness_score=faithfulness.score,
        context_utilization_score=utilization.score,
        attribution=chunk_attribution,
        hedging_mismatch=hedging,
        query_fit=query_corpus_fit,
    )
    reasoning = build_verdict_reasoning(signals)

    return AnalyzeResponse(
        question=question,
        generated_answer=answer,
        retrieved_chunks=[c.text for c in chunks],
        retrieved_chunk_details=chunks,
        ragas=RAGASMetrics(
            context_utilization=utilization,
            faithfulness=faithfulness,
            utilization_context_excerpts=utilization_context_excerpts,
            faithfulness_context_excerpts=faithfulness_context_excerpts,
        ),
        hedging_mismatch=hedging,
        chunk_attribution=chunk_attribution,
        retrieval_distribution=retrieval_distribution,
        embedding_space=embedding_space,
        query_corpus_fit=query_corpus_fit,
        verdict_signals=[
            VerdictSignal(
                name=s.name,
                priority_score=s.priority_score,
                description=s.description,
                reliability=s.reliability,
            )
            for s in signals
        ],
        verdict_reasoning=VerdictReasoning.model_validate(asdict(reasoning)),
        recommendation=render_recommendation(reasoning),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    logger.info("analyze request: example_id=%s", request.example_id)
    try:
        question, retrieval_result = retrieve_for_example(request.example_id)
        chunks = retrieval_result.chunks
        logger.debug("retrieved %d chunks for example_id=%s", len(chunks), request.example_id)
        answer = generate_answer(question, chunks)
        logger.debug("generated answer (%d chars)", len(answer))
        response = build_analysis(
            question=question,
            answer=answer,
            chunks=chunks,
            query_embedding=np.array(retrieval_result.query_embedding),
            chunk_embeddings=[np.array(e) for e in retrieval_result.chunk_embeddings],
        )
    except Exception as exc:
        logger.exception("analyze failed for example_id=%s", request.example_id)
        raise HTTPException(status_code=500, detail=_ANALYSIS_FAILED_DETAIL) from exc

    logger.info("analyze complete: example_id=%s", request.example_id)
    return response


@router.post("/analyze/custom", response_model=AnalyzeResponse)
def analyze_custom(request: CustomAnalyzeRequest) -> AnalyzeResponse:
    logger.info("analyze/custom request: %d chunks", len(request.chunks))
    try:
        model = get_embedding_model()
        query_embedding = np.array(model.encode([request.question])[0])
        # Batch-encode all chunk texts in a single model call.
        chunk_embeddings = [np.array(e) for e in model.encode([c.text for c in request.chunks])]
        response = build_analysis(
            question=request.question,
            answer=request.answer,
            chunks=request.chunks,
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
        )
    except Exception as exc:
        logger.exception("analyze/custom failed")
        raise HTTPException(status_code=500, detail=_ANALYSIS_FAILED_DETAIL) from exc

    logger.info("analyze/custom complete")
    return response
