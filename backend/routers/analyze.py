import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from models import AnalyzeRequest, AnalyzeResponse, DimensionResult, RAGASMetrics
from services.retriever import retrieve_for_example
from services.generator import generate_answer
from services.ragas_scorer import score_retrieval_relevance, score_answer_faithfulness
from services.forensics.retrieval_distribution import analyze_retrieval_distribution
from services.forensics.embedding_analysis import analyze_embedding_space

logger = logging.getLogger(__name__)
router = APIRouter()

_STUB_DIMENSION = DimensionResult(
    verdict="pass",
    explanation="Not yet implemented.",
    evidence=[],
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

        relevance_score, relevance_evidence = score_retrieval_relevance(question, chunks)
        logger.debug("retrieval_relevance_score=%.3f", relevance_score)

        faithfulness_score, faithfulness_evidence = score_answer_faithfulness(answer, chunks, question)
        logger.debug("faithfulness_score=%.3f", faithfulness_score)

        embedding_space = analyze_embedding_space(
            query_embedding=np.array(retrieval_result.query_embedding),
            chunk_embeddings=[np.array(e) for e in retrieval_result.chunk_embeddings],
            chunk_ids=[c.chunk_id for c in chunks],
        )
    except Exception as exc:
        logger.exception("analyze failed for example_id=%s", request.example_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("analyze complete: example_id=%s", request.example_id)

    return AnalyzeResponse(
        question=question,
        generated_answer=answer,
        retrieved_chunks=[c.text for c in chunks],
        ragas=RAGASMetrics(
            retrieval_relevance_score=relevance_score,
            faithfulness_score=faithfulness_score,
            relevance_evidence=relevance_evidence,
            faithfulness_evidence=faithfulness_evidence,
        ),
        retrieval_score_distribution=_STUB_DIMENSION,
        hedging_verification_mismatch=_STUB_DIMENSION,
        chunk_attribution=_STUB_DIMENSION,
        confidence_calibration=_STUB_DIMENSION,
        attribution_map=[],
        retrieval_distribution=analyze_retrieval_distribution(chunks),
        embedding_space=embedding_space,
    )
