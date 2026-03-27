import logging
from fastapi import APIRouter, HTTPException
from models import AnalyzeRequest, AnalyzeResponse, DimensionResult, AttributionEntry
from services.retriever import retrieve_for_example
from services.generator import generate_answer
from services.ragas_scorer import score_retrieval_relevance, score_answer_faithfulness

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
        question, chunks = retrieve_for_example(request.example_id)
        logger.debug("retrieved %d chunks for example_id=%s", len(chunks), request.example_id)

        answer = generate_answer(question, chunks)
        logger.debug("generated answer (%d chars)", len(answer))

        retrieval_relevance = score_retrieval_relevance(question, chunks)
        logger.debug("retrieval_relevance: verdict=%s", retrieval_relevance.verdict)

        answer_faithfulness = score_answer_faithfulness(answer, chunks, question)
        logger.debug("answer_faithfulness: verdict=%s", answer_faithfulness.verdict)
    except Exception as exc:
        logger.exception("analyze failed for example_id=%s", request.example_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("analyze complete: example_id=%s", request.example_id)

    return AnalyzeResponse(
        question=question,
        generated_answer=answer,
        retrieved_chunks=[c.text for c in chunks],
        retrieval_relevance=retrieval_relevance,
        answer_faithfulness=answer_faithfulness,
        retrieval_score_distribution=_STUB_DIMENSION,
        hedging_verification_mismatch=_STUB_DIMENSION,
        chunk_attribution=_STUB_DIMENSION,
        confidence_calibration=_STUB_DIMENSION,
        attribution_map=[],
    )
