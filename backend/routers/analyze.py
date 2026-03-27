from fastapi import APIRouter
from models import AnalyzeRequest, AnalyzeResponse, DimensionResult, AttributionEntry

router = APIRouter()

_STUB_DIMENSION = DimensionResult(
    verdict="pass",
    explanation="Stub response — not yet implemented.",
    evidence=[],
)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return AnalyzeResponse(
        question="Stub question for example_id: " + request.example_id,
        generated_answer="Stub generated answer.",
        retrieved_chunks=[],
        retrieval_relevance=_STUB_DIMENSION,
        answer_faithfulness=_STUB_DIMENSION,
        retrieval_score_distribution=_STUB_DIMENSION,
        hedging_verification_mismatch=_STUB_DIMENSION,
        chunk_attribution=_STUB_DIMENSION,
        confidence_calibration=_STUB_DIMENSION,
        attribution_map=[],
    )
