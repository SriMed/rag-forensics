import logging
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._context_precision import context_precision
from langchain_anthropic import ChatAnthropic
from models import RetrievedChunk, DimensionResult

logger = logging.getLogger(__name__)
_LLM_MODEL = "claude-haiku-4-5-20251001"


def _verdict_from_score(score: float) -> str:
    if score >= 0.75:
        return "pass"
    if score >= 0.5:
        return "warn"
    return "fail"


def _extract_evidence(chunks: list[RetrievedChunk], n: int = 3) -> list[str]:
    """Return up to n verbatim spans, one per top chunk."""
    evidence = []
    for chunk in chunks[:n]:
        text = chunk.text.strip()
        # First sentence keeps evidence readable without cutting a claim mid-thought.
        for sep in (". ", ".\n", "! ", "? "):
            idx = text.find(sep)
            if 0 < idx < 200:
                evidence.append(text[: idx + 1])
                break
        else:
            evidence.append(text[:150])
    return evidence


def _run_ragas(
    sample: SingleTurnSample,
    metric,
    metric_name: str,
    chunks: list[RetrievedChunk],
    label: str,
) -> DimensionResult:
    logger.debug("running ragas metric=%s", metric_name)
    llm = ChatAnthropic(model=_LLM_MODEL)
    dataset = EvaluationDataset(samples=[sample])
    result = evaluate(dataset, metrics=[metric], llm=llm, show_progress=False)
    score = float(result[metric_name][0])
    verdict = _verdict_from_score(score)
    logger.debug("ragas metric=%s score=%.3f verdict=%s", metric_name, score, verdict)
    return DimensionResult(
        verdict=verdict,
        explanation=f"{label} score: {score:.2f}",
        evidence=_extract_evidence(chunks),
    )


def score_retrieval_relevance(question: str, chunks: list[RetrievedChunk]) -> DimensionResult:
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[c.text for c in chunks],
        reference="N/A",
    )
    return _run_ragas(sample, context_precision, "context_precision", chunks, "Retrieval relevance")


def score_answer_faithfulness(
    answer: str, chunks: list[RetrievedChunk], question: str
) -> DimensionResult:
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[c.text for c in chunks],
        response=answer,
    )
    return _run_ragas(sample, faithfulness, "faithfulness", chunks, "Answer faithfulness")
