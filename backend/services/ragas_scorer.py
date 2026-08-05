import logging
import math
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._context_precision import context_utilization
from langchain_anthropic import ChatAnthropic
from models import RAGASMetricResult, RetrievedChunk

logger = logging.getLogger(__name__)
_LLM_MODEL = "claude-haiku-4-5-20251001"


def _extract_context_excerpts(chunks: list[RetrievedChunk], n: int = 3) -> list[str]:
    """Return context excerpts for inspection; these do not explain the judge score."""
    excerpts = []
    for chunk in chunks[:n]:
        text = chunk.text.strip()
        # First sentence keeps evidence readable without cutting a claim mid-thought.
        for sep in (". ", ".\n", "! ", "? "):
            idx = text.find(sep)
            if 0 < idx < 200:
                excerpts.append(text[: idx + 1])
                break
        else:
            excerpts.append(text[:150])
    return excerpts


def _run_ragas(
    sample: SingleTurnSample,
    metric,
    metric_name: str,
    chunks: list[RetrievedChunk],
) -> tuple[RAGASMetricResult, list[str]]:
    logger.debug("running ragas metric=%s", metric_name)
    excerpts = _extract_context_excerpts(chunks)
    try:
        llm = ChatAnthropic(model=_LLM_MODEL)
        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(dataset, metrics=[metric], llm=llm, show_progress=False)
        score = float(result[metric_name][0])
    except Exception:
        logger.warning("ragas metric=%s evaluation failed", metric_name, exc_info=True)
        return RAGASMetricResult(score=None, status="unavailable", error="evaluation_failed"), excerpts
    if not math.isfinite(score):
        logger.warning("ragas metric=%s returned a non-finite score", metric_name)
        return RAGASMetricResult(score=None, status="unavailable", error="non_finite_score"), excerpts
    logger.debug("ragas metric=%s score=%.3f", metric_name, score)
    return RAGASMetricResult(score=score, status="ok"), excerpts


def score_context_utilization(
    question: str, answer: str, chunks: list[RetrievedChunk]
) -> tuple[RAGASMetricResult, list[str]]:
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[c.text for c in chunks],
        response=answer,
    )
    return _run_ragas(sample, context_utilization, "context_utilization", chunks)


def score_answer_faithfulness(
    answer: str, chunks: list[RetrievedChunk], question: str
) -> tuple[RAGASMetricResult, list[str]]:
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[c.text for c in chunks],
        response=answer,
    )
    return _run_ragas(sample, faithfulness, "faithfulness", chunks)
