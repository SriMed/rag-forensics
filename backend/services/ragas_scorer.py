import logging
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._context_precision import context_precision
from langchain_anthropic import ChatAnthropic
from models import RetrievedChunk

logger = logging.getLogger(__name__)
_LLM_MODEL = "claude-haiku-4-5-20251001"


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
) -> tuple[float, list[str]]:
    logger.debug("running ragas metric=%s", metric_name)
    llm = ChatAnthropic(model=_LLM_MODEL)
    dataset = EvaluationDataset(samples=[sample])
    result = evaluate(dataset, metrics=[metric], llm=llm, show_progress=False)
    score = float(result[metric_name][0])
    logger.debug("ragas metric=%s score=%.3f", metric_name, score)
    return score, _extract_evidence(chunks)


def score_retrieval_relevance(question: str, chunks: list[RetrievedChunk]) -> tuple[float, list[str]]:
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[c.text for c in chunks],
        reference="N/A",
    )
    return _run_ragas(sample, context_precision, "context_precision", chunks)


def score_answer_faithfulness(
    answer: str, chunks: list[RetrievedChunk], question: str
) -> tuple[float, list[str]]:
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[c.text for c in chunks],
        response=answer,
    )
    return _run_ragas(sample, faithfulness, "faithfulness", chunks)
