"""Contract and failure-semantics tests for the installed RAGAS metrics."""
import math
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from models import RAGASMetricResult, RetrievedChunk


CHUNKS = [
    RetrievedChunk(chunk_id="c1", text="The mitochondria produces ATP.", score=0.9),
    RetrievedChunk(chunk_id="c2", text="Cells require energy to function.", score=0.8),
]


def _mock_evaluate(mocker, metric_name: str, score):
    return mocker.patch(
        "services.ragas_scorer.evaluate", return_value={metric_name: [score]}
    )


def test_installed_context_utilization_contract_is_pinned():
    import ragas
    from ragas.metrics._context_precision import ContextUtilization, context_utilization

    assert ragas.__version__ == "0.4.3"
    assert type(context_utilization) is ContextUtilization
    assert context_utilization.name == "context_utilization"
    assert context_utilization.required_columns == {
        "SINGLE_TURN": {"user_input", "response", "retrieved_contexts"}
    }
    assert context_utilization.context_precision_prompt.input_model.__name__ == "QAC"


def test_context_utilization_supplies_answer_without_reference(mocker):
    evaluate = _mock_evaluate(mocker, "context_utilization", 0.8)
    from services.ragas_scorer import score_context_utilization

    result, excerpts = score_context_utilization(
        "What makes ATP?", "Mitochondria make ATP.", CHUNKS
    )

    sample = evaluate.call_args.args[0].samples[0]
    assert sample.user_input == "What makes ATP?"
    assert sample.response == "Mitochondria make ATP."
    assert sample.retrieved_contexts == [chunk.text for chunk in CHUNKS]
    assert sample.reference is None
    assert result.score == pytest.approx(0.8)
    assert result.status == "ok"
    assert result.error is None
    assert excerpts


def test_faithfulness_supplies_its_installed_contract(mocker):
    evaluate = _mock_evaluate(mocker, "faithfulness", 0.9)
    from services.ragas_scorer import score_answer_faithfulness

    result, _ = score_answer_faithfulness(
        "Mitochondria make ATP.", CHUNKS, "What makes ATP?"
    )

    sample = evaluate.call_args.args[0].samples[0]
    assert sample.user_input == "What makes ATP?"
    assert sample.response == "Mitochondria make ATP."
    assert sample.retrieved_contexts == [chunk.text for chunk in CHUNKS]
    assert result.score == pytest.approx(0.9)
    assert result.status == "ok"


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    ("function_name", "metric_name", "args"),
    [
        ("score_context_utilization", "context_utilization", ("Q?", "A.", CHUNKS)),
        ("score_answer_faithfulness", "faithfulness", ("A.", CHUNKS, "Q?")),
    ],
)
def test_non_finite_scores_are_explicitly_unavailable(
    mocker, score, function_name, metric_name, args
):
    _mock_evaluate(mocker, metric_name, score)
    import services.ragas_scorer as scorer

    result, excerpts = getattr(scorer, function_name)(*args)

    assert result.score is None
    assert result.status == "unavailable"
    assert result.error == "non_finite_score"
    assert excerpts


@pytest.mark.parametrize(
    ("function_name", "args"),
    [
        ("score_context_utilization", ("Q?", "A.", CHUNKS)),
        ("score_answer_faithfulness", ("A.", CHUNKS, "Q?")),
    ],
)
def test_metric_exceptions_are_explicitly_unavailable(mocker, function_name, args):
    mocker.patch("services.ragas_scorer.evaluate", side_effect=RuntimeError("provider down"))
    import services.ragas_scorer as scorer

    result, excerpts = getattr(scorer, function_name)(*args)

    assert result.score is None
    assert result.status == "unavailable"
    assert result.error == "evaluation_failed"
    assert excerpts


def test_llm_initialization_failure_is_explicitly_unavailable(mocker):
    mocker.patch(
        "services.ragas_scorer.ChatAnthropic", side_effect=RuntimeError("missing credential")
    )
    from services.ragas_scorer import score_context_utilization

    result, _ = score_context_utilization("Q?", "A.", CHUNKS)

    assert result.score is None
    assert result.status == "unavailable"
    assert result.error == "evaluation_failed"


def test_context_excerpts_are_verbatim_and_limited_to_three():
    from services.ragas_scorer import _extract_context_excerpts

    excerpts = _extract_context_excerpts(CHUNKS)
    combined = " ".join(chunk.text for chunk in CHUNKS)
    assert 1 <= len(excerpts) <= 3
    assert all(excerpt in combined for excerpt in excerpts)


def test_labeled_context_utilization_cases_cover_required_contrasts():
    path = Path(__file__).parents[1] / "evals/context_utilization/v1/cases.json"
    cases = json.loads(path.read_text())
    by_id = {case["id"]: case for case in cases}
    assert by_id["relevant_single"]["human_context_labels"] == [1]
    assert by_id["irrelevant_single"]["human_context_labels"] == [0]
    assert by_id["relevant_then_irrelevant"]["human_context_labels"] == [1, 0]
    assert by_id["irrelevant_then_relevant"]["human_context_labels"] == [0, 1]


def test_reviewed_context_utilization_results_match_all_human_labels():
    path = (
        Path(__file__).parents[1]
        / "evals/context_utilization/v1/reviewed-results.json"
    )
    review = json.loads(path.read_text())
    assert review["review_status"] == "reviewed"
    assert review["summary"]["before_context_label_matches"] == 3
    assert review["summary"]["after_context_label_matches"] == 6
    assert review["summary"]["after_context_label_total"] == 6
    for result in review["results"]:
        assert result["after_context_verdicts"] == result["human_context_labels"]
    assert (
        next(item for item in review["results"] if item["id"] == "relevant_then_irrelevant")["after_score"]
        > next(item for item in review["results"] if item["id"] == "irrelevant_then_relevant")["after_score"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score": None, "status": "ok"},
        {"score": 0.4, "status": "unavailable", "error": "evaluation_failed"},
        {"score": None, "status": "unavailable"},
    ],
)
def test_metric_result_rejects_inconsistent_states(kwargs):
    with pytest.raises(ValidationError):
        RAGASMetricResult(**kwargs)
