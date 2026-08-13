import subprocess

from evals.truncated_evidence.run_production_comparison import render, run


CASE = {
    "question": "What is the result?",
    "complete": "The result may be a risk factor.",
    "truncated": "The result may be a risk",
}


def test_baseline_preserves_pre_issue_27_prompt():
    system, prompt = render(CASE, "truncated", "pre_issue_27")
    assert "completeness=" not in prompt
    assert "missing continuation" not in system


def test_contract_marks_complete_source_evidence():
    _, prompt = render(CASE, "complete", "source_metadata_contract")
    assert "completeness=complete" in prompt
    assert "completeness_source=source" in prompt


def test_contract_marks_truncated_source_evidence():
    system, prompt = render(CASE, "truncated", "source_metadata_contract")
    assert "completeness=truncated" in prompt
    assert "do not supply or guess the missing continuation" in system


def test_runner_preserves_unavailable_failure(mocker):
    mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 120))
    result = run("system", "prompt")
    assert result == {"status": "unavailable", "response": "", "error": "TimeoutExpired"}
