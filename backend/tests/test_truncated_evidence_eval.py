import pytest

from evals.truncated_evidence.run_comparison import (
    CONDITIONS,
    assess,
    detect_incomplete_terminal_sentence,
    render,
)


def test_detector_distinguishes_complete_and_incomplete_prose():
    assert detect_incomplete_terminal_sentence("A lower count may be a risk") is True
    assert detect_incomplete_terminal_sentence("A lower count may be a risk factor.") is False
    assert detect_incomplete_terminal_sentence("Is this complete?") is False


@pytest.mark.parametrize("text", [
    "SECTION HEADING",
    "status: ready",
    '[["year", "value"], ["2025", "10"]]',
    "42",
])
def test_detector_has_known_structured_text_false_positives(text):
    """The heuristic is intentionally not authoritative completeness provenance."""
    assert detect_incomplete_terminal_sentence(text) is True


def test_hybrid_adds_metadata_only_when_detector_fires():
    _, complete_prompt, complete_detected = render("Question?", "Complete evidence.", "hybrid")
    _, truncated_prompt, truncated_detected = render("Question?", "Incomplete evidence", "hybrid")
    assert complete_detected is False
    assert "Metadata:" not in complete_prompt
    assert truncated_detected is True
    assert "Metadata:" in truncated_prompt


def test_all_four_conditions_are_preserved():
    assert CONDITIONS == ("baseline", "metadata", "qualification", "hybrid")


def test_assessment_keeps_deterministic_labels_separate_from_human_review():
    case = {
        "forbidden_additions_when_truncated": ["risk factor"],
        "required_complete": ["supported"],
    }
    result = assess(case, "truncated", "The incomplete evidence does not establish a risk factor.")
    assert result["forbidden_phrase_present"] is True
    assert result["explicit_truncation"] is True
    assert result["complete_answer_useful"] is None
    assert result["human_review"] is None
