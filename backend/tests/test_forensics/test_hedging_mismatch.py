"""Tests for hedging mismatch forensics module (Issue #6).

All Claude API calls are mocked. classify_confidence tests need no mocking.
"""
import inspect
import json
import pytest
from unittest.mock import MagicMock

from models import ClaimEntry, HedgingMismatchMetrics, RetrievedChunk
from services.forensics.hedging_mismatch import analyze_hedging_mismatch, classify_confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=f"c{i}", text=f"chunk text {i}", score=round(0.9 - i * 0.1, 1))
        for i in range(n)
    ]


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_mock(mocker, responses: list) -> MagicMock:
    """Patch anthropic.Anthropic; responses is a list of str (returned) or Exception instance (raised)."""
    mock_client = MagicMock()
    side_effects = []
    for r in responses:
        if isinstance(r, BaseException):
            side_effects.append(r)
        else:
            side_effects.append(_mock_response(r))
    mock_client.messages.create.side_effect = side_effects
    mock_cls = MagicMock(return_value=mock_client)
    mocker.patch("services.forensics.hedging_mismatch.anthropic.Anthropic", mock_cls)
    return mock_client


# ---------------------------------------------------------------------------
# classify_confidence — pure function, no mocking needed
# ---------------------------------------------------------------------------

def test_classify_confidence_definitive():
    assert classify_confidence("The deadline is March 15.") == "definitive"


def test_classify_confidence_hedged_modal():
    assert classify_confidence("It may apply after March.") == "hedged"


def test_classify_confidence_hedged_first_person():
    assert classify_confidence("I think the fee is $50.") == "hedged"


def test_classify_confidence_hedged_approximator():
    assert classify_confidence("Approximately 50 users were affected.") == "hedged"


def test_classify_confidence_uncertain_not_sure():
    assert classify_confidence("I'm not sure this applies.") == "uncertain"


def test_classify_confidence_uncertain_unclear():
    assert classify_confidence("It's unclear whether this applies.") == "uncertain"


# ---------------------------------------------------------------------------
# Test 7 — all definitive + not supported → overconfident_fraction=1.0
# ---------------------------------------------------------------------------

def test_all_definitive_not_supported(mocker):
    chunks = _chunks(2)
    mock_client = _make_mock(mocker, [
        '["The deadline is March 15."]',  # extraction: 1 definitive claim
        "not_supported",  # chunk c0
        "not_supported",  # chunk c1
    ])
    result = analyze_hedging_mismatch("The deadline is March 15.", chunks)
    assert result.overconfident_fraction == pytest.approx(1.0)
    assert result.underconfident_fraction == pytest.approx(0.0)
    assert result.total_claims == 1
    assert result.claim_breakdown[0].mismatch_type == "overconfident"


# ---------------------------------------------------------------------------
# Test 8 — all hedged + supported → matched (binary entailment cannot establish underconfidence)_fraction=1.0
# ---------------------------------------------------------------------------

def test_all_hedged_supported(mocker):
    chunks = _chunks(2)
    _make_mock(mocker, [
        '["It may apply after March."]',  # extraction: 1 hedged claim
        "supported",  # chunk c0 → short-circuit, c1 not checked
    ])
    result = analyze_hedging_mismatch("It may apply after March.", chunks)
    assert result.underconfident_fraction == pytest.approx(0.0)
    assert result.overconfident_fraction == pytest.approx(0.0)
    assert result.claim_breakdown[0].mismatch_type == "matched"


# ---------------------------------------------------------------------------
# Test 9 — all matched → both fractions 0.0
# ---------------------------------------------------------------------------

def test_all_matched(mocker):
    chunks = _chunks(2)
    _make_mock(mocker, [
        '["The deadline is March 15.", "It may apply after March."]',
        "supported",      # claim 0 (definitive) + supported → matched, short-circuit
        "not_supported",  # claim 1 (hedged) + not_supported → matched, chunk c0
        "not_supported",  # claim 1, chunk c1
    ])
    result = analyze_hedging_mismatch("answer", chunks)
    assert result.overconfident_fraction == pytest.approx(0.0)
    assert result.underconfident_fraction == pytest.approx(0.0)
    for entry in result.claim_breakdown:
        assert entry.mismatch_type == "matched"


# ---------------------------------------------------------------------------
# Test 10 — no claims extracted → zeroed metrics, empty breakdown, no crash
# ---------------------------------------------------------------------------

def test_no_claims_extracted(mocker):
    chunks = _chunks(2)
    _make_mock(mocker, ["[]"])
    result = analyze_hedging_mismatch("No factual claims here.", chunks)
    assert result.total_claims == 0
    assert result.claim_breakdown == []
    assert result.overconfident_fraction == pytest.approx(0.0)
    assert result.underconfident_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 11 — mixed claims, fractions sum <= 1.0
# ---------------------------------------------------------------------------

def test_mixed_claims_fractions(mocker):
    chunks = _chunks(3)
    _make_mock(mocker, [
        '["The deadline is March 15.", "It may apply after March.", "The fee is $50."]',
        # claim 0: definitive + not_supported → overconfident (k=3 chunks all checked)
        "not_supported", "not_supported", "not_supported",
        # claim 1: hedged + supported → matched (binary entailment cannot establish underconfidence) (short-circuit at c0)
        "supported",
        # claim 2: definitive + supported → matched (short-circuit at c0)
        "supported",
    ])
    result = analyze_hedging_mismatch("answer", chunks)
    assert result.total_claims == 3
    assert result.overconfident_fraction == pytest.approx(1 / 3)
    assert result.underconfident_fraction == pytest.approx(0.0)
    assert result.overconfident_fraction + result.underconfident_fraction <= 1.0


# ---------------------------------------------------------------------------
# Test 12 — total_claims == len(claim_breakdown)
# ---------------------------------------------------------------------------

def test_total_claims_equals_breakdown_length(mocker):
    chunks = _chunks(1)
    _make_mock(mocker, [
        '["Claim A.", "Claim B.", "Claim C."]',
        "not_supported",  # claim 0
        "not_supported",  # claim 1
        "not_supported",  # claim 2
    ])
    result = analyze_hedging_mismatch("answer", chunks)
    assert result.total_claims == len(result.claim_breakdown)
    assert result.total_claims == 3


# ---------------------------------------------------------------------------
# Test 13 — mismatch_type correctly set for all combinations
# ---------------------------------------------------------------------------

def test_mismatch_type_definitive_not_supported(mocker):
    _make_mock(mocker, ['["The deadline is March 15."]', "not_supported"])
    result = analyze_hedging_mismatch("The deadline is March 15.", _chunks(1))
    assert result.claim_breakdown[0].mismatch_type == "overconfident"


def test_mismatch_type_hedged_supported(mocker):
    _make_mock(mocker, ['["It may apply after March."]', "supported"])
    result = analyze_hedging_mismatch("It may apply after March.", _chunks(1))
    assert result.claim_breakdown[0].mismatch_type == "matched"


def test_mismatch_type_definitive_supported(mocker):
    _make_mock(mocker, ['["The fee is $50."]', "supported"])
    result = analyze_hedging_mismatch("The fee is $50.", _chunks(1))
    assert result.claim_breakdown[0].mismatch_type == "matched"


def test_mismatch_type_hedged_not_supported(mocker):
    _make_mock(mocker, ['["It may apply after March."]', "not_supported"])
    result = analyze_hedging_mismatch("It may apply after March.", _chunks(1))
    assert result.claim_breakdown[0].mismatch_type == "matched"


def test_mismatch_type_uncertain_not_supported(mocker):
    _make_mock(mocker, ['["I\'m not sure this applies."]', "not_supported"])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.claim_breakdown[0].mismatch_type == "matched"


def test_mismatch_type_uncertain_supported(mocker):
    _make_mock(mocker, ['["I\'m not sure this applies."]', "supported"])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.claim_breakdown[0].mismatch_type == "matched"


# ---------------------------------------------------------------------------
# Test 14 — extraction throws → zeroed metrics, no exception raised
# ---------------------------------------------------------------------------

def test_extraction_failure_is_explicit_not_healthy(mocker):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API error")
    mock_cls = MagicMock(return_value=mock_client)
    mocker.patch("services.forensics.hedging_mismatch.anthropic.Anthropic", mock_cls)

    result = analyze_hedging_mismatch("answer", _chunks(2))
    assert result.total_claims == 0
    assert result.claim_breakdown == []
    assert result.overconfident_fraction == pytest.approx(0.0)
    assert result.underconfident_fraction == pytest.approx(0.0)
    assert result.status == "error"
    assert result.error == "claim_extraction_failed"


# ---------------------------------------------------------------------------
# Test 15 — extraction returns invalid JSON → zeroed metrics, no exception
# ---------------------------------------------------------------------------

def test_extraction_invalid_json_returns_zeroed(mocker):
    _make_mock(mocker, ["not valid json at all"])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.total_claims == 0
    assert result.claim_breakdown == []
    assert result.status == "error"
    assert result.error == "claim_extraction_parse_failed"


def test_extraction_response_wrapped_in_code_fence_is_rejected(mocker):
    fenced = '```json\n["The deadline is March 15."]\n```'
    _make_mock(mocker, [fenced])
    result = analyze_hedging_mismatch("The deadline is March 15.", _chunks(1))
    assert result.status == "error"
    assert result.error == "claim_extraction_parse_failed"


def test_extraction_response_wrapped_in_plain_code_fence_is_rejected(mocker):
    fenced = '```\n["The deadline is March 15."]\n```'
    _make_mock(mocker, [fenced])
    result = analyze_hedging_mismatch("The deadline is March 15.", _chunks(1))
    assert result.status == "error"
    assert result.error == "claim_extraction_parse_failed"


@pytest.mark.parametrize("payload", [
    '["The deadline is March 15."] trailing prose',
    '["The deadline is March 15."]\n```',
])
def test_extraction_trailing_content_is_rejected_as_parse_failure(mocker, payload):
    _make_mock(mocker, [payload])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.status == "error"
    assert result.error == "claim_extraction_parse_failed"


@pytest.mark.parametrize("payload", [
    '{"claim": "The deadline is March 15."}',
    '"The deadline is March 15."',
    '["The deadline is March 15.", 23]',
    '[null]',
])
def test_extraction_schema_violations_are_rejected(mocker, payload):
    _make_mock(mocker, [payload])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.status == "error"
    assert result.error == "claim_extraction_schema_failed"
    assert result.total_claims == 0


def test_extraction_uses_exact_array_of_strings_schema(mocker):
    mock_client = _make_mock(mocker, ["[]"])
    result = analyze_hedging_mismatch("No factual claims.", _chunks(1))

    output_config = mock_client.messages.create.call_args.kwargs["output_config"]
    assert output_config == {
        "format": {
            "type": "json_schema",
            "schema": {"items": {"type": "string"}, "type": "array"},
        }
    }
    assert result.status == "ok"
    assert result.error is None


# ---------------------------------------------------------------------------
# Test 16 — per-claim entailment failure → that claim is unavailable, others unaffected
# ---------------------------------------------------------------------------

def test_per_claim_entailment_failure_is_isolated(mocker):
    chunks = _chunks(1)  # 1 chunk → k=1 for pre-filter
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_response('["The deadline is March 15.", "It may apply after March."]'),  # extraction
        Exception("timeout"),   # claim 0 (definitive) entailment raises → unavailable
        _mock_response("supported"),  # claim 1 (hedged) entailment succeeds → supported
    ]
    mock_cls = MagicMock(return_value=mock_client)
    mocker.patch("services.forensics.hedging_mismatch.anthropic.Anthropic", mock_cls)

    result = analyze_hedging_mismatch("answer", chunks)
    # Must NOT be zeroed — extraction succeeded
    assert result.total_claims == 2
    # claim 0 is excluded from mismatch metrics rather than converted to a negative verdict
    assert result.claim_breakdown[0].supported is None
    assert result.claim_breakdown[0].mismatch_type is None
    assert result.claim_breakdown[0].entailment_checks[0].status == "error"
    # claim 1: hedged + supported → matched (binary entailment cannot establish underconfidence)
    assert result.claim_breakdown[1].supported is True
    assert result.claim_breakdown[1].mismatch_type == "matched"
    assert result.evaluated_claim_count == 1
    assert result.unavailable_claim_count == 1


# ---------------------------------------------------------------------------
# Test 17 — CLAIM_EXTRACTION_PROMPT and ENTAILMENT_PROMPT imported from constants
# ---------------------------------------------------------------------------

def test_prompts_imported_from_constants(mocker):
    from prompts.hedging_prompts import CLAIM_EXTRACTION_PROMPT, ENTAILMENT_PROMPT

    chunks = [RetrievedChunk(chunk_id="c0", text="chunk text here", score=0.9)]
    answer = "The answer is 42."

    mock_client = _make_mock(mocker, ['["The answer is 42."]', "not_supported"])
    analyze_hedging_mismatch(answer, chunks)

    calls = mock_client.messages.create.call_args_list
    # First call: claim extraction
    extraction_content = calls[0].kwargs["messages"][0]["content"]
    assert extraction_content == CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    # Second call: entailment
    entailment_content = calls[1].kwargs["messages"][0]["content"]
    assert entailment_content == ENTAILMENT_PROMPT.format(
        chunk_text=chunks[0].text, claim="The answer is 42."
    )


# ---------------------------------------------------------------------------
# Test 18 — overconfident_fraction + underconfident_fraction <= 1.0 always
# ---------------------------------------------------------------------------

def test_fractions_sum_at_most_one(mocker):
    chunks = _chunks(1)
    _make_mock(mocker, [
        '["The deadline is March 15.", "It may apply after March."]',
        "not_supported",  # claim 0 (definitive) → overconfident
        "supported",      # claim 1 (hedged) → underconfident
    ])
    result = analyze_hedging_mismatch("answer", chunks)
    assert result.overconfident_fraction + result.underconfident_fraction <= 1.0
    assert result.overconfident_fraction == pytest.approx(0.5)
    assert result.underconfident_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 19 — AnalyzeResponse has hedging_mismatch, not hedging_verification_mismatch
# ---------------------------------------------------------------------------

def test_analyze_response_has_hedging_mismatch_field():
    from models import AnalyzeResponse
    fields = AnalyzeResponse.model_fields
    assert "hedging_mismatch" in fields
    assert "hedging_verification_mismatch" not in fields


# ---------------------------------------------------------------------------
# Test 20 — DimensionResult not imported in hedging_mismatch module
# ---------------------------------------------------------------------------

def test_dimension_result_not_imported_in_module():
    import services.forensics.hedging_mismatch as mod
    source = inspect.getsource(mod)
    assert "DimensionResult" not in source


# ---------------------------------------------------------------------------
# Issue #24 — exact entailment response contract
# ---------------------------------------------------------------------------

def test_entailment_supported_with_trailing_punctuation_is_invalid(mocker):
    _make_mock(mocker, ['["It may apply after March."]', "supported."])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.claim_breakdown[0].supported is None
    assert result.claim_breakdown[0].mismatch_type is None
    assert result.claim_breakdown[0].entailment_checks[0].status == "invalid_format"


def test_entailment_supported_with_prefix_is_invalid(mocker):
    _make_mock(mocker, ['["It may apply after March."]', "yes, supported"])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.claim_breakdown[0].supported is None


def test_entailment_not_supported_with_trailing_punctuation_is_invalid(mocker):
    _make_mock(mocker, ['["The deadline is March 15."]', "not_supported."])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.claim_breakdown[0].supported is None
    assert result.overconfident_fraction == 0.0


def test_entailment_not_supported_capitalized_is_invalid(mocker):
    _make_mock(mocker, ['["The deadline is March 15."]', "Not supported"])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.claim_breakdown[0].supported is None


def test_entailment_unexpected_response_is_unavailable(mocker):
    _make_mock(mocker, ['["The deadline is March 15."]', "I cannot determine this."])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    assert result.claim_breakdown[0].supported is None
    assert result.total_claims == 1
    assert result.evaluated_claim_count == 0
    assert result.unavailable_claim_count == 1


@pytest.mark.parametrize("verdict", [
    "supported because the context says so",
    "not_supported: the number contradicts the claim",
    "unsupported",
])
def test_entailment_arbitrary_prose_cannot_supply_a_verdict(mocker, verdict):
    _make_mock(mocker, ['["The deadline is March 15."]', verdict])
    result = analyze_hedging_mismatch("answer", _chunks(1))
    check = result.claim_breakdown[0].entailment_checks[0]
    assert check.status == "invalid_format"
    assert check.verdict is None
    assert result.claim_breakdown[0].supported is None


def test_invalid_chunk_then_supported_chunk_preserves_short_circuit(mocker):
    mock_client = _make_mock(mocker, [
        '["The deadline is March 15."]',
        "supported because...",
        "supported",
    ])
    result = analyze_hedging_mismatch("answer", _chunks(3))
    entry = result.claim_breakdown[0]
    assert entry.supported is True
    assert entry.source_chunk_id == "c1"
    assert [check.status for check in entry.entailment_checks] == [
        "invalid_format", "evaluated"
    ]
    assert mock_client.messages.create.call_count == 3


def test_valid_negative_and_errors_preserve_per_chunk_coverage(mocker):
    _make_mock(mocker, [
        '["The deadline is March 15."]',
        "not_supported",
        Exception("timeout"),
        "not_supported",
    ])
    result = analyze_hedging_mismatch("answer", _chunks(3))
    entry = result.claim_breakdown[0]
    assert entry.supported is False
    assert entry.mismatch_type == "overconfident"
    assert [check.status for check in entry.entailment_checks] == [
        "evaluated", "error", "evaluated"
    ]
    assert result.evaluated_chunk_count == 2
