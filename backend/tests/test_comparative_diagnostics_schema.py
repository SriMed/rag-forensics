import pytest
from pydantic import ValidationError

from benchmark.comparative_diagnostics import (
    CaseJudgments,
    ComparativeCase,
    ComparativeCaseSet,
    NativeSystemOutput,
    SystemDiagnosticRecord,
    make_population_sha256,
)


def _native(status="healthy", raw=None):
    return NativeSystemOutput(
        system="rag_forensics",
        system_version="0.1.0",
        availability=status,
        raw_output=raw if raw is not None else {"verdict_signals": []},
        error=None if status != "failed" else "extractor_timeout",
    )


def _record(**overrides):
    defaults = dict(
        system="rag_forensics",
        native=_native(),
        suspected_component="evidence_selection",
        supporting_observation="Low retrieval-to-answer similarity for sentence 2.",
        evidence_attribution=["chunk_3"],
        method="semantic_similarity",
        reliability="unvalidated",
        causal_strength_language="candidate, not proof",
        proposed_intervention="Re-run generation with annotated evidence supplied.",
        no_equivalent_fields=[],
    )
    defaults.update(overrides)
    return SystemDiagnosticRecord(**defaults)


def _judgments(**overrides):
    defaults = dict(
        dataset_label="unsupported",
        model_judgments={"rag_forensics": "unsupported", "ragchecker": "unsupported"},
        reviewer_judgment=None,
        intervention_evidence=None,
    )
    defaults.update(overrides)
    return CaseJudgments(**defaults)


def _case(**overrides):
    defaults = dict(
        case_id="techqa_DEV_Q243-a",
        dataset="ragbench",
        domain="techqa",
        dataset_revision="ragbench@2026-01-15",
        stratum="component_diagnoses_disagree",
        selection_rationale="RAG Forensics flags evidence_selection; RAGChecker flags faithfulness.",
        systems={"rag_forensics": _record()},
        judgments=_judgments(),
    )
    defaults.update(overrides)
    return ComparativeCase(**defaults)


class TestNativeSystemOutput:
    def test_availability_restricted_to_four_states(self):
        for ok in ("healthy", "missing", "unavailable", "failed"):
            _native(status=ok)
        with pytest.raises(ValidationError):
            NativeSystemOutput(
                system="ragchecker",
                system_version="0.1.9",
                availability="ok",  # not one of the four declared states
                raw_output={},
                error=None,
            )

    def test_failed_requires_error_message(self):
        with pytest.raises(ValidationError):
            NativeSystemOutput(
                system="ragvue",
                system_version="0.6.1",
                availability="failed",
                raw_output={},
                error=None,
            )

    def test_healthy_forbids_error_message(self):
        with pytest.raises(ValidationError):
            NativeSystemOutput(
                system="ragvue",
                system_version="0.6.1",
                availability="healthy",
                raw_output={},
                error="should not be set",
            )


class TestSystemDiagnosticRecord:
    def test_preserves_native_output_verbatim_alongside_mapped_fields(self):
        raw = {"faithfulness": 0.42, "claims": ["Paris is the capital of France."]}
        record = _record(native=_native(raw=raw))
        assert record.native.raw_output == raw
        assert record.suspected_component == "evidence_selection"

    def test_no_equivalent_fields_records_unmappable_constructs_instead_of_forcing_a_score(self):
        record = _record(
            reliability=None,
            no_equivalent_fields=["calibration_retrieval_coverage has no RAG Forensics analogue"],
        )
        assert record.reliability is None
        assert record.no_equivalent_fields


class TestCaseJudgments:
    def test_dataset_label_model_and_reviewer_judgments_and_intervention_evidence_are_distinct_fields(self):
        judgments = _judgments(
            reviewer_judgment="supported_with_caveat",
            intervention_evidence="Oracle evidence flips the verdict to supported.",
        )
        assert judgments.dataset_label == "unsupported"
        assert judgments.model_judgments["ragchecker"] == "unsupported"
        assert judgments.reviewer_judgment == "supported_with_caveat"
        assert judgments.intervention_evidence == "Oracle evidence flips the verdict to supported."

    def test_model_judgments_cannot_silently_overwrite_dataset_label(self):
        judgments = _judgments()
        with pytest.raises(ValidationError):
            judgments.dataset_label = judgments.model_judgments["rag_forensics"]  # frozen model


class TestComparativeCase:
    def test_valid_minimal_case_round_trips(self):
        case = _case()
        dumped = case.model_dump()
        restored = ComparativeCase(**dumped)
        assert restored == case

    def test_rejects_unknown_stratum(self):
        with pytest.raises(ValidationError):
            _case(stratum="not_a_declared_stratum")

    def test_missing_system_output_uses_missing_state_not_omission(self):
        case = _case(
            systems={
                "rag_forensics": _record(),
                "ragvue": _record(
                    system="ragvue",
                    native=NativeSystemOutput(
                        system="ragvue",
                        system_version="0.6.1",
                        availability="missing",
                        raw_output=None,
                        error=None,
                    ),
                    suspected_component=None,
                    supporting_observation=None,
                    evidence_attribution=[],
                    method=None,
                    reliability=None,
                    causal_strength_language=None,
                    proposed_intervention=None,
                    no_equivalent_fields=[],
                ),
            }
        )
        assert case.systems["ragvue"].native.availability == "missing"
        assert case.systems["ragvue"].native.raw_output is None


class TestComparativeCaseSet:
    def test_draft_case_set_does_not_require_selection_rationale(self):
        case_set = ComparativeCaseSet(
            schema_version="comparative-diagnostics.v1",
            status="draft",
            population_sha256=make_population_sha256([]),
            cases=[],
        )
        assert case_set.status == "draft"

    def test_frozen_case_set_requires_reviewer_identity_and_nonempty_cases(self):
        with pytest.raises(ValidationError):
            ComparativeCaseSet(
                schema_version="comparative-diagnostics.v1",
                status="frozen",
                population_sha256=make_population_sha256([]),
                cases=[],
            )

    def test_frozen_case_set_with_reviewer_identity_and_cases_is_valid(self):
        case = _case()
        case_set = ComparativeCaseSet(
            schema_version="comparative-diagnostics.v1",
            status="frozen",
            reviewer_identity="reviewer@example.com",
            population_sha256=make_population_sha256([case.case_id]),
            cases=[case],
        )
        assert case_set.status == "frozen"

    def test_rejects_duplicate_case_ids(self):
        case = _case()
        with pytest.raises(ValidationError):
            ComparativeCaseSet(
                schema_version="comparative-diagnostics.v1",
                status="draft",
                population_sha256=make_population_sha256([case.case_id, case.case_id]),
                cases=[case, case],
            )


def test_make_population_sha256_is_deterministic_and_order_sensitive():
    a = make_population_sha256(["case-1", "case-2"])
    b = make_population_sha256(["case-1", "case-2"])
    c = make_population_sha256(["case-2", "case-1"])
    assert a == b
    assert a != c
