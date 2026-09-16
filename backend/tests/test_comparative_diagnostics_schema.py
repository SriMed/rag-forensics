import pytest
from pydantic import ValidationError

from benchmark.comparative_diagnostics import (
    CANDIDATE_POOL_DOMAINS,
    CANDIDATE_POOL_LIMIT,
    CANDIDATE_POOL_REVISION,
    CANDIDATE_POOL_SEED,
    CANDIDATE_POOL_SPLIT,
    CaseJudgments,
    ComparativeCase,
    ComparativeCaseSet,
    NativeSystemOutput,
    RAGCHECKER_METRICS_REQUIRING_GT_ANSWER,
    SystemDiagnosticRecord,
    assert_ragchecker_metrics_computable,
    dataset_label_for_record,
    load_case_candidate_pool,
    make_population_sha256,
    map_rag_forensics_native,
    map_ragchecker_native,
    map_ragvue_native,
    ragchecker_result_input,
    ragvue_item_from_record,
)
from models import (
    BenchmarkSentence,
    BenchmarkSentenceSupport,
    RAGBenchEvaluationRecord,
    RetrievedChunk,
    VerdictSignal,
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


def _ragbench_record(unsupported_keys=frozenset()):
    return RAGBenchEvaluationRecord(
        example_id="techqa_DEV_Q243",
        domain="techqa",
        question="What is the capital of France?",
        response="Paris is the capital of France.",
        chunks=[RetrievedChunk(chunk_id="doc_0", text="Paris is the capital of France.", score=0.9)],
        response_sentences=[],
        document_sentences=[],
        document_sentence_keys=set(),
        unsupported_response_sentence_keys=set(unsupported_keys),
        sentence_support={},
    )


class TestDatasetLabelForRecord:
    def test_fully_supported_when_no_unsupported_sentences(self):
        assert dataset_label_for_record(_ragbench_record()) == "fully_supported"

    def test_contains_unsupported_when_any_sentence_flagged(self):
        record = _ragbench_record(unsupported_keys={"a"})
        assert dataset_label_for_record(record) == "contains_unsupported"


class TestRagcheckerResultInput:
    def test_maps_record_without_inventing_a_reference_answer(self):
        record = _ragbench_record()
        payload = ragchecker_result_input(record)
        assert payload["query_id"] == record.example_id
        assert payload["query"] == record.question
        assert payload["response"] == record.response
        assert payload["gt_answer"] == ""  # RAGBench has no reference answer distinct from response
        assert payload["retrieved_context"] == [{"doc_id": "doc_0", "text": "Paris is the capital of France."}]

    def test_metrics_requiring_gt_answer_are_named_explicitly(self):
        # faithfulness is the only RAGChecker metric that does not derive from gt_answer.
        assert "faithfulness" not in RAGCHECKER_METRICS_REQUIRING_GT_ANSWER
        assert "claim_recall" in RAGCHECKER_METRICS_REQUIRING_GT_ANSWER
        assert "context_precision" in RAGCHECKER_METRICS_REQUIRING_GT_ANSWER


class TestAssertRagcheckerMetricsComputable:
    def test_faithfulness_is_computable_without_a_reference_answer(self):
        assert_ragchecker_metrics_computable(["faithfulness"], has_gt_answer=False)

    def test_gt_answer_dependent_metric_without_a_reference_answer_raises(self):
        with pytest.raises(ValueError, match="gt_answer"):
            assert_ragchecker_metrics_computable(["claim_recall"], has_gt_answer=False)

    def test_all_metrics_permitted_once_a_reference_answer_exists(self):
        assert_ragchecker_metrics_computable(["claim_recall", "faithfulness"], has_gt_answer=True)


class TestRagvueItemFromRecord:
    def test_maps_to_reference_free_question_answer_contexts_shape(self):
        record = _ragbench_record()
        item = ragvue_item_from_record(record)
        assert item == {
            "question": record.question,
            "answer": record.response,
            "contexts": ["Paris is the capital of France."],
        }


class TestMapRagForensicsNative:
    def test_top_ranked_signal_becomes_the_suspected_component(self):
        signals = [
            VerdictSignal(
                name="evidence_selection_weak",
                priority_score=0.8,
                description="Low chunk-attribution score.",
                reliability="unvalidated",
            ),
            VerdictSignal(
                name="hedging_mismatch",
                priority_score=0.3,
                description="Definitive language with weak evidence.",
                reliability="model_judged",
            ),
        ]
        record = map_rag_forensics_native(
            case_id="techqa_DEV_Q243",
            verdict_signals=signals,
            raw_output={"verdict_signals": [s.model_dump() for s in signals]},
        )
        assert record.system == "rag_forensics"
        assert record.native.availability == "healthy"
        assert record.suspected_component == "evidence_selection_weak"
        assert record.reliability == "unvalidated"
        assert "not a calibrated severity" in record.causal_strength_language.lower() \
            or "heuristic priority" in record.causal_strength_language.lower()

    def test_no_signals_maps_to_unavailable_not_a_fabricated_component(self):
        record = map_rag_forensics_native(case_id="x", verdict_signals=[], raw_output={"verdict_signals": []})
        assert record.native.availability == "unavailable"
        assert record.suspected_component is None


class TestMapRagcheckerNative:
    def test_faithfulness_only_metrics_map_with_gt_answer_gap_named(self):
        record = map_ragchecker_native(
            case_id="techqa_DEV_Q243",
            metrics_for_item={"faithfulness": 0.62},
            requested_metrics=["faithfulness"],
        )
        assert record.system == "ragchecker"
        assert record.native.availability == "healthy"
        assert record.method == "ragchecker_faithfulness"
        assert record.suspected_component is None
        assert any("gt_answer" in note for note in record.no_equivalent_fields)

    def test_failed_run_is_marked_failed_not_healthy_zero(self):
        record = map_ragchecker_native(
            case_id="techqa_DEV_Q243",
            metrics_for_item=None,
            requested_metrics=["faithfulness"],
            error="litellm timeout",
        )
        assert record.native.availability == "failed"
        assert record.native.error == "litellm timeout"


class TestLoadCaseCandidatePool:
    def test_reuses_issue_29s_exact_population_parameters(self, mocker):
        load_records = mocker.patch(
            "benchmark.comparative_diagnostics._load_records", return_value=([], [])
        )
        load_case_candidate_pool()
        load_records.assert_called_once_with(
            list(CANDIDATE_POOL_DOMAINS), CANDIDATE_POOL_SPLIT, CANDIDATE_POOL_LIMIT,
            CANDIDATE_POOL_SEED, CANDIDATE_POOL_REVISION,
        )

    def test_pool_is_deduplicated_to_one_entry_per_eligible_parent_example(self, mocker):
        # Two eligible sentences sharing one parent example must yield one case, not two — the
        # comparative pipeline compares whole answers, not isolated sentences (see
        # CASE-SELECTION-PROTOCOL.md, "candidate pool").
        multi_sentence_record = RAGBenchEvaluationRecord(
            example_id="techqa_multi",
            domain="techqa",
            question="Q",
            response="A. B.",
            chunks=[RetrievedChunk(chunk_id="d0", text="Evidence.", score=0.9)],
            response_sentences=[BenchmarkSentence(key="a", text="A."), BenchmarkSentence(key="b", text="B.")],
            document_sentences=[],
            document_sentence_keys={"d0"},
            unsupported_response_sentence_keys=set(),
            sentence_support={
                "a": BenchmarkSentenceSupport(response_sentence_key="a", fully_supported=True, supporting_sentence_keys=["d0"]),
                "b": BenchmarkSentenceSupport(response_sentence_key="b", fully_supported=True, supporting_sentence_keys=["d0"]),
            },
        )
        ineligible_record = RAGBenchEvaluationRecord(
            example_id="techqa_none",
            domain="techqa",
            question="Q2",
            response="C.",
            chunks=[RetrievedChunk(chunk_id="d0", text="Evidence.", score=0.9)],
            response_sentences=[BenchmarkSentence(key="c", text="C.")],
            document_sentences=[],
            document_sentence_keys={"d0"},
            unsupported_response_sentence_keys={"c"},
            sentence_support={
                "c": BenchmarkSentenceSupport(response_sentence_key="c", fully_supported=False, supporting_sentence_keys=[]),
            },
        )
        mocker.patch(
            "benchmark.comparative_diagnostics._load_records",
            return_value=([multi_sentence_record, ineligible_record], []),
        )
        pool = load_case_candidate_pool()
        assert [record.example_id for record in pool] == ["techqa_multi"]


class TestMapRagvueNative:
    def test_all_metrics_erroring_is_unavailable_not_a_healthy_zero(self):
        # A real RAGVue/anthropic-SDK version break returns score 0.0 with an embedded "error" per
        # metric instead of raising. Presenting that as a healthy 0.0 is exactly the anti-pattern
        # this project's own README argues against ("explicit evaluator failures instead of
        # healthy-looking zeroes").
        raw = {
            "metrics": [
                {"name": "strict_faithfulness", "score": 0.0, "details": {"error": "LLM error: boom"}},
                {"name": "retrieval_relevance", "score": 0.0, "details": {"error": "LLM error: boom"}},
            ]
        }
        record = map_ragvue_native(case_id="c1", raw_result=raw, configured_model="anthropic:claude-haiku-4-5-20251001")
        assert record.native.availability == "unavailable"
        assert record.supporting_observation is None
        assert any("strict_faithfulness" in note and "boom" in note for note in record.no_equivalent_fields)

    def test_partial_metric_errors_report_only_the_healthy_ones(self):
        raw = {
            "metrics": [
                {"name": "context_similarity", "score": 0.64, "details": {"method": "tf_python"}},
                {"name": "retrieval_relevance", "score": 0.0, "details": {"error": "LLM error: boom"}},
            ]
        }
        record = map_ragvue_native(case_id="c1", raw_result=raw, configured_model="anthropic:claude-haiku-4-5-20251001")
        assert record.native.availability == "unavailable"
        assert record.supporting_observation == "context_similarity=0.64"
        assert any("retrieval_relevance" in note for note in record.no_equivalent_fields)

    def test_maps_metric_and_records_configured_model_not_self_reported_one(self):
        raw = {
            "metrics": [
                {"name": "retrieval_relevance", "score": 0.2, "details": {"raw": {"model": "gpt-4o-mini"}}}
            ]
        }
        record = map_ragvue_native(
            case_id="techqa_DEV_Q243",
            raw_result=raw,
            configured_model="anthropic:claude-haiku-4-5-20251001",
        )
        assert record.native.raw_output == raw  # RAGVue's self-reported "gpt-4o-mini" is preserved untouched
        assert record.method == "anthropic:claude-haiku-4-5-20251001"
        assert "self-reported" in " ".join(record.no_equivalent_fields).lower()
