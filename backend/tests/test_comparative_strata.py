"""Tests for the frozen issue #30 stratum-eligibility and selection rules (synthetic inputs only)."""

from types import SimpleNamespace

import pytest

from benchmark import comparative_strata as cs


def _native(system, availability="healthy", raw=None, error=None):
    return {"system": system, "native": {"availability": availability, "raw_output": raw, "error": error}}


def _rf(top="noisy_context", component="retriever", signals=(), faith=1.0, util=1.0, faith_status="ok"):
    names = [top, *signals]
    raw = {
        "verdict_signals": [{"name": n} for n in names],
        "verdict_reasoning": {"test": {"component": component}},
        "ragas": {
            "faithfulness": {"score": faith if faith_status == "ok" else None, "status": faith_status},
            "context_utilization": {"score": util, "status": "ok"},
        },
    }
    return _native("rag_forensics", raw=raw)


def _rc(faithfulness=1.0, availability="healthy", error=None):
    raw = {"faithfulness": faithfulness} if availability == "healthy" else None
    return _native("ragchecker", availability, raw, error)


def _rv(strict=1.0, relevance=1.0, availability="healthy", error=None):
    raw = {"metrics": [
        {"name": "strict_faithfulness", "score": strict, "details": {}},
        {"name": "retrieval_relevance", "score": relevance, "details": {}},
    ]}
    return _native("ragvue", availability, raw, error)


def _entry(rf=None, rc=None, rv=None):
    return {"rag_forensics": rf or _rf(), "ragchecker": rc or _rc(), "ragvue": rv or _rv()}


def _record(domain="covidqa", response="It spreads by droplets.", unsupported=False):
    return SimpleNamespace(
        domain=domain, response=response,
        unsupported_response_sentence_keys=["a"] if unsupported else [],
    )


def _pred(selected_unsupported, oracle_unsupported):
    return SimpleNamespace(
        selected=SimpleNamespace(predicted_unsupported=selected_unsupported),
        oracle=SimpleNamespace(predicted_unsupported=oracle_unsupported),
    )


class TestUnsupportedFlags:
    def test_any_claim_unsupported_rule_per_system(self):
        flags = cs.unsupported_flags(_entry(
            rf=_rf(top="low_faithfulness", faith=0.75), rc=_rc(1.0), rv=_rv(strict=0.5),
        ))
        assert flags == {"rag_forensics": True, "ragas_baseline": True, "ragchecker": False, "ragvue": True}

    @pytest.mark.parametrize("top", ["low_faithfulness", "unattributed_content", "overconfidence"])
    def test_rag_forensics_flag_signals(self, top):
        assert cs.unsupported_flags(_entry(rf=_rf(top=top)))["rag_forensics"] is True

    def test_unhealthy_or_missing_scores_have_no_flag(self):
        flags = cs.unsupported_flags(_entry(
            rf=_rf(faith_status="unavailable"), rc=_rc(availability="failed", error="boom"),
            rv=_native("ragvue", "unavailable", {"metrics": [
                {"name": "strict_faithfulness", "score": 0.0, "details": {"error": "x"}},
            ]}),
        ))
        assert flags["ragas_baseline"] is None
        assert flags["ragchecker"] is None
        assert flags["ragvue"] is None


class TestComponentDirections:
    def test_rag_forensics_component_mapping(self):
        assert cs.rag_forensics_direction(_entry(rf=_rf(component="retriever"))) == "retrieval"
        assert cs.rag_forensics_direction(_entry(rf=_rf(component="answer generator"))) == "generation"
        assert cs.rag_forensics_direction(_entry(rf=_rf(component="retrieval-to-generation boundary"))) is None

    def test_ragvue_direction(self):
        assert cs.ragvue_direction(_entry(rv=_rv(strict=0.8, relevance=0.1))) == "generation"
        assert cs.ragvue_direction(_entry(rv=_rv(strict=1.0, relevance=0.4))) == "retrieval"
        assert cs.ragvue_direction(_entry(rv=_rv(strict=1.0, relevance=0.5))) is None


class TestFailureExposure:
    def test_native_failures_and_rag_forensics_unavailable_signals(self):
        exposed = cs.failure_exposures(_entry(
            rf=_rf(signals=["hedging_analysis_unavailable"]), rc=_rc(availability="failed", error="e"),
        ))
        assert exposed == {"rag_forensics", "ragchecker"}

    def test_passed_through_ragas_failures_count_once_as_ragas(self):
        exposed = cs.failure_exposures(_entry(
            rf=_rf(signals=["faithfulness_unavailable"], faith_status="unavailable"),
        ))
        assert exposed == {"ragas_baseline"}


class TestInterventionOutcomes:
    def test_discriminates_and_fails(self):
        assert cs.intervention_outcomes([_pred(True, False)]) == {"intervention_discriminates_hypotheses"}
        assert cs.intervention_outcomes([_pred(True, True)]) == {"intervention_fails_to_localize"}
        assert cs.intervention_outcomes([_pred(False, False), _pred(None, True)]) == set()


class TestEligibleStrata:
    def test_agreement_with_and_against_label(self):
        agree_flagged = _entry(rf=_rf(top="low_faithfulness", faith=0.5), rc=_rc(0.5), rv=_rv(strict=0.5))
        assert "systems_agree_labels_support" in cs.eligible_strata(agree_flagged, _record(unsupported=True), [])
        assert "systems_agree_labels_contradict" in cs.eligible_strata(agree_flagged, _record(), [])

    def test_component_disagreement(self):
        entry = _entry(rf=_rf(component="retriever"), rv=_rv(strict=0.5))
        assert "component_diagnoses_disagree" in cs.eligible_strata(entry, _record(), [])

    def test_evidence_attribution_stratum_is_never_eligible(self):
        assert "evidence_attributions_disagree" not in cs.eligible_strata(_entry(), _record(), [])

    def test_single_system_failure(self):
        entry = _entry(rc=_rc(availability="failed", error="e"))
        assert "single_system_exposes_failure" in cs.eligible_strata(entry, _record(), [])

    @pytest.mark.parametrize(("domain", "response", "expected"), [
        ("finqa", "Revenue rose.", True),
        ("covidqa", "It is not airborne.", True),
        ("covidqa", "Only adults were tested.", True),
        ("covidqa", "Nothing notable was found.", False),
    ])
    def test_qualifier_lexical_rule(self, domain, response, expected):
        strata = cs.eligible_strata(_entry(), _record(domain=domain, response=response), [])
        assert ("qualifier_negation_numerical_tabular_multisource_or_granularity" in strata) is expected

    def test_counterexample_when_rag_forensics_contradicts_label_regardless_of_others(self):
        # v1.1: v1 also required every other system to agree with the label, which no pool case met.
        stratum = "counterexample_to_preferred_interpretation"
        assert stratum in cs.eligible_strata(_entry(rf=_rf(top="overconfidence")), _record(), [])
        assert stratum in cs.eligible_strata(_entry(rf=_rf(top="overconfidence"), rc=_rc(0.5)), _record(), [])
        assert stratum not in cs.eligible_strata(_entry(rf=_rf(top="overconfidence")), _record(unsupported=True), [])
        assert stratum in cs.eligible_strata(_entry(), _record(unsupported=True), [])

    def test_rules_version_is_recorded(self):
        assert cs.RULES_VERSION == "v1.1"


class TestExclusion:
    def test_batch_crash_or_rag_forensics_failure_excludes(self):
        crashed = _entry(rc=_rc(availability="failed", error="runner batch failed: X"))
        assert cs.exclusion_reason(crashed) is not None
        assert cs.exclusion_reason(_entry(rf=_native("rag_forensics", "failed", None, "E"))) is not None
        assert cs.exclusion_reason(_entry()) is None


class TestSelection:
    def test_seeded_draw_respects_order_targets_and_one_stratum_per_case(self):
        eligibility = {f"c{i}": {"single_system_exposes_failure", "systems_agree_labels_support"} for i in range(6)}
        selected = cs.select_cases(eligibility, seed=cs.SELECTION_SEED)
        assert selected == cs.select_cases(eligibility, seed=cs.SELECTION_SEED)
        by_stratum: dict[str, list[str]] = {}
        for case_id, stratum in selected:
            by_stratum.setdefault(stratum, []).append(case_id)
        assert len(by_stratum["single_system_exposes_failure"]) == cs.STRATUM_TARGETS["single_system_exposes_failure"]
        assert len(by_stratum["systems_agree_labels_support"]) == min(
            6 - cs.STRATUM_TARGETS["single_system_exposes_failure"], cs.STRATUM_TARGETS["systems_agree_labels_support"],
        )
        assert len({c for c, _ in selected}) == len(selected)

    def test_targets_cover_every_declared_stratum_and_sum_within_range(self):
        from benchmark.comparative_diagnostics import DECLARED_STRATA

        assert set(cs.SELECTION_ORDER) == set(DECLARED_STRATA)
        assert 12 <= sum(cs.STRATUM_TARGETS.values()) <= 20
        assert cs.STRATUM_TARGETS["evidence_attributions_disagree"] == 0


class TestBuildCaseSet:
    def _inputs(self):
        def native(system, raw):
            return {
                "system": system,
                "native": {"system": system, "system_version": "t", "availability": "healthy", "raw_output": raw, "error": None},
                "suspected_component": None, "supporting_observation": None, "evidence_attribution": [],
                "method": None, "reliability": None, "causal_strength_language": None,
                "proposed_intervention": None, "no_equivalent_fields": [],
            }

        def entry(rc_faith=1.0):
            return {
                "rag_forensics": native("rag_forensics", _rf()["native"]["raw_output"]),
                "ragchecker": native("ragchecker", {"faithfulness": rc_faith}),
                "ragvue": native("ragvue", _rv()["native"]["raw_output"]),
            }

        crashed = entry()
        crashed["ragvue"]["native"] = {
            "system": "ragvue", "system_version": "unknown", "availability": "failed",
            "raw_output": None, "error": "runner batch failed: X",
        }
        pool = {"a": entry(), "b": entry(), "c": crashed}
        records = {
            cid: SimpleNamespace(example_id=cid, domain="covidqa", response="It spreads.",
                                 unsupported_response_sentence_keys=[])
            for cid in pool
        }
        predictions = {"a": [_pred(True, False)], "b": [], "c": []}
        return pool, records, predictions

    def test_builds_draft_with_selection_record_and_ragas_baseline(self):
        pool, records, predictions = self._inputs()
        case_set = cs.build_case_set(pool, records, predictions, rules="STRATUM-RULES.md@abc")
        assert case_set.selection.rules.endswith("(rules v1.1)")

        assert case_set.status == "draft"
        sel = case_set.selection
        assert sel is not None
        assert sel.pool_size == 3
        assert sel.exclusions == {"c": "ragvue batch crashed and was not rerun"}
        assert sel.eligible_counts["intervention_discriminates_hypotheses"] == 1
        assert sel.eligible_counts["systems_agree_labels_support"] == 2
        by_id = {c.case_id: c for c in case_set.cases}
        assert by_id["a"].stratum == "intervention_discriminates_hypotheses"
        assert by_id["b"].stratum == "systems_agree_labels_support"
        assert set(by_id["a"].systems) == {"rag_forensics", "ragas_baseline", "ragchecker", "ragvue"}
        assert by_id["a"].judgments.dataset_label == "fully_supported"
        assert by_id["a"].judgments.intervention_evidence is not None
        assert "seed 30" in by_id["a"].selection_rationale
