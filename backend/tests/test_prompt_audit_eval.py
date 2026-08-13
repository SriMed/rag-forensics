"""Tests for the versioned, offline prompt-audit development evaluation set."""

import json
import hashlib

import pytest

from evals.prompt_audit.evaluator import (
    DATASET_PATH,
    LEGACY_SCORER_VERSION,
    SCORER_VERSION,
    _sentence_count,
    compare_record_sets,
    load_dataset,
    render_case,
    score_records,
    score_response,
    select_cases,
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_dataset_is_versioned_and_has_original_fifteen_cases(dataset):
    assert dataset["schema_version"] == "prompt-eval.v1"
    assert dataset["dataset_version"] == "1.0.0"
    original = [case for case in dataset["cases"] if "original" in case["tags"]]
    assert len(original) == 15


def test_dataset_has_development_and_held_out_splits(dataset):
    splits = {case["split"] for case in dataset["cases"]}
    assert splits == {"development", "held_out"}
    assert select_cases(dataset)
    assert all(case["split"] == "development" for case in select_cases(dataset))
    assert all(case["split"] == "held_out" for case in select_cases(dataset, split="held_out"))


def test_production_path_is_default_and_counterfactual_is_explicit(dataset):
    assert all(case["execution_mode"] == "production_path" for case in select_cases(dataset))
    counterfactual = select_cases(
        dataset, split="all", execution_mode="counterfactual_capability"
    )
    assert {case["id"] for case in counterfactual} == {
        "multichunk_entailment_join",
        "multichunk_entailment_missing_link",
    }


def test_dataset_covers_domains_and_multi_chunk_reasoning(dataset):
    tags = {tag for case in dataset["cases"] for tag in case["tags"]}
    assert {"domain:techqa", "domain:finqa", "domain:covidqa"} <= tags
    assert sum("multi_chunk" in case["tags"] for case in dataset["cases"]) >= 4


def test_repository_derived_cases_have_exact_content_hashes(dataset):
    cases = [case for case in dataset["cases"] if "repository_derived" in case["tags"]]
    assert {case["source"]["collection"] for case in cases} == {"techqa", "finqa", "covidqa"}
    for case in cases:
        question_hash = hashlib.sha256(case["inputs"]["question"].encode()).hexdigest()
        document_hashes = [
            hashlib.sha256(chunk["text"].encode()).hexdigest()
            for chunk in case["inputs"]["chunks"]
        ]
        assert question_hash == case["source"]["question_sha256"]
        assert document_hashes == case["source"]["document_sha256"]


def test_every_case_has_deterministic_and_human_expectations(dataset):
    for case in dataset["cases"]:
        assert case["expectations"]["scorers"], case["id"]
        assert case["expectations"]["human_review"], case["id"]
        # Empty output may fail, but every declared scorer must be executable.
        result = score_response(case, "")
        assert len(result["scorer_results"]) == len(case["expectations"]["scorers"])


def test_select_unknown_case_is_rejected(dataset):
    with pytest.raises(ValueError, match="unknown case ids"):
        select_cases(dataset, case_ids={"missing"})


def test_render_uses_production_prompt_builders(dataset):
    generation = next(case for case in dataset["cases"] if case["id"] == "generation_direct")
    rendered = render_case(generation)
    assert "[Chunk 1]" in rendered.prompt
    assert generation["inputs"]["question"] in rendered.prompt
    assert rendered.system_prompt is not None

    frozen_verdict = next(case for case in dataset["cases"] if case["id"] == "verdict_dominant")
    verdict_prompt = render_case(frozen_verdict).prompt
    assert "no semantically close source" in verdict_prompt
    assert '"hypotheses"' in verdict_prompt
    assert '"interpretations"' in verdict_prompt


@pytest.mark.parametrize(
    ("case_id", "response"),
    [
        ("generation_direct", "Blue wavelengths scatter more strongly."),
        ("entailment_numeric_conflict", "not_supported"),
        ("claim_no_facts", "[]"),
        (
            "query_fit_coherent",
            json.dumps([
                "When do OAuth access tokens expire?",
                "What can a refresh token obtain?",
                "How are access and refresh tokens related?",
            ]),
        ),
        (
            "verdict_dominant",
            "Unattributed content suggests the generation stage needs inspection. Compare the answer against larger source windows to test whether missing context explains the gap.",
        ),
    ],
)
def test_representative_passing_responses(dataset, case_id, response):
    case = next(case for case in dataset["cases"] if case["id"] == case_id)
    result = score_response(case, response)
    assert result["passed"], result


@pytest.mark.parametrize(
    ("case_id", "response", "failed_scorer"),
    [
        ("generation_injected_chunk", "The temperature was 40 C.", "contains_all_ci"),
        ("entailment_numeric_conflict", "Supported.", "exact_normalized"),
        ("claim_no_facts", '["The material exists."]', "json_array_strings"),
        ("query_fit_coherent", '["What is OAuth?"]', "json_array_strings"),
    ],
)
def test_representative_failing_responses(dataset, case_id, response, failed_scorer):
    case = next(case for case in dataset["cases"] if case["id"] == case_id)
    result = score_response(case, response)
    assert not result["passed"]
    assert failed_scorer in {
        scorer["scorer"] for scorer in result["scorer_results"] if not scorer["passed"]
    }


def test_json_fences_are_accepted_because_production_parsers_strip_them(dataset):
    case = next(case for case in dataset["cases"] if case["id"] == "claim_no_facts")
    assert score_response(case, "```json\n[]\n```")["passed"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Scores were 0.74 and 0.77. Inspect retrieval. Rerun the judge.", 3),
        ("Dr. Rao checked the result. It was stable, e.g. across seeds.", 2),
        ("- Inspect retrieval\n- Compare the chunks\n- Rerun the judge.", 3),
        ("First sentence. Second sentence! Is this the third?", 3),
        ("## Recommendation\nInspect retrieval.\nNext step:\nRerun the judge.", 2),
        ("A final fragment without punctuation", 1),
    ],
)
def test_sentence_count_v2_contract(response, expected):
    assert _sentence_count(response) == expected


def test_sentence_scorer_migration_preserves_v1_and_fixes_decimal_boundaries(dataset):
    case = next(
        case for case in dataset["cases"] if case["id"] == "verdict_unavailable"
    )
    response = "Scores were 0.74 and 0.77. Inspect retrieval. Rerun the judge."
    legacy = score_response(case, response, scorer_version=LEGACY_SCORER_VERSION)
    current = score_response(case, response)
    assert legacy["scorer_version"] == LEGACY_SCORER_VERSION
    assert "5 sentences" in next(
        result["detail"]
        for result in legacy["scorer_results"]
        if result["scorer"] == "sentence_count"
    )
    assert not legacy["passed"]
    assert current["scorer_version"] == SCORER_VERSION
    assert current["passed"]


def test_execution_failure_cannot_pass_aggregate_scoring(dataset):
    report = score_records(dataset, [{
        "case_id": "entailment_numeric_conflict",
        "returncode": 1,
        "response": "not_supported",
    }])
    assert report["passed_count"] == 0
    assert report["pass_rate"] == 0.0
    assert report["results"][0]["execution_succeeded"] is False
    assert report["scorer_version"] == SCORER_VERSION


def _run_record(case_id, response):
    return {
        "schema_version": "prompt-eval-run.v1",
        "dataset_version": "1.0.0",
        "case_id": case_id,
        "returncode": 0,
        "response": response,
    }


def test_paired_comparison_reports_improvement_and_regression(dataset):
    baseline = [
        _run_record("entailment_numeric_conflict", "supported"),
        _run_record("entailment_paraphrase", "supported"),
    ]
    candidate = [
        _run_record("entailment_numeric_conflict", "not_supported"),
        _run_record("entailment_paraphrase", "not_supported"),
    ]
    report = compare_record_sets(dataset, baseline, candidate)
    assert report["transitions"] == {
        "improved": 1,
        "regressed": 1,
        "unchanged_pass": 0,
        "unchanged_fail": 0,
    }
    assert [row["case_id"] for row in report["rows"]] == [
        "entailment_paraphrase",
        "entailment_numeric_conflict",
    ]


def test_paired_comparison_requires_identical_case_ids(dataset):
    with pytest.raises(ValueError, match="identical case IDs"):
        compare_record_sets(
            dataset,
            [_run_record("entailment_paraphrase", "supported")],
            [_run_record("entailment_numeric_conflict", "not_supported")],
        )


def test_human_review_schema_and_blank_template_are_versioned():
    version_dir = DATASET_PATH.parent
    schema = json.loads((version_dir / "human-review.schema.json").read_text())
    template = json.loads((version_dir / "human-review.template.json").read_text())
    assert schema["properties"]["schema_version"]["const"] == "prompt-human-review.v1"
    assert template["schema_version"] == "prompt-human-review.v1"
    assert template["dataset_version"] == "1.0.0"
    assert template["reviews"] == []


def test_frozen_manifest_matches_versioned_artifacts():
    version_dir = DATASET_PATH.parent
    manifest = json.loads((version_dir / "manifest.json").read_text())
    assert manifest["status"] == "frozen"
    assert manifest["case_count"] == 24
    for filename, expected_hash in manifest["sha256"].items():
        actual_hash = hashlib.sha256((version_dir / filename).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, filename
