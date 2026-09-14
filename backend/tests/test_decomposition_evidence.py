import numpy as np
import pytest

from benchmark.decomposition_evidence import (
    ClaimReviewArtifact,
    ClaimReviewItem,
    make_claim_review_template,
    make_residual_review_template,
    run_decomposition_evidence_experiment,
)
from benchmark.grounding import DeterministicClaimDecomposer, FixtureEntailmentVerifier
from benchmark.decomposition_evidence_cli import build_parser
from benchmark.ragbench import adapt_ragbench_row


class TextEmbedding:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if text == "Revenue rose.":
                vectors.append([1.0, 0.0])
            elif text == "Revenue rose according to the report.":
                vectors.append([0.0, 1.0])
            elif text == "Unrelated material.":
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors)


def _record():
    return adapt_ragbench_row(
        {
            "id": "example-1",
            "question": "What happened?",
            "documents": ["Revenue rose. Unrelated material."],
            "documents_sentences": [[
                ["0a", "Revenue rose."], ["0b", "Unrelated material."]
            ]],
            "response": "Revenue rose according to the report.",
            "response_sentences": [["a", "Revenue rose according to the report."]],
            "sentence_support_information": [{
                "response_sentence_key": "a", "fully_supported": True,
                "supporting_sentence_keys": ["0a"],
            }],
            "unsupported_response_sentence_keys": [],
        },
        domain="finqa",
    )


def _review(status="frozen"):
    return ClaimReviewArtifact(
        schema_version="decomposition-evidence-claims.v1",
        status=status,
        reviewer_identity="reviewer@example.com" if status == "frozen" else None,
        reviewer_instructions="Review claims.",
        population_sha256=make_claim_review_template(
            [_record()], DeterministicClaimDecomposer()
        ).population_sha256,
        decomposer_name="deterministic_clause",
        decomposer_version="1",
        items=[ClaimReviewItem(
            domain="finqa", example_id="example-1", sentence_key="a",
            question="What happened?",
            full_response="Revenue rose according to the report.",
            sentence="Revenue rose according to the report.",
            deterministic_claims=["Revenue rose according to the report."],
            accept_deterministic=False if status == "frozen" else None,
            reviewed_claims=["Revenue rose."] if status == "frozen" else [],
        )],
    )


def test_experiment_builds_paired_conditions_and_interaction():
    verifier = FixtureEntailmentVerifier({
        ("Revenue rose according to the report.", "0b"): 0.1,
        ("Revenue rose according to the report.", "0a"): 0.9,
        ("Revenue rose.", "0a"): 0.9,
    })
    report = run_decomposition_evidence_experiment(
        [_record()], TextEmbedding(), DeterministicClaimDecomposer(), verifier,
        0.5, _review(), "b" * 64, bootstrap_iterations=20, seed=7,
    )

    assert report.population_size == 1
    assert {name: value.false_unsupported_rate for name, value in report.conditions.items()} == {
        "A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0,
    }
    assert report.effects["interaction"].interval.point_estimate == pytest.approx(1.0)
    assert report.residual_condition_d_keys == []


def test_experiment_rejects_unfrozen_or_mismatched_review():
    with pytest.raises(ValueError, match="must be frozen"):
        run_decomposition_evidence_experiment(
            [_record()], TextEmbedding(), DeterministicClaimDecomposer(),
            FixtureEntailmentVerifier({}), 0.5, _review("draft"), "b" * 64,
        )
    review = _review()
    review.items[0].sentence_key = "wrong"
    with pytest.raises(ValueError, match="does not exactly match"):
        run_decomposition_evidence_experiment(
            [_record()], TextEmbedding(), DeterministicClaimDecomposer(),
            FixtureEntailmentVerifier({}), 0.5, review, "b" * 64,
        )


def test_claim_review_cli_has_no_oracle_or_evidence_inputs():
    parser = build_parser()
    args = parser.parse_args(["prepare-claims", "--output", "claims.json"])
    assert not hasattr(args, "oracle_report")
    assert not hasattr(args, "include_dataset_context")


def test_claim_template_preserves_review_context_without_prefilling_judgment():
    artifact = make_claim_review_template([_record()], DeterministicClaimDecomposer())
    item = artifact.items[0]
    assert artifact.status == "draft"
    assert item.question == "What happened?"
    assert item.full_response == "Revenue rose according to the report."
    fields = set(type(item).model_fields)
    assert not any("evidence" in field or "verifier" in field for field in fields)
    assert item.reviewed_claims == []


def test_experiment_reports_exclusions_from_ineligible_sentences():
    ineligible_record = adapt_ragbench_row(
        {
            "id": "example-2",
            "question": "What else?",
            "documents": ["Revenue rose. Unrelated material."],
            "documents_sentences": [[
                ["0a", "Revenue rose."], ["0b", "Unrelated material."]
            ]],
            "response": "Costs fell. Margins widened.",
            "response_sentences": [["b", "Costs fell."], ["c", "Margins widened."]],
            "sentence_support_information": [
                {"response_sentence_key": "b", "fully_supported": True, "supporting_sentence_keys": []},
                {"response_sentence_key": "c", "fully_supported": True, "supporting_sentence_keys": ["well_known_fact"]},
            ],
            "unsupported_response_sentence_keys": [],
        },
        domain="finqa",
    )
    verifier = FixtureEntailmentVerifier({
        ("Revenue rose according to the report.", "0b"): 0.1,
        ("Revenue rose according to the report.", "0a"): 0.9,
        ("Revenue rose.", "0a"): 0.9,
    })
    report = run_decomposition_evidence_experiment(
        [_record(), ineligible_record], TextEmbedding(), DeterministicClaimDecomposer(), verifier,
        0.5, _review(), "b" * 64, bootstrap_iterations=20, seed=7,
    )

    assert report.population_size == 1
    assert report.exclusions == {"missing_annotation": 1, "non_document_support": 1}


def test_experiment_marks_unavailable_verifier_states_and_excludes_from_rate():
    verifier = FixtureEntailmentVerifier({})  # every lookup raises KeyError -> verifier_error
    report = run_decomposition_evidence_experiment(
        [_record()], TextEmbedding(), DeterministicClaimDecomposer(), verifier,
        0.5, _review(), "b" * 64, bootstrap_iterations=20, seed=7,
    )

    condition_a = report.conditions["A"]
    assert condition_a.evaluated == 0
    assert condition_a.false_unsupported_rate is None
    assert condition_a.predictions[0].predicted_unsupported is None
    assert condition_a.predictions[0].claims[0].status == "verifier_error"
    assert report.effects["evidence_at_deterministic"].interval is None


def test_residual_review_is_separate_and_contains_post_run_evidence():
    verifier = FixtureEntailmentVerifier({
        ("Revenue rose according to the report.", "0b"): 0.1,
        ("Revenue rose according to the report.", "0a"): 0.1,
        ("Revenue rose.", "0a"): 0.1,
    })
    report = run_decomposition_evidence_experiment(
        [_record()], TextEmbedding(), DeterministicClaimDecomposer(), verifier,
        0.5, _review(), "b" * 64, bootstrap_iterations=20, seed=7,
    )
    residual = make_residual_review_template(
        report, "c" * 64, [_record()], _review()
    )

    assert len(residual.items) == 1
    assert residual.items[0].annotated_evidence == ["Revenue rose."]
    assert residual.items[0].verifier_outcomes[0].support_score == pytest.approx(0.1)
