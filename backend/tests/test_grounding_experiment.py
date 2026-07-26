import numpy as np
import pytest

from benchmark.grounding import (
    DeterministicClaimDecomposer,
    FixtureClaimDecomposer,
    FixtureEntailmentVerifier,
    aggregate_claims,
    bootstrap_paired_difference,
    calculate_binary_metrics,
    calibrate_threshold,
    normalize_nli_scores,
    summarize_method,
    run_grounding_methods,
)
from benchmark.ragbench import adapt_ragbench_row
from benchmark.experiment import categorize_error, run_experiment
from benchmark.experiment_cli import build_parser as build_experiment_parser
from models import GroundingRunMetadata, GroundingSentencePrediction


def _row(**overrides):
    row = {
        "id": "example-1",
        "question": "What happened?",
        "documents": ["Revenue rose to $20 million. Costs remained flat."],
        "documents_sentences": [
            [
                ["0a", "Revenue rose to $20 million."],
                ["0b", "Costs remained flat."],
            ]
        ],
        "response": "Revenue rose to $20 million, but costs doubled.",
        "response_sentences": [
            ["a", "Revenue rose to $20 million, but costs doubled."]
        ],
        "sentence_support_information": [
            {
                "response_sentence_key": "a",
                "fully_supported": False,
                "supporting_sentence_keys": ["0a"],
                "explanation": "The revenue claim is supported but the cost claim is not.",
            }
        ],
        "unsupported_response_sentence_keys": ["a"],
        "adherence_score": False,
        "relevance_score": 1.0,
        "utilization_score": 0.5,
        "completeness_score": 1.0,
    }
    row.update(overrides)
    return row


def test_adapter_preserves_document_sentence_provenance():
    record = adapt_ragbench_row(_row(), domain="finqa")

    assert [(item.key, item.document_id) for item in record.document_sentences] == [
        ("0a", "document_0"),
        ("0b", "document_0"),
    ]


def test_deterministic_decomposer_keeps_parent_and_splits_contrast():
    claims = DeterministicClaimDecomposer().decompose(
        "a",
        "Revenue rose, but costs doubled.",
    )

    assert [claim.parent_sentence_key for claim in claims] == ["a", "a"]
    assert [claim.text for claim in claims] == ["Revenue rose", "costs doubled."]
    assert [claim.claim_id for claim in claims] == ["a.claim-0", "a.claim-1"]


def test_fixture_decomposer_is_exact_and_offline():
    claims = FixtureClaimDecomposer({"a": ["First fact.", "Second fact."]}).decompose(
        "a", "Ignored fixture sentence."
    )
    assert [claim.text for claim in claims] == ["First fact.", "Second fact."]


@pytest.mark.parametrize(
    ("sentence", "expected_count"),
    [
        ("Revenue rose to $20 million.", 1),
        ("Costs did not fall; revenue may increase.", 2),
        ("", 0),
    ],
)
def test_decomposer_handles_numeric_negation_qualifier_and_empty(sentence, expected_count):
    assert (
        len(DeterministicClaimDecomposer().decompose("a", sentence))
        == expected_count
    )


def test_normalize_nli_scores_uses_model_label_mapping():
    scores = normalize_nli_scores(
        logits=[-1.0, 3.0, 0.0],
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
    )

    assert scores["entailment"] > scores["neutral"] > scores["contradiction"]
    assert sum(scores.values()) == pytest.approx(1.0)


def test_normalize_nli_scores_rejects_unknown_label_order():
    with pytest.raises(ValueError, match="entailment, neutral, and contradiction"):
        normalize_nli_scores([1.0, 2.0], {0: "LABEL_0", 1: "LABEL_1"})


def test_any_unsupported_claim_makes_parent_sentence_unsupported():
    assert aggregate_claims([True, False]) is False
    assert aggregate_claims([True, True]) is True
    assert aggregate_claims([True, None]) is None
    assert aggregate_claims([]) is None


def test_metrics_include_prevalence_auroc_auprc_and_coverage():
    metrics = calculate_binary_metrics(
        gold_unsupported=[False, True, True, False],
        predicted_unsupported=[False, True, False, True],
        unsupported_scores=[0.1, 0.9, 0.6, 0.2],
        total=5,
    )

    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.prevalence == pytest.approx(0.5)
    assert metrics.coverage == pytest.approx(0.8)
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.auprc == pytest.approx(1.0)


def test_metrics_make_single_class_rank_metrics_explicitly_unavailable():
    metrics = calculate_binary_metrics(
        [False, False],
        [False, False],
        [0.1, 0.2],
    )
    assert metrics.auroc is None
    assert metrics.auprc is None


def test_calibration_uses_only_supplied_partition_and_is_deterministic():
    result = calibrate_threshold(
        support_scores=[0.9, 0.8, 0.3, 0.1],
        gold_unsupported=[False, False, True, True],
        candidates=[0.2, 0.4, 0.85],
    )

    assert result.threshold == pytest.approx(0.4)
    assert result.partition == "calibration"
    assert result.objective == "f1"


def test_paired_bootstrap_is_reproducible_and_resamples_examples():
    baseline = {"a": 0.0, "b": 0.2, "c": 0.4}
    candidate = {"a": 0.5, "b": 0.6, "c": 0.8}

    first = bootstrap_paired_difference(
        baseline, candidate, iterations=200, seed=7
    )
    second = bootstrap_paired_difference(
        baseline, candidate, iterations=200, seed=7
    )

    assert first == second
    assert first.point_estimate == pytest.approx(
        np.mean(list(candidate.values())) - np.mean(list(baseline.values()))
    )
    assert first.lower > 0


def test_all_methods_share_interface_and_preserve_claim_evidence(mocker):
    record = adapt_ragbench_row(_row(), domain="finqa")
    embedding_model = mocker.MagicMock()
    embedding_model.encode.side_effect = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # document sentences
        np.array([[1.0, 0.0]]),  # response sentence
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # claims
    ]
    verifier = FixtureEntailmentVerifier(
        {
            ("Revenue rose to $20 million", "0a"): 0.95,
            ("costs doubled.", "0b"): {
                "entailment": 0.05,
                "neutral": 0.05,
                "contradiction": 0.9,
            },
        }
    )

    results = run_grounding_methods(
        [record],
        embedding_model=embedding_model,
        decomposer=DeterministicClaimDecomposer(),
        entailment_verifier=verifier,
        similarity_threshold=0.4,
        entailment_threshold=0.5,
    )

    assert set(results) == {
        "b0_always_supported",
        "b0_always_unsupported",
        "b1_sentence_similarity",
        "b2_claim_similarity",
        "b3_claim_entailment",
    }
    b3 = results["b3_claim_entailment"][0]
    assert b3.predicted_unsupported is True
    assert [claim.parent_sentence_key for claim in b3.claims] == ["a", "a"]
    assert [claim.evidence.sentence_key for claim in b3.claims] == ["0a", "0b"]
    assert b3.claims[1].verifier_label == "contradiction"
    assert b3.claims[1].nli_scores.contradiction == pytest.approx(0.9)


def test_verifier_failure_is_unevaluated_not_unsupported(mocker):
    record = adapt_ragbench_row(_row(), domain="finqa")
    embedding_model = mocker.MagicMock()
    embedding_model.encode.side_effect = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    verifier = mocker.MagicMock()
    verifier.score.side_effect = RuntimeError("model failed")

    results = run_grounding_methods(
        [record],
        embedding_model=embedding_model,
        decomposer=DeterministicClaimDecomposer(),
        entailment_verifier=verifier,
        similarity_threshold=0.4,
        entailment_threshold=0.5,
    )

    assert results["b3_claim_entailment"][0].predicted_unsupported is None
    assert all(
        claim.status == "verifier_error"
        for claim in results["b3_claim_entailment"][0].claims
    )


def test_method_summary_reports_domains_pooled_macro_and_intervals(mocker):
    record = adapt_ragbench_row(_row(), domain="finqa")
    embedding_model = mocker.MagicMock()
    embedding_model.encode.side_effect = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    predictions = run_grounding_methods(
        [record],
        embedding_model=embedding_model,
        decomposer=DeterministicClaimDecomposer(),
        entailment_verifier=FixtureEntailmentVerifier(
            {
                ("Revenue rose to $20 million", "0a"): 0.95,
                ("costs doubled.", "0b"): 0.05,
            }
        ),
        similarity_threshold=0.4,
        entailment_threshold=0.5,
    )["b3_claim_entailment"]

    report = summarize_method(
        "b3_claim_entailment",
        predictions,
        threshold=0.5,
        bootstrap_iterations=50,
        seed=4,
    )

    assert set(report.per_domain) == {"finqa"}
    assert report.pooled.f1 == 1.0
    assert report.macro_f1 == 1.0
    assert "pooled_f1" in report.confidence_intervals
    assert "finqa_f1" in report.confidence_intervals


def test_experiment_rejects_calibration_test_overlap(mocker):
    record = adapt_ragbench_row(_row(), domain="finqa")
    with pytest.raises(ValueError, match="partitions overlap"):
        run_experiment(
            [record],
            [record],
            embedding_model=mocker.MagicMock(),
            decomposer=DeterministicClaimDecomposer(),
            entailment_verifier=mocker.MagicMock(),
            metadata=_metadata(),
            bootstrap_iterations=10,
        )


def test_experiment_rejects_empty_partitions(mocker):
    with pytest.raises(ValueError, match="must both be non-empty"):
        run_experiment(
            [],
            [],
            embedding_model=mocker.MagicMock(),
            decomposer=DeterministicClaimDecomposer(),
            entailment_verifier=mocker.MagicMock(),
            metadata=_metadata(),
            bootstrap_iterations=10,
        )


def test_error_taxonomy_covers_numeric_negation_qualifier_partial_and_multisource():
    row = _row(
        response="Costs may not exceed $20 million.",
        response_sentences=[["a", "Costs may not exceed $20 million."]],
    )
    row["sentence_support_information"][0]["supporting_sentence_keys"] = ["0a", "0b"]
    record = adapt_ragbench_row(row, domain="finqa")
    prediction = GroundingSentencePrediction(
        example_id=record.example_id,
        domain=record.domain,
        sentence_key="a",
        sentence=record.response_sentences[0].text,
        gold_unsupported=True,
        predicted_unsupported=False,
        unsupported_score=0.1,
        claims=[],
    )

    assert categorize_error(prediction, record) == [
        "false_negative",
        "numeric",
        "negation",
        "qualifier",
        "partial_support",
        "multi_source",
    ]


def _metadata():
    return GroundingRunMetadata(
        dataset="galileo-ai/ragbench",
        dataset_revision="revision-1",
        split_strategy="official",
        calibration_split="validation",
        evaluation_split="test",
        seed=42,
        embedding_model="mini",
        embedding_model_revision="revision-1",
        entailment_model="fixture",
        entailment_model_revision="revision-1",
        claim_decomposer="deterministic_clause",
        claim_decomposer_version="1",
        similarity_threshold=0.0,
        entailment_threshold=0.0,
        code_commit="abc123",
    )


def test_experiment_cli_uses_pinned_revisions_and_method_selection():
    defaults = build_experiment_parser().parse_args(["--output", "report.json"])
    assert len(defaults.dataset_revision) == 40
    assert len(defaults.embedding_revision) == 40
    assert len(defaults.entailment_revision) == 40
    args = build_experiment_parser().parse_args(
        [
            "--domains",
            "techqa",
            "finqa",
            "--entailment-revision",
            "immutable-hash",
            "--methods",
            "b1_sentence_similarity",
            "b3_claim_entailment",
            "--output",
            "report.json",
        ]
    )
    assert args.domains == ["techqa", "finqa"]
    assert args.calibration_split == "validation"
    assert args.evaluation_split == "test"
    assert args.entailment_revision == "immutable-hash"
