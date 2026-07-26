import json

import numpy as np
import pytest

from benchmark.ragbench import (
    RAGBenchRowError,
    adapt_ragbench_row,
    calculate_unsupported_metrics,
    evaluate_records,
)
from benchmark.cli import build_parser, run_rows


def ragbench_row(**overrides):
    row = {
        "id": "example-1",
        "question": "What is X?",
        "documents": ["X is supported.", "Unrelated material."],
        "response": "X is supported. Y is invented.",
        "documents_sentences": [
            [["0a", "X is supported."]],
            [["1a", "Unrelated material."]],
        ],
        "response_sentences": [
            ["a", "X is supported."],
            ["b", "Y is invented."],
        ],
        "sentence_support_information": [
            {
                "response_sentence_key": "a",
                "fully_supported": True,
                "supporting_sentence_keys": ["0a"],
                "explanation": "Supported by document sentence 0a.",
            },
            {
                "response_sentence_key": "b",
                "fully_supported": False,
                "supporting_sentence_keys": [],
                "explanation": "No document supports this sentence.",
            },
        ],
        "unsupported_response_sentence_keys": ["b"],
        "adherence_score": False,
        "relevance_score": 0.5,
        "utilization_score": 0.5,
        "completeness_score": 1.0,
    }
    row.update(overrides)
    return row


def test_adapter_preserves_original_content_and_labels():
    row = ragbench_row()
    record = adapt_ragbench_row(row, domain="techqa")

    assert record.example_id == row["id"]
    assert record.question == row["question"]
    assert record.response == row["response"]
    assert [chunk.text for chunk in record.chunks] == row["documents"]
    assert [sentence.key for sentence in record.response_sentences] == ["a", "b"]
    assert record.unsupported_response_sentence_keys == {"b"}
    assert record.sentence_support["a"].supporting_sentence_keys == ["0a"]
    assert record.relevance_score == 0.5


def test_adapter_rejects_unknown_unsupported_sentence_key():
    with pytest.raises(RAGBenchRowError, match="unknown unsupported response sentence key"):
        adapt_ragbench_row(
            ragbench_row(unsupported_response_sentence_keys=["missing"]),
            domain="techqa",
        )


def test_adapter_rejects_duplicate_response_sentence_keys():
    with pytest.raises(RAGBenchRowError, match="duplicate response sentence key"):
        adapt_ragbench_row(
            ragbench_row(
                response_sentences=[
                    ["a", "First."],
                    ["a", "Second."],
                ]
            ),
            domain="techqa",
        )


def test_adapter_rejects_unknown_supporting_document_sentence_key():
    row = ragbench_row()
    row["sentence_support_information"][0]["supporting_sentence_keys"] = ["missing"]
    with pytest.raises(RAGBenchRowError, match="unknown supporting document sentence key"):
        adapt_ragbench_row(row, domain="techqa")


@pytest.mark.parametrize("sentinel", ["general", "well_known_fact", "supported_without_sentence"])
def test_adapter_accepts_ragbench_non_document_support_sentinels(sentinel):
    row = ragbench_row()
    row["sentence_support_information"][0]["supporting_sentence_keys"] = [sentinel]
    record = adapt_ragbench_row(row, domain="techqa")
    assert record.sentence_support["a"].supporting_sentence_keys == [sentinel]


def test_adapter_normalizes_trailing_periods_in_label_references():
    row = ragbench_row(unsupported_response_sentence_keys=["b."])
    row["sentence_support_information"][0]["response_sentence_key"] = "a."
    row["sentence_support_information"][1]["response_sentence_key"] = "b."
    record = adapt_ragbench_row(row, domain="techqa")
    assert record.unsupported_response_sentence_keys == {"b"}
    assert set(record.sentence_support) == {"a", "b"}


def test_adapter_rejects_document_sentence_count_mismatch():
    with pytest.raises(RAGBenchRowError, match="documents_sentences"):
        adapt_ragbench_row(
            ragbench_row(documents_sentences=[[["0a", "X is supported."]]]),
            domain="techqa",
        )


def test_calculate_metrics_exact_confusion_matrix():
    metrics = calculate_unsupported_metrics(
        gold=[False, True, True, False],
        predicted=[False, True, False, True],
    )
    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_negative == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.coverage == pytest.approx(1.0)


def test_calculate_metrics_handles_no_positive_predictions():
    metrics = calculate_unsupported_metrics(
        gold=[False, True],
        predicted=[False, False],
    )
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_evaluate_records_is_offline_and_preserves_sentence_ids(mocker):
    record = adapt_ragbench_row(ragbench_row(), domain="techqa")
    model = mocker.MagicMock()
    model.encode.side_effect = [
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]

    report = evaluate_records(
        records=[record],
        embedding_model=model,
        dataset_config="techqa",
        split="test",
        seed=7,
        requested_limit=1,
        timestamp="2026-07-26T12:00:00Z",
    )

    assert report.metadata.dataset == "galileo-ai/ragbench"
    assert report.metadata.dataset_config == "techqa"
    assert report.metadata.sample_count == 1
    assert report.metadata.skipped_count == 0
    assert report.metrics.true_positive == 1
    assert report.metrics.true_negative == 1
    assert [item.sentence_key for item in report.examples[0].predictions] == ["a", "b"]
    assert report.examples[0].predictions[1].predicted_unsupported is True
    assert report.examples[0].predictions[0].ragbench_supporting_sentence_keys == ["0a"]
    assert report.examples[0].relevance_score == 0.5
    assert model.encode.call_count == 2


def test_report_serializes_as_machine_readable_json(mocker):
    record = adapt_ragbench_row(ragbench_row(), domain="covidqa")
    model = mocker.MagicMock()
    model.encode.side_effect = [
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    report = evaluate_records(
        [record],
        embedding_model=model,
        dataset_config="covidqa",
        split="validation",
        seed=11,
        requested_limit=None,
        timestamp="2026-07-26T12:00:00Z",
    )

    parsed = json.loads(report.model_dump_json())
    assert parsed["metadata"]["split"] == "validation"
    assert parsed["examples"][0]["example_id"] == "example-1"
    assert parsed["metrics"]["coverage"] == 1.0


def test_run_rows_counts_invalid_rows_as_explicit_skips(mocker):
    model = mocker.MagicMock()
    model.encode.side_effect = [
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    invalid = ragbench_row(id="bad", unsupported_response_sentence_keys=["missing"])

    report = run_rows(
        rows=[ragbench_row(), invalid],
        embedding_model=model,
        domain="finqa",
        split="test",
        seed=42,
        requested_limit=2,
        timestamp="2026-07-26T12:00:00Z",
    )

    assert report.metadata.sample_count == 1
    assert report.metadata.skipped_count == 1
    assert report.metadata.skipped_rows[0].startswith("bad:")


def test_cli_supports_required_reproducibility_arguments():
    args = build_parser().parse_args(
        [
            "--domain",
            "covidqa",
            "--split",
            "validation",
            "--limit",
            "25",
            "--seed",
            "99",
            "--output",
            "report.json",
        ]
    )
    assert args.domain == "covidqa"
    assert args.split == "validation"
    assert args.limit == 25
    assert args.seed == 99
    assert args.output == "report.json"
