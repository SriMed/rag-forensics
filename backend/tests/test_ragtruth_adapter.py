import pytest

from benchmark.ragtruth import RAGTruthRowError, adapt_ragtruth_row
from benchmark.ragtruth_cli import build_parser


def _source(**overrides):
    row = {
        "source_id": "source-1",
        "task_type": "QA",
        "source_info": {
            "question": "What changed?",
            "passages": "Revenue rose. Costs stayed flat.",
        },
        "prompt": "Answer using the passages.",
    }
    row.update(overrides)
    return row


def _response(**overrides):
    text = "Revenue rose. Costs doubled."
    row = {
        "id": "response-1",
        "source_id": "source-1",
        "response": text,
        "labels": [
            {
                "start": text.index("Costs doubled"),
                "end": text.index("Costs doubled") + len("Costs doubled"),
                "text": "Costs doubled",
                "label_type": "Evident Baseless Info",
            }
        ],
        "split": "test",
        "quality": "good",
    }
    row.update(overrides)
    return row


def test_ragtruth_adapter_maps_span_overlap_to_sentence_label():
    record = adapt_ragtruth_row(_response(), _source())

    assert record.example_id == "response-1"
    assert record.domain == "ragtruth_qa"
    assert [item.key for item in record.response_sentences] == ["s0", "s1"]
    assert record.unsupported_response_sentence_keys == {"s1"}
    assert record.question == "Answer using the passages."
    assert record.document_sentences
    assert all(
        item.document_id == "document_0" for item in record.document_sentences
    )


def test_ragtruth_adapter_preserves_implicit_true_as_context_unsupported():
    response = _response()
    response["labels"][0]["implicit_true"] = True
    record = adapt_ragtruth_row(response, _source())
    assert record.unsupported_response_sentence_keys == {"s1"}


def test_ragtruth_adapter_rejects_misaligned_span_text():
    response = _response()
    response["labels"][0]["text"] = "different"
    with pytest.raises(RAGTruthRowError, match="does not match"):
        adapt_ragtruth_row(response, _source())


def test_ragtruth_adapter_rejects_source_mismatch():
    with pytest.raises(RAGTruthRowError, match="source_id"):
        adapt_ragtruth_row(_response(source_id="other"), _source())


def test_ragtruth_adapter_accepts_string_source_info():
    record = adapt_ragtruth_row(
        _response(),
        _source(source_info="Revenue rose. Costs stayed flat."),
    )
    assert [item.text for item in record.document_sentences] == [
        "Revenue rose.",
        "Costs stayed flat.",
    ]


def test_ragtruth_cli_requires_revision_and_official_files():
    args = build_parser().parse_args(
        [
            "--source-info",
            "source_info.jsonl",
            "--responses",
            "response.jsonl",
            "--dataset-revision",
            "immutable-hash",
            "--output",
            "report.json",
        ]
    )
    assert args.dataset_revision == "immutable-hash"
    assert args.seed == 42
