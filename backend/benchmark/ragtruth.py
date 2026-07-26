"""RAGTruth adapter with explicit span-to-sentence label conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from models import (
    BenchmarkDocumentSentence,
    BenchmarkSentence,
    RAGBenchEvaluationRecord,
    RetrievedChunk,
)

_SENTENCE = re.compile(r"\S(?:.*?\S)?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)


class RAGTruthRowError(ValueError):
    """Raised when joined RAGTruth source/response data is malformed."""


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    return [
        (match.start(), match.end(), match.group().strip())
        for match in _SENTENCE.finditer(text)
        if match.group().strip()
    ]


def _source_text(source_info) -> str:
    if isinstance(source_info, str):
        return source_info
    if isinstance(source_info, Mapping):
        return json.dumps(source_info, ensure_ascii=False, sort_keys=True)
    raise RAGTruthRowError("source_info must be a string or object")


def adapt_ragtruth_row(
    response_row: Mapping,
    source_row: Mapping,
) -> RAGBenchEvaluationRecord:
    """Convert RAGTruth spans to sentence outcomes while preserving raw source text."""
    if str(response_row.get("source_id")) != str(source_row.get("source_id")):
        raise RAGTruthRowError("response and source_info source_id values do not match")
    response = response_row.get("response")
    if not isinstance(response, str) or not response.strip():
        raise RAGTruthRowError("response must be a non-empty string")
    source_text = _source_text(source_row.get("source_info"))
    if not source_text.strip():
        raise RAGTruthRowError("source_info must not be empty")

    raw_labels = response_row.get("labels", [])
    label_spans: list[tuple[int, int]] = []
    for label in raw_labels:
        try:
            start, end = int(label["start"]), int(label["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RAGTruthRowError("labels require integer start and end offsets") from exc
        if start < 0 or end <= start or end > len(response):
            raise RAGTruthRowError(f"invalid hallucination span: {start}:{end}")
        if label.get("text") is not None and response[start:end] != label["text"]:
            raise RAGTruthRowError(
                f"hallucination span text does not match response at {start}:{end}"
            )
        label_spans.append((start, end))

    response_sentences = []
    unsupported_keys: set[str] = set()
    for index, (start, end, text) in enumerate(_sentence_spans(response)):
        key = f"s{index}"
        response_sentences.append(BenchmarkSentence(key=key, text=text))
        if any(start < label_end and label_start < end for label_start, label_end in label_spans):
            unsupported_keys.add(key)

    document_sentences = [
        BenchmarkDocumentSentence(
            key=f"d{index}",
            text=text,
            document_id="document_0",
        )
        for index, (_, _, text) in enumerate(_sentence_spans(source_text))
    ]
    if not response_sentences or not document_sentences:
        raise RAGTruthRowError("sentence conversion produced an empty response or source")

    domain = str(source_row.get("task_type", "unknown")).lower()
    return RAGBenchEvaluationRecord(
        example_id=str(response_row.get("id")),
        domain=f"ragtruth_{domain}",
        question=str(source_row.get("prompt", "")),
        response=response,
        chunks=[RetrievedChunk(chunk_id="document_0", text=source_text, score=1.0)],
        response_sentences=response_sentences,
        document_sentences=document_sentences,
        document_sentence_keys={item.key for item in document_sentences},
        unsupported_response_sentence_keys=unsupported_keys,
        sentence_support={},
    )
