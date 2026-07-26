"""Label-preserving RAGBench adapter and unsupported-sentence evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from models import (
    BenchmarkSentence,
    BenchmarkSentenceSupport,
    RAGBenchBenchmarkMetadata,
    RAGBenchBenchmarkReport,
    RAGBenchEvaluationRecord,
    RAGBenchExampleResult,
    RetrievedChunk,
    UnsupportedDetectionMetrics,
    UnsupportedSentencePrediction,
)
from services.forensics.chunk_attribution import (
    UNATTRIBUTED_THRESHOLD,
    analyze_sentences_attribution,
)

DATASET_NAME = "galileo-ai/ragbench"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_SUPPORT_SENTINELS = {"general", "well_known_fact", "supported_without_sentence"}


class RAGBenchRowError(ValueError):
    """Raised when a RAGBench row cannot be mapped without losing label alignment."""


def _pairs(value, field_name: str) -> list[BenchmarkSentence]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RAGBenchRowError(f"{field_name} must be a sequence")
    sentences: list[BenchmarkSentence] = []
    seen: set[str] = set()
    singular = "response sentence" if field_name == "response_sentences" else "document sentence"
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise RAGBenchRowError(f"{field_name} entries must be [key, text] pairs")
        key, text = str(item[0]), str(item[1])
        if key in seen:
            raise RAGBenchRowError(f"duplicate {singular} key: {key}")
        if not key or not text:
            raise RAGBenchRowError(f"{field_name} contains an empty key or text")
        seen.add(key)
        sentences.append(BenchmarkSentence(key=key, text=text))
    return sentences


def _optional_float(row: Mapping, key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def _normalize_response_reference(key, response_keys: set[str]) -> str:
    value = str(key)
    if value not in response_keys and value.endswith(".") and value[:-1] in response_keys:
        return value[:-1]
    return value


def adapt_ragbench_row(row: Mapping, domain: str) -> RAGBenchEvaluationRecord:
    """Map one row while preserving original text and validating label identifiers."""
    required = ("id", "question", "documents", "response", "response_sentences")
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise RAGBenchRowError(f"missing required fields: {', '.join(missing)}")

    documents = row["documents"]
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)) or not documents:
        raise RAGBenchRowError("documents must be a non-empty sequence")

    document_sentence_groups = row.get("documents_sentences")
    if not isinstance(document_sentence_groups, Sequence) or len(document_sentence_groups) != len(documents):
        raise RAGBenchRowError("documents_sentences must align one-to-one with documents")

    document_sentence_keys: set[str] = set()
    for group in document_sentence_groups:
        for sentence in _pairs(group, "documents_sentences"):
            if sentence.key in document_sentence_keys:
                raise RAGBenchRowError(f"duplicate document sentence key: {sentence.key}")
            document_sentence_keys.add(sentence.key)

    response_sentences = _pairs(row["response_sentences"], "response_sentences")
    response_keys = {sentence.key for sentence in response_sentences}
    unsupported_keys = {
        _normalize_response_reference(key, response_keys)
        for key in row.get("unsupported_response_sentence_keys", [])
    }
    unknown_unsupported = unsupported_keys - response_keys
    if unknown_unsupported:
        raise RAGBenchRowError(
            f"unknown unsupported response sentence key: {sorted(unknown_unsupported)[0]}"
        )

    support_by_key: dict[str, BenchmarkSentenceSupport] = {}
    for raw_support in row.get("sentence_support_information", []):
        response_key = _normalize_response_reference(
            raw_support.get("response_sentence_key", ""),
            response_keys,
        )
        if response_key not in response_keys:
            raise RAGBenchRowError(f"unknown support response sentence key: {response_key}")
        if response_key in support_by_key:
            raise RAGBenchRowError(f"duplicate support record for response sentence key: {response_key}")
        supporting_keys = [str(key) for key in raw_support.get("supporting_sentence_keys", [])]
        unknown_sources = set(supporting_keys) - document_sentence_keys - _SUPPORT_SENTINELS
        if unknown_sources:
            raise RAGBenchRowError(
                f"unknown supporting document sentence key: {sorted(unknown_sources)[0]}"
            )
        support_by_key[response_key] = BenchmarkSentenceSupport(
            response_sentence_key=response_key,
            fully_supported=bool(raw_support.get("fully_supported", False)),
            supporting_sentence_keys=supporting_keys,
            explanation=str(raw_support.get("explanation", "")),
        )

    chunks = [
        RetrievedChunk(chunk_id=f"document_{index}", text=str(text), score=1.0)
        for index, text in enumerate(documents)
    ]
    return RAGBenchEvaluationRecord(
        example_id=str(row["id"]),
        domain=domain,
        question=str(row["question"]),
        response=str(row["response"]),
        chunks=chunks,
        response_sentences=response_sentences,
        document_sentence_keys=document_sentence_keys,
        unsupported_response_sentence_keys=unsupported_keys,
        sentence_support=support_by_key,
        adherence_score=row.get("adherence_score"),
        relevance_score=_optional_float(row, "relevance_score"),
        utilization_score=_optional_float(row, "utilization_score"),
        completeness_score=_optional_float(row, "completeness_score"),
    )


def calculate_unsupported_metrics(
    gold: Sequence[bool],
    predicted: Sequence[bool],
    total_sentences: int | None = None,
) -> UnsupportedDetectionMetrics:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have equal length")
    tp = sum(actual and guess for actual, guess in zip(gold, predicted))
    fp = sum(not actual and guess for actual, guess in zip(gold, predicted))
    tn = sum(not actual and not guess for actual, guess in zip(gold, predicted))
    fn = sum(actual and not guess for actual, guess in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(gold) if total_sentences is None else total_sentences
    coverage = len(gold) / total if total else 0.0
    return UnsupportedDetectionMetrics(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        coverage=coverage,
        evaluated_sentences=len(gold),
        total_sentences=total,
    )


def evaluate_records(
    records: Iterable[RAGBenchEvaluationRecord],
    embedding_model,
    dataset_config: str,
    split: str,
    seed: int,
    requested_limit: int | None,
    timestamp: str,
    skipped_rows: list[str] | None = None,
) -> RAGBenchBenchmarkReport:
    examples: list[RAGBenchExampleResult] = []
    all_gold: list[bool] = []
    all_predicted: list[bool] = []

    for record in records:
        chunk_embeddings = embedding_model.encode([chunk.text for chunk in record.chunks])
        attribution = analyze_sentences_attribution(
            sentences=[sentence.text for sentence in record.response_sentences],
            chunks=record.chunks,
            chunk_embeddings=[list(vector) for vector in chunk_embeddings],
            embedding_model=embedding_model,
        )
        predictions: list[UnsupportedSentencePrediction] = []
        for sentence, entry in zip(record.response_sentences, attribution.attribution_map):
            gold_unsupported = sentence.key in record.unsupported_response_sentence_keys
            predicted_unsupported = entry.attribution_strength == "unattributed"
            support = record.sentence_support.get(sentence.key)
            all_gold.append(gold_unsupported)
            all_predicted.append(predicted_unsupported)
            predictions.append(
                UnsupportedSentencePrediction(
                    sentence_key=sentence.key,
                    sentence=sentence.text,
                    gold_unsupported=gold_unsupported,
                    predicted_unsupported=predicted_unsupported,
                    similarity_score=entry.similarity_score,
                    source_chunk_id=entry.chunk_id,
                    ragbench_fully_supported=support.fully_supported if support else None,
                    ragbench_supporting_sentence_keys=(
                        support.supporting_sentence_keys if support else []
                    ),
                )
            )
        examples.append(
            RAGBenchExampleResult(
                example_id=record.example_id,
                domain=record.domain,
                predictions=predictions,
                adherence_score=record.adherence_score,
                relevance_score=record.relevance_score,
                utilization_score=record.utilization_score,
                completeness_score=record.completeness_score,
            )
        )

    skips = skipped_rows or []
    return RAGBenchBenchmarkReport(
        metadata=RAGBenchBenchmarkMetadata(
            dataset=DATASET_NAME,
            dataset_config=dataset_config,
            split=split,
            seed=seed,
            requested_limit=requested_limit,
            sample_count=len(examples),
            skipped_count=len(skips),
            skipped_rows=skips,
            embedding_model=EMBEDDING_MODEL_NAME,
            unattributed_threshold=UNATTRIBUTED_THRESHOLD,
            timestamp=timestamp,
        ),
        metrics=calculate_unsupported_metrics(all_gold, all_predicted),
        examples=examples,
    )
