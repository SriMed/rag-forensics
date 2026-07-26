"""Comparable grounding methods and statistics for issue #18."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from models import (
    AtomicClaim,
    BinaryClassificationMetrics,
    CalibrationResult,
    ClaimVerification,
    ConfidenceInterval,
    EvidenceCandidate,
    GroundingSentencePrediction,
    GroundingMethodReport,
    NLIVerifierScores,
    RAGBenchEvaluationRecord,
)

METHODS = (
    "b0_always_supported",
    "b0_always_unsupported",
    "b1_sentence_similarity",
    "b2_claim_similarity",
    "b3_claim_entailment",
)


class ClaimDecomposer(Protocol):
    name: str
    version: str

    def decompose(self, sentence_key: str, sentence: str) -> list[AtomicClaim]: ...


class EntailmentVerifier(Protocol):
    name: str
    revision: str

    def score(
        self, claim: AtomicClaim, evidence: EvidenceCandidate
    ) -> NLIVerifierScores: ...


class DeterministicClaimDecomposer:
    """A reproducible clause baseline; it is not asserted to recover semantic atoms."""

    name = "deterministic_clause"
    version = "1"
    _boundary = re.compile(r"\s*(?:;|,\s*(?:but|and)|\bbut\b|\band\b)\s*", re.IGNORECASE)

    def decompose(self, sentence_key: str, sentence: str) -> list[AtomicClaim]:
        parts = [part.strip() for part in self._boundary.split(sentence) if part.strip()]
        return [
            AtomicClaim(
                claim_id=f"{sentence_key}.claim-{index}",
                parent_sentence_key=sentence_key,
                text=part,
            )
            for index, part in enumerate(parts)
        ]


class FixtureClaimDecomposer:
    """Exact claim map for offline evaluation fixtures."""

    name = "fixture"
    version = "1"

    def __init__(self, claims_by_sentence: Mapping[str, Sequence[str]]):
        self._claims_by_sentence = claims_by_sentence

    def decompose(self, sentence_key: str, sentence: str) -> list[AtomicClaim]:
        return [
            AtomicClaim(
                claim_id=f"{sentence_key}.claim-{index}",
                parent_sentence_key=sentence_key,
                text=text,
            )
            for index, text in enumerate(
                self._claims_by_sentence.get(sentence_key, [sentence])
            )
        ]


class FixtureEntailmentVerifier:
    """Deterministic verifier used by tests and offline fixtures."""

    name = "fixture"
    revision = "1"

    def __init__(
        self,
        scores: Mapping[
            tuple[str, str],
            float | Mapping[str, float],
        ],
    ):
        self._scores = scores

    def score(
        self, claim: AtomicClaim, evidence: EvidenceCandidate
    ) -> NLIVerifierScores:
        value = self._scores[(claim.text, evidence.sentence_key)]
        if isinstance(value, Mapping):
            probabilities = {key: float(score) for key, score in value.items()}
        else:
            probabilities = {
                "entailment": float(value),
                "neutral": 1.0 - float(value),
                "contradiction": 0.0,
            }
        label = max(probabilities, key=probabilities.get)
        return NLIVerifierScores(**probabilities, label=label)


class CrossEncoderNLIVerifier:
    """Lazy, local cross-encoder adapter with explicit label-order validation."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        revision: str = "main",
        model=None,
    ):
        self.name = model_name
        self.revision = revision
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, revision=revision)
        self._model = model

    def score(
        self, claim: AtomicClaim, evidence: EvidenceCandidate
    ) -> NLIVerifierScores:
        logits = self._model.predict([(evidence.text, claim.text)])[0]
        id2label = self._model.model.config.id2label
        probabilities = normalize_nli_scores(logits, id2label)
        return NLIVerifierScores(
            **probabilities,
            label=max(probabilities, key=probabilities.get),
        )


def normalize_nli_scores(
    logits: Sequence[float],
    id2label: Mapping[int, str],
) -> dict[str, float]:
    required = {"entailment", "neutral", "contradiction"}
    normalized_labels = {
        int(index): str(label).lower().strip() for index, label in id2label.items()
    }
    if set(normalized_labels.values()) != required:
        raise ValueError("NLI labels must identify entailment, neutral, and contradiction")
    values = np.asarray(logits, dtype=float)
    probabilities = np.exp(values - np.max(values))
    probabilities /= probabilities.sum()
    return {
        normalized_labels[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }


def aggregate_claims(predicted_supported: Sequence[bool | None]) -> bool | None:
    """Return sentence support; unknown claims prevent a forced classification."""
    if not predicted_supported or any(value is None for value in predicted_supported):
        return None
    return all(predicted_supported)


def calculate_binary_metrics(
    gold_unsupported: Sequence[bool],
    predicted_unsupported: Sequence[bool | None],
    unsupported_scores: Sequence[float | None],
    total: int | None = None,
) -> BinaryClassificationMetrics:
    if not (
        len(gold_unsupported) == len(predicted_unsupported) == len(unsupported_scores)
    ):
        raise ValueError("gold, predictions, and scores must have equal length")
    evaluated = [
        (gold, prediction, score)
        for gold, prediction, score in zip(
            gold_unsupported, predicted_unsupported, unsupported_scores
        )
        if prediction is not None and score is not None
    ]
    gold = [item[0] for item in evaluated]
    predicted = [bool(item[1]) for item in evaluated]
    scores = [float(item[2]) for item in evaluated]
    tp = sum(actual and guess for actual, guess in zip(gold, predicted))
    fp = sum(not actual and guess for actual, guess in zip(gold, predicted))
    tn = sum(not actual and not guess for actual, guess in zip(gold, predicted))
    fn = sum(actual and not guess for actual, guess in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rank_metrics_available = bool(gold) and len(set(gold)) == 2
    total_count = len(gold_unsupported) if total is None else total
    return BinaryClassificationMetrics(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=float(roc_auc_score(gold, scores)) if rank_metrics_available else None,
        auprc=float(average_precision_score(gold, scores))
        if rank_metrics_available
        else None,
        prevalence=sum(gold) / len(gold) if gold else 0.0,
        coverage=len(evaluated) / total_count if total_count else 0.0,
        evaluated=len(evaluated),
        total=total_count,
    )


def calibrate_threshold(
    support_scores: Sequence[float],
    gold_unsupported: Sequence[bool],
    candidates: Sequence[float] | None = None,
) -> CalibrationResult:
    if len(support_scores) != len(gold_unsupported) or not support_scores:
        raise ValueError("calibration scores and labels must be non-empty and aligned")
    if candidates is not None:
        thresholds = sorted(set(float(value) for value in candidates))
    else:
        unique = sorted(set(float(value) for value in support_scores))
        thresholds = [float(np.nextafter(unique[0], -np.inf))]
        thresholds.extend(
            (left + right) / 2 for left, right in zip(unique, unique[1:])
        )
        thresholds.append(float(np.nextafter(unique[-1], np.inf)))
    best_threshold = thresholds[0]
    best_f1 = -1.0
    for threshold in thresholds:
        predictions = [score < threshold for score in support_scores]
        metrics = calculate_binary_metrics(
            gold_unsupported,
            predictions,
            [1.0 - score for score in support_scores],
        )
        if metrics.f1 > best_f1:
            best_threshold, best_f1 = threshold, metrics.f1
    return CalibrationResult(threshold=best_threshold, objective_value=best_f1)


def bootstrap_paired_difference(
    baseline_by_example: Mapping[str, float],
    candidate_by_example: Mapping[str, float],
    iterations: int = 2000,
    seed: int = 42,
) -> ConfidenceInterval:
    keys = sorted(set(baseline_by_example) & set(candidate_by_example))
    if not keys:
        raise ValueError("paired bootstrap requires shared example identifiers")
    baseline = np.asarray([baseline_by_example[key] for key in keys], dtype=float)
    candidate = np.asarray([candidate_by_example[key] for key in keys], dtype=float)
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [
            np.mean(differences[rng.integers(0, len(keys), size=len(keys))])
            for _ in range(iterations)
        ]
    )
    return ConfidenceInterval(
        point_estimate=float(np.mean(differences)),
        lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)),
        iterations=iterations,
        seed=seed,
    )


def _prediction_metrics(
    predictions: Sequence[GroundingSentencePrediction],
) -> BinaryClassificationMetrics:
    return calculate_binary_metrics(
        [item.gold_unsupported for item in predictions],
        [item.predicted_unsupported for item in predictions],
        [item.unsupported_score for item in predictions],
        total=len(predictions),
    )


def _cluster_bootstrap_interval(
    predictions: Sequence[GroundingSentencePrediction],
    metric: str,
    iterations: int,
    seed: int,
) -> ConfidenceInterval:
    by_example: dict[str, list[GroundingSentencePrediction]] = {}
    for prediction in predictions:
        key = f"{prediction.domain}:{prediction.example_id}"
        by_example.setdefault(key, []).append(prediction)
    keys = sorted(by_example)
    if not keys:
        raise ValueError("bootstrap requires at least one example")
    point = getattr(_prediction_metrics(predictions), metric)
    if point is None:
        raise ValueError(f"{metric} is unavailable for these labels")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled: list[GroundingSentencePrediction] = []
        for index in rng.integers(0, len(keys), size=len(keys)):
            sampled.extend(by_example[keys[int(index)]])
        value = getattr(_prediction_metrics(sampled), metric)
        if value is not None:
            samples.append(float(value))
    if not samples:
        raise ValueError(f"{metric} is unavailable in every bootstrap sample")
    return ConfidenceInterval(
        point_estimate=float(point),
        lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)),
        iterations=iterations,
        seed=seed,
    )


def summarize_method(
    method: str,
    predictions: Sequence[GroundingSentencePrediction],
    threshold: float | None,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> GroundingMethodReport:
    domains = sorted({item.domain for item in predictions})
    per_domain = {
        domain: _prediction_metrics(
            [item for item in predictions if item.domain == domain]
        )
        for domain in domains
    }
    pooled = _prediction_metrics(predictions)
    intervals: dict[str, ConfidenceInterval] = {}
    groups = {"pooled": list(predictions)}
    groups.update(
        {
            domain: [item for item in predictions if item.domain == domain]
            for domain in domains
        }
    )
    for group_name, group_predictions in groups.items():
        for metric in (
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "prevalence",
            "coverage",
        ):
            try:
                intervals[f"{group_name}_{metric}"] = _cluster_bootstrap_interval(
                    group_predictions,
                    metric,
                    bootstrap_iterations,
                    seed,
                )
            except ValueError:
                continue
    auprc_values = [
        metrics.auprc for metrics in per_domain.values() if metrics.auprc is not None
    ]
    return GroundingMethodReport(
        method=method,
        threshold=threshold,
        per_domain=per_domain,
        pooled=pooled,
        macro_f1=(
            float(np.mean([metrics.f1 for metrics in per_domain.values()]))
            if per_domain
            else 0.0
        ),
        macro_auprc=float(np.mean(auprc_values)) if auprc_values else None,
        confidence_intervals=intervals,
        predictions=list(predictions),
    )


def _candidate(
    record: RAGBenchEvaluationRecord,
    scores: np.ndarray,
) -> EvidenceCandidate:
    index = int(np.argmax(scores))
    sentence = record.document_sentences[index]
    return EvidenceCandidate(
        sentence_key=sentence.key,
        document_id=sentence.document_id,
        text=sentence.text,
        selection_score=float(scores[index]),
    )


def _verification(
    claim: AtomicClaim,
    evidence: EvidenceCandidate,
    score: float | None,
    threshold: float,
    status: str = "ok",
    error: str | None = None,
    nli_scores: NLIVerifierScores | None = None,
) -> ClaimVerification:
    return ClaimVerification(
        claim=claim,
        parent_sentence_key=claim.parent_sentence_key,
        evidence=evidence,
        support_score=score,
        predicted_supported=score >= threshold if score is not None else None,
        status=status,
        verifier_label=(
            nli_scores.label
            if nli_scores is not None
            else "similar" if score is not None and score >= threshold
            else "dissimilar" if score is not None
            else None
        ),
        nli_scores=nli_scores,
        error=error,
    )


def run_grounding_methods(
    records: Sequence[RAGBenchEvaluationRecord],
    embedding_model,
    decomposer: ClaimDecomposer,
    entailment_verifier: EntailmentVerifier,
    similarity_threshold: float,
    entailment_threshold: float,
    claim_similarity_threshold: float | None = None,
) -> dict[str, list[GroundingSentencePrediction]]:
    results: dict[str, list[GroundingSentencePrediction]] = {
        method: [] for method in METHODS
    }
    claim_threshold = (
        similarity_threshold
        if claim_similarity_threshold is None
        else claim_similarity_threshold
    )
    for record in records:
        if not record.document_sentences:
            continue
        evidence_embeddings = np.asarray(
            embedding_model.encode(
                [sentence.text for sentence in record.document_sentences]
            ),
            dtype=float,
        )
        response_embeddings = np.asarray(
            embedding_model.encode(
                [sentence.text for sentence in record.response_sentences]
            ),
            dtype=float,
        )
        claims_by_sentence = [
            decomposer.decompose(sentence.key, sentence.text)
            for sentence in record.response_sentences
        ]
        flattened_claims = [claim for claims in claims_by_sentence for claim in claims]
        claim_embeddings = np.asarray(
            embedding_model.encode([claim.text for claim in flattened_claims]),
            dtype=float,
        )
        claim_offset = 0
        for sentence_index, (sentence, claims) in enumerate(
            zip(record.response_sentences, claims_by_sentence)
        ):
            gold = sentence.key in record.unsupported_response_sentence_keys
            for method, prediction, score in (
                ("b0_always_supported", False, 0.0),
                ("b0_always_unsupported", True, 1.0),
            ):
                results[method].append(
                    GroundingSentencePrediction(
                        example_id=record.example_id,
                        domain=record.domain,
                        sentence_key=sentence.key,
                        sentence=sentence.text,
                        gold_unsupported=gold,
                        predicted_unsupported=prediction,
                        unsupported_score=score,
                        claims=[],
                    )
                )

            sentence_scores = cosine_similarity(
                response_embeddings[sentence_index : sentence_index + 1],
                evidence_embeddings,
            )[0]
            sentence_evidence = _candidate(record, sentence_scores)
            sentence_claim = AtomicClaim(
                claim_id=f"{sentence.key}.sentence",
                parent_sentence_key=sentence.key,
                text=sentence.text,
            )
            b1_verification = _verification(
                sentence_claim,
                sentence_evidence,
                sentence_evidence.selection_score,
                similarity_threshold,
            )
            results["b1_sentence_similarity"].append(
                GroundingSentencePrediction(
                    example_id=record.example_id,
                    domain=record.domain,
                    sentence_key=sentence.key,
                    sentence=sentence.text,
                    gold_unsupported=gold,
                    predicted_unsupported=not bool(b1_verification.predicted_supported),
                    unsupported_score=1.0 - sentence_evidence.selection_score,
                    claims=[b1_verification],
                )
            )

            similarity_verifications: list[ClaimVerification] = []
            entailment_verifications: list[ClaimVerification] = []
            for local_index, claim in enumerate(claims):
                claim_scores = cosine_similarity(
                    claim_embeddings[claim_offset + local_index : claim_offset + local_index + 1],
                    evidence_embeddings,
                )[0]
                evidence = _candidate(record, claim_scores)
                similarity_verifications.append(
                    _verification(
                        claim,
                        evidence,
                        evidence.selection_score,
                        claim_threshold,
                    )
                )
                try:
                    nli_scores = entailment_verifier.score(claim, evidence)
                    entailment_verifications.append(
                        _verification(
                            claim,
                            evidence,
                            nli_scores.entailment,
                            entailment_threshold,
                            nli_scores=nli_scores,
                        )
                    )
                except Exception as exc:
                    entailment_verifications.append(
                        _verification(
                            claim,
                            evidence,
                            None,
                            entailment_threshold,
                            status="verifier_error",
                            error=str(exc),
                        )
                    )
            claim_offset += len(claims)
            for method, verifications in (
                ("b2_claim_similarity", similarity_verifications),
                ("b3_claim_entailment", entailment_verifications),
            ):
                supported = aggregate_claims(
                    [item.predicted_supported for item in verifications]
                )
                support_scores = [
                    item.support_score
                    for item in verifications
                    if item.support_score is not None
                ]
                results[method].append(
                    GroundingSentencePrediction(
                        example_id=record.example_id,
                        domain=record.domain,
                        sentence_key=sentence.key,
                        sentence=sentence.text,
                        gold_unsupported=gold,
                        predicted_unsupported=(
                            not supported if supported is not None else None
                        ),
                        unsupported_score=(
                            1.0 - min(support_scores) if support_scores else None
                        ),
                        claims=verifications,
                    )
                )
    return results
