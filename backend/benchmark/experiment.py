"""Leakage-resistant orchestration for the issue #18 grounding experiment."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from benchmark.grounding import (
    ClaimDecomposer,
    EntailmentVerifier,
    calculate_binary_metrics,
    calibrate_threshold,
    run_grounding_methods,
    summarize_method,
)
from models import (
    ConfidenceInterval,
    GroundingExperimentReport,
    GroundingRunMetadata,
    GroundingSentencePrediction,
    RAGBenchEvaluationRecord,
)


def _calibration_inputs(
    predictions: Sequence[GroundingSentencePrediction],
) -> tuple[list[float], list[bool]]:
    usable = [
        prediction
        for prediction in predictions
        if prediction.unsupported_score is not None
    ]
    return (
        [1.0 - float(item.unsupported_score) for item in usable],
        [item.gold_unsupported for item in usable],
    )


def categorize_error(
    prediction: GroundingSentencePrediction,
    record: RAGBenchEvaluationRecord,
) -> list[str]:
    if (
        prediction.predicted_unsupported is None
        or prediction.predicted_unsupported == prediction.gold_unsupported
    ):
        return []
    categories = [
        "false_positive"
        if prediction.predicted_unsupported
        else "false_negative"
    ]
    text = prediction.sentence.lower()
    if re.search(r"\d", text):
        categories.append("numeric")
    if re.search(r"\b(?:no|not|never|neither|without)\b", text):
        categories.append("negation")
    if re.search(
        r"\b(?:may|might|could|possibly|likely|always|all|only|must)\b",
        text,
    ):
        categories.append("qualifier")
    support = record.sentence_support.get(prediction.sentence_key)
    if support is not None and not support.fully_supported:
        categories.append("partial_support")
    if support is not None and len(support.supporting_sentence_keys) > 1:
        categories.append("multi_source")
    if any(claim.verifier_label == "contradiction" for claim in prediction.claims):
        categories.append("contradiction")
    if len(categories) == 1:
        categories.append("other")
    return categories


def _annotate_errors(
    predictions: Sequence[GroundingSentencePrediction],
    records: Sequence[RAGBenchEvaluationRecord],
) -> list[GroundingSentencePrediction]:
    record_index = {
        (record.domain, record.example_id): record for record in records
    }
    return [
        prediction.model_copy(
            update={
                "error_categories": categorize_error(
                    prediction,
                    record_index[(prediction.domain, prediction.example_id)],
                )
            }
        )
        for prediction in predictions
    ]


def _by_example(
    predictions: Sequence[GroundingSentencePrediction],
) -> dict[str, list[GroundingSentencePrediction]]:
    grouped: dict[str, list[GroundingSentencePrediction]] = {}
    for prediction in predictions:
        key = f"{prediction.domain}:{prediction.example_id}"
        grouped.setdefault(key, []).append(prediction)
    return grouped


def _macro_metric(
    predictions: Sequence[GroundingSentencePrediction],
    metric: str,
) -> float | None:
    values = []
    for domain in sorted({item.domain for item in predictions}):
        domain_predictions = [
            item for item in predictions if item.domain == domain
        ]
        result = calculate_binary_metrics(
            [item.gold_unsupported for item in domain_predictions],
            [item.predicted_unsupported for item in domain_predictions],
            [item.unsupported_score for item in domain_predictions],
        )
        value = getattr(result, metric)
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _paired_macro_interval(
    baseline: Sequence[GroundingSentencePrediction],
    candidate: Sequence[GroundingSentencePrediction],
    metric: str,
    iterations: int,
    seed: int,
) -> ConfidenceInterval:
    baseline_groups = _by_example(baseline)
    candidate_groups = _by_example(candidate)
    keys = sorted(set(baseline_groups) & set(candidate_groups))
    if not keys:
        raise ValueError("paired comparison requires shared examples")
    baseline_point = _macro_metric(baseline, metric)
    candidate_point = _macro_metric(candidate, metric)
    if baseline_point is None or candidate_point is None:
        raise ValueError(f"macro {metric} is unavailable")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sampled_keys = [
            keys[int(index)]
            for index in rng.integers(0, len(keys), size=len(keys))
        ]
        sampled_baseline = [
            item for key in sampled_keys for item in baseline_groups[key]
        ]
        sampled_candidate = [
            item for key in sampled_keys for item in candidate_groups[key]
        ]
        baseline_value = _macro_metric(sampled_baseline, metric)
        candidate_value = _macro_metric(sampled_candidate, metric)
        if baseline_value is not None and candidate_value is not None:
            differences.append(candidate_value - baseline_value)
    if not differences:
        raise ValueError(f"macro {metric} was unavailable in every bootstrap sample")
    return ConfidenceInterval(
        point_estimate=candidate_point - baseline_point,
        lower=float(np.quantile(differences, 0.025)),
        upper=float(np.quantile(differences, 0.975)),
        iterations=iterations,
        seed=seed,
    )


def _paired_intervals(
    baseline: Sequence[GroundingSentencePrediction],
    candidate: Sequence[GroundingSentencePrediction],
    iterations: int,
    seed: int,
) -> dict[str, ConfidenceInterval]:
    intervals = {
        "macro_f1": _paired_macro_interval(
            baseline, candidate, "f1", iterations, seed
        )
    }
    try:
        intervals["macro_auprc"] = _paired_macro_interval(
            baseline, candidate, "auprc", iterations, seed
        )
    except ValueError:
        pass
    domains = sorted(
        {item.domain for item in baseline} & {item.domain for item in candidate}
    )
    for domain in domains:
        intervals[f"{domain}_f1"] = _paired_macro_interval(
            [item for item in baseline if item.domain == domain],
            [item for item in candidate if item.domain == domain],
            "f1",
            iterations,
            seed,
        )
    return intervals


def run_experiment(
    calibration_records: Sequence[RAGBenchEvaluationRecord],
    evaluation_records: Sequence[RAGBenchEvaluationRecord],
    embedding_model,
    decomposer: ClaimDecomposer,
    entailment_verifier: EntailmentVerifier,
    metadata: GroundingRunMetadata,
    bootstrap_iterations: int = 2000,
    threshold_candidates: Sequence[float] | None = None,
) -> GroundingExperimentReport:
    """Calibrate on one partition and evaluate once on a distinct partition."""
    if not calibration_records or not evaluation_records:
        raise ValueError("calibration and evaluation records must both be non-empty")
    calibration_ids = {
        (record.domain, record.example_id) for record in calibration_records
    }
    evaluation_ids = {
        (record.domain, record.example_id) for record in evaluation_records
    }
    overlap = calibration_ids & evaluation_ids
    if overlap:
        raise ValueError(
            f"calibration and evaluation partitions overlap at {sorted(overlap)[0]}"
        )

    raw_calibration = run_grounding_methods(
        calibration_records,
        embedding_model=embedding_model,
        decomposer=decomposer,
        entailment_verifier=entailment_verifier,
        similarity_threshold=0.5,
        claim_similarity_threshold=0.5,
        entailment_threshold=0.5,
    )
    calibration = {}
    for method in (
        "b1_sentence_similarity",
        "b2_claim_similarity",
        "b3_claim_entailment",
    ):
        support_scores, labels = _calibration_inputs(raw_calibration[method])
        calibration[method] = calibrate_threshold(
            support_scores,
            labels,
            candidates=threshold_candidates,
        )

    evaluated = run_grounding_methods(
        evaluation_records,
        embedding_model=embedding_model,
        decomposer=decomposer,
        entailment_verifier=entailment_verifier,
        similarity_threshold=calibration["b1_sentence_similarity"].threshold,
        claim_similarity_threshold=calibration["b2_claim_similarity"].threshold,
        entailment_threshold=calibration["b3_claim_entailment"].threshold,
    )
    evaluated = {
        method: _annotate_errors(predictions, evaluation_records)
        for method, predictions in evaluated.items()
    }
    reports = {}
    for method, predictions in evaluated.items():
        threshold = (
            calibration[method].threshold if method in calibration else None
        )
        reports[method] = summarize_method(
            method,
            predictions,
            threshold=threshold,
            bootstrap_iterations=bootstrap_iterations,
            seed=metadata.seed,
        )
    metadata = metadata.model_copy(
        update={
            "similarity_threshold": calibration[
                "b1_sentence_similarity"
            ].threshold,
            "entailment_threshold": calibration[
                "b3_claim_entailment"
            ].threshold,
        }
    )
    paired = _paired_intervals(
        evaluated["b1_sentence_similarity"],
        evaluated["b3_claim_entailment"],
        iterations=bootstrap_iterations,
        seed=metadata.seed,
    )
    return GroundingExperimentReport(
        metadata=metadata,
        calibration=calibration,
        methods=reports,
        paired_b3_vs_b1=paired,
    )
