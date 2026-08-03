"""Oracle-evidence failure localization for supported RAGBench sentences.

This diagnostic uses label-derived evidence and is therefore not a deployable
grounding method. It asks whether B3 false negatives persist when evidence
selection is replaced with RAGBench's annotated supporting sentences.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from benchmark.grounding import (
    ClaimDecomposer,
    EntailmentVerifier,
    aggregate_claims,
    run_grounding_methods,
)
from models import (
    ClaimVerification,
    ConfidenceInterval,
    EvidenceCandidate,
    GroundingSentencePrediction,
    OracleEvidenceDiagnosticReport,
    OracleEvidenceEligibility,
    OracleEvidenceSentenceResult,
    OracleEvidenceStratumMetrics,
    RAGBenchEvaluationRecord,
)


def _rate(values: Sequence[bool | None]) -> float | None:
    evaluated = [value for value in values if value is not None]
    return float(np.mean(evaluated)) if evaluated else None


def _paired_interval(
    results: Sequence[OracleEvidenceSentenceResult], iterations: int, seed: int
) -> ConfidenceInterval | None:
    by_example: dict[str, list[tuple[bool, bool]]] = {}
    for result in results:
        selected = result.selected.predicted_unsupported
        oracle = result.oracle.predicted_unsupported
        if selected is None or oracle is None:
            continue
        key = f"{result.domain}:{result.example_id}"
        by_example.setdefault(key, []).append((selected, oracle))
    keys = sorted(by_example)
    if not keys:
        return None

    def difference(sampled: Sequence[str]) -> float:
        pairs = [pair for key in sampled for pair in by_example[key]]
        return float(np.mean([oracle for _, oracle in pairs])) - float(
            np.mean([selected for selected, _ in pairs])
        )

    point = difference(keys)
    rng = np.random.default_rng(seed)
    samples = [
        difference([keys[int(i)] for i in rng.integers(0, len(keys), len(keys))])
        for _ in range(iterations)
    ]
    return ConfidenceInterval(
        point_estimate=point,
        lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)),
        iterations=iterations,
        seed=seed,
    )


def _eligible_records(
    records: Sequence[RAGBenchEvaluationRecord],
) -> tuple[list[RAGBenchEvaluationRecord], int, Counter[str]]:
    eligible_records = []
    total = 0
    excluded: Counter[str] = Counter()
    for record in records:
        eligible_sentence_keys = set()
        for sentence in record.response_sentences:
            support = record.sentence_support.get(sentence.key)
            if support is None or not support.fully_supported:
                continue
            total += 1
            if not support.supporting_sentence_keys:
                excluded["missing_annotation"] += 1
                continue
            document_keys = set(record.document_sentence_keys)
            if any(key not in document_keys for key in support.supporting_sentence_keys):
                excluded["non_document_support"] += 1
                continue
            eligible_sentence_keys.add(sentence.key)
        if eligible_sentence_keys:
            eligible_records.append(
                record.model_copy(
                    update={
                        "response_sentences": [
                            sentence
                            for sentence in record.response_sentences
                            if sentence.key in eligible_sentence_keys
                        ],
                        "unsupported_response_sentence_keys": set(),
                    }
                )
            )
    return eligible_records, total, excluded


def _oracle_prediction(
    record: RAGBenchEvaluationRecord,
    selected: GroundingSentencePrediction,
    verifier: EntailmentVerifier,
    threshold: float,
) -> tuple[GroundingSentencePrediction, list[ClaimVerification]]:
    support = record.sentence_support[selected.sentence_key]
    documents = {sentence.key: sentence for sentence in record.document_sentences}
    evidence = [
        EvidenceCandidate(
            sentence_key=documents[key].key,
            document_id=documents[key].document_id,
            text=documents[key].text,
            selection_score=1.0,
        )
        for key in support.supporting_sentence_keys
    ]
    claims = [verification.claim for verification in selected.claims]
    all_pairs: list[ClaimVerification] = []
    best: list[ClaimVerification] = []
    for claim in claims:
        claim_pairs = []
        for candidate in evidence:
            try:
                scores = verifier.score(claim, candidate)
                verification = ClaimVerification(
                    claim=claim,
                    parent_sentence_key=claim.parent_sentence_key,
                    evidence=candidate,
                    support_score=scores.entailment,
                    predicted_supported=scores.entailment >= threshold,
                    status="ok",
                    verifier_label=scores.label,
                    nli_scores=scores,
                )
            except Exception as exc:
                verification = ClaimVerification(
                    claim=claim,
                    parent_sentence_key=claim.parent_sentence_key,
                    evidence=candidate,
                    support_score=None,
                    predicted_supported=None,
                    status="verifier_error",
                    error=str(exc),
                )
            claim_pairs.append(verification)
            all_pairs.append(verification)
        usable = [item for item in claim_pairs if item.support_score is not None]
        best.append(
            max(usable, key=lambda item: float(item.support_score))
            if usable
            else claim_pairs[0]
        )
    supported = aggregate_claims([item.predicted_supported for item in best])
    scores = [item.support_score for item in best if item.support_score is not None]
    return (
        GroundingSentencePrediction(
            example_id=record.example_id,
            domain=record.domain,
            sentence_key=selected.sentence_key,
            sentence=selected.sentence,
            gold_unsupported=False,
            predicted_unsupported=not supported if supported is not None else None,
            unsupported_score=1.0 - min(scores) if scores else None,
            claims=best,
        ),
        all_pairs,
    )


def _stratum_metrics(
    results: Sequence[OracleEvidenceSentenceResult],
) -> OracleEvidenceStratumMetrics:
    selected = [item.selected.predicted_unsupported for item in results]
    oracle = [item.oracle.predicted_unsupported for item in results]
    paired = [
        (left, right)
        for left, right in zip(selected, oracle)
        if left is not None and right is not None
    ]
    return OracleEvidenceStratumMetrics(
        sentences=len(results),
        selected_evaluated=sum(value is not None for value in selected),
        oracle_evaluated=sum(value is not None for value in oracle),
        paired_evaluated=len(paired),
        selected_false_unsupported_rate=_rate(selected),
        oracle_false_unsupported_rate=_rate(oracle),
        paired_difference=(
            float(np.mean([right for _, right in paired]))
            - float(np.mean([left for left, _ in paired]))
            if paired
            else None
        ),
    )


def run_oracle_evidence_diagnostic(
    records: Sequence[RAGBenchEvaluationRecord],
    embedding_model,
    decomposer: ClaimDecomposer,
    entailment_verifier: EntailmentVerifier,
    entailment_threshold: float,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> OracleEvidenceDiagnosticReport:
    """Compare selected versus annotated evidence on eligible supported sentences."""
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1")
    eligible_records, total, excluded = _eligible_records(records)
    if not eligible_records:
        return OracleEvidenceDiagnosticReport(
            eligibility=OracleEvidenceEligibility(
                total_fully_supported=total, eligible=0, excluded=dict(excluded)
            ),
            selected_false_unsupported_rate=None,
            oracle_false_unsupported_rate=None,
            selected_evidence_hit_at_1=None,
            paired_false_unsupported_difference=None,
            predictions=[],
        )
    selected_by_record = run_grounding_methods(
        eligible_records,
        embedding_model=embedding_model,
        decomposer=decomposer,
        entailment_verifier=entailment_verifier,
        similarity_threshold=0.5,
        claim_similarity_threshold=0.5,
        entailment_threshold=entailment_threshold,
    )["b3_claim_entailment"]
    record_index = {(item.domain, item.example_id): item for item in eligible_records}
    results = []
    for selected in selected_by_record:
        record = record_index[(selected.domain, selected.example_id)]
        oracle, pairs = _oracle_prediction(
            record, selected, entailment_verifier, entailment_threshold
        )
        annotated = record.sentence_support[selected.sentence_key].supporting_sentence_keys
        selected_keys = {
            claim.evidence.sentence_key
            for claim in selected.claims
            if claim.evidence is not None
        }
        results.append(
            OracleEvidenceSentenceResult(
                example_id=selected.example_id,
                domain=selected.domain,
                sentence_key=selected.sentence_key,
                sentence=selected.sentence,
                annotated_evidence_keys=annotated,
                selected=selected,
                oracle=oracle,
                oracle_pairs=pairs,
                selected_evidence_hit_at_1=bool(selected_keys & set(annotated)),
            )
        )
    overall = _stratum_metrics(results)
    domains = sorted({item.domain for item in results})
    source_strata = {
        "single_source": [
            item for item in results if len(item.annotated_evidence_keys) == 1
        ],
        "multi_source": [
            item for item in results if len(item.annotated_evidence_keys) > 1
        ],
    }
    return OracleEvidenceDiagnosticReport(
        eligibility=OracleEvidenceEligibility(
            total_fully_supported=total,
            eligible=len(results),
            excluded=dict(excluded),
        ),
        selected_false_unsupported_rate=_rate(
            [item.selected.predicted_unsupported for item in results]
        ),
        oracle_false_unsupported_rate=_rate(
            [item.oracle.predicted_unsupported for item in results]
        ),
        selected_evidence_hit_at_1=float(
            np.mean([item.selected_evidence_hit_at_1 for item in results])
        ),
        paired_false_unsupported_difference=_paired_interval(
            results, bootstrap_iterations, seed
        ),
        selected_evaluated=overall.selected_evaluated,
        oracle_evaluated=overall.oracle_evaluated,
        paired_evaluated=overall.paired_evaluated,
        per_domain={
            domain: _stratum_metrics(
                [item for item in results if item.domain == domain]
            )
            for domain in domains
        },
        by_source_count={
            name: _stratum_metrics(items)
            for name, items in source_strata.items()
            if items
        },
        predictions=results,
    )
