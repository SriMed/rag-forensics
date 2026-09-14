"""Two-by-two decomposition/evidence failure-localization experiment.

Human claim reviews are inputs to this module, never inferred ground truth.  The
runner deliberately rejects drafts so results cannot accidentally precede review.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from benchmark.grounding import ClaimDecomposer, EntailmentVerifier, FixtureClaimDecomposer
from benchmark.oracle_evidence import _eligible_records, run_oracle_evidence_diagnostic
from models import (
    ClaimVerification,
    ConfidenceInterval,
    GroundingSentencePrediction,
    RAGBenchEvaluationRecord,
)

CONDITIONS = ("A", "B", "C", "D")
RESIDUAL_CATEGORIES = (
    "verifier_error",
    "multi_sentence_support",
    "numerical_or_tabular_reasoning",
    "annotation_granularity_mismatch",
    "ambiguous_label",
    "undetermined",
)


class ClaimReviewItem(BaseModel):
    domain: str
    example_id: str
    sentence_key: str
    question: str
    full_response: str
    sentence: str
    deterministic_claims: list[str]
    accept_deterministic: bool | None = None
    reviewed_claims: list[str] = Field(default_factory=list)
    reviewer_notes: str | None = None


class ClaimReviewArtifact(BaseModel):
    schema_version: Literal["decomposition-evidence-claims.v1"]
    status: Literal["draft", "frozen"] = "draft"
    reviewer_identity: str | None = None
    reviewer_instructions: str
    population_sha256: str
    decomposer_name: str
    decomposer_version: str
    revision: int = 1
    unresolved_judgments: list[str] = Field(default_factory=list)
    items: list[ClaimReviewItem]

    @model_validator(mode="after")
    def validate_frozen_review(self):
        keys = [(x.domain, x.example_id, x.sentence_key) for x in self.items]
        if len(keys) != len(set(keys)):
            raise ValueError("claim review contains duplicate sentence identities")
        if self.status == "frozen":
            if not self.reviewer_identity:
                raise ValueError("a frozen claim review requires reviewer_identity")
            if any(item.accept_deterministic is None for item in self.items):
                raise ValueError("every frozen item requires an accept_deterministic decision")
            if any(item.accept_deterministic is False and not item.reviewed_claims for item in self.items):
                raise ValueError("rejected deterministic claims require reviewed_claims")
        return self


class ResidualReviewItem(BaseModel):
    domain: str
    example_id: str
    sentence_key: str
    question: str
    full_response: str
    sentence: str
    reviewed_claims: list[str]
    annotated_evidence: list[str]
    verifier_outcomes: list[ClaimVerification]
    category: Literal[
        "verifier_error", "multi_sentence_support", "numerical_or_tabular_reasoning",
        "annotation_granularity_mismatch", "ambiguous_label", "undetermined"
    ] | None = None
    notes: str | None = None


class ResidualReviewArtifact(BaseModel):
    schema_version: Literal["decomposition-evidence-residuals.v1"]
    status: Literal["draft", "frozen"] = "draft"
    reviewer_identity: str | None = None
    reviewer_instructions: str
    experiment_report_sha256: str
    revision: int = 1
    items: list[ResidualReviewItem]

    @model_validator(mode="after")
    def validate_frozen_review(self):
        if self.status == "frozen":
            if not self.reviewer_identity:
                raise ValueError("a frozen residual review requires reviewer_identity")
            if any(item.category is None for item in self.items):
                raise ValueError("every frozen residual requires a category")
        return self


class ConditionResult(BaseModel):
    condition: Literal["A", "B", "C", "D"]
    false_unsupported_rate: float | None
    evaluated: int
    predictions: list[GroundingSentencePrediction]
    pair_evaluations: dict[str, list[ClaimVerification]] = Field(default_factory=dict)


class PairedEffect(BaseModel):
    contrast: str
    interval: ConfidenceInterval | None


class ExperimentMetadata(BaseModel):
    dataset: str
    dataset_revision: str
    evaluation_split: str
    sample_limit_per_domain: int | None
    seed: int
    bootstrap_iterations: int
    embedding_model: str
    embedding_model_revision: str
    verifier_model: str
    verifier_revision: str
    entailment_threshold: float
    aggregation: Literal["all_claims_supported"] = "all_claims_supported"
    deterministic_decomposer: str
    deterministic_decomposer_version: str
    claim_review_revision: int
    reviewer_identity: str


class DecompositionEvidenceReport(BaseModel):
    schema_version: Literal["decomposition-evidence-report.v1"]
    status: Literal["frozen"] = "frozen"
    metadata: ExperimentMetadata | None = None
    population_size: int
    exclusions: dict[str, int]
    claim_review_sha256: str
    conditions: dict[str, ConditionResult]
    effects: dict[str, PairedEffect]
    residual_condition_d_keys: list[str]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population_sha256(records: Sequence[RAGBenchEvaluationRecord]) -> str:
    payload = [
        {
            "domain": record.domain,
            "example_id": record.example_id,
            "question": record.question,
            "response": record.response,
            "sentences": [sentence.model_dump() for sentence in record.response_sentences],
        }
        for record in records
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_claim_review_template(
    records: Sequence[RAGBenchEvaluationRecord], decomposer: ClaimDecomposer,
) -> ClaimReviewArtifact:
    eligible, _, _ = _eligible_records(records)
    items = []
    for record in eligible:
        for sentence in record.response_sentences:
            items.append(ClaimReviewItem(
                domain=record.domain,
                example_id=record.example_id,
                sentence_key=sentence.key,
                question=record.question,
                full_response=record.response,
                sentence=sentence.text,
                deterministic_claims=[
                    claim.text
                    for claim in decomposer.decompose(sentence.key, sentence.text)
                ],
            ))
    return ClaimReviewArtifact(
        schema_version="decomposition-evidence-claims.v1",
        reviewer_instructions=(
            "Rewrite each supported response sentence as the smallest independently verifiable "
            "claims without adding facts. Use only the question, full response, target sentence, and "
            "proposed decomposition in this packet. Do not consult evidence, condition assignments, "
            "verifier outcomes, or downstream results. Preserve "
            "qualifiers, negation, quantities, entities, and relations. Set accept_deterministic=true "
            "when the proposed split is already atomic; otherwise set it false and provide reviewed_claims. "
            "Record uncertainty in unresolved_judgments and still provide the best review decision; "
            "set status=frozen only after every item is reviewed."
        ),
        population_sha256=population_sha256(eligible),
        decomposer_name=decomposer.name,
        decomposer_version=decomposer.version,
        items=items,
    )


def _key(domain: str, example_id: str, sentence_key: str) -> tuple[str, str, str]:
    return domain, example_id, sentence_key


def _interval(
    predictions: Mapping[str, Sequence[GroundingSentencePrediction]],
    weights: Mapping[str, float], iterations: int, seed: int,
) -> ConfidenceInterval | None:
    by_example: dict[str, list[dict[str, bool]]] = {}
    for condition, rows in predictions.items():
        for row in rows:
            if row.predicted_unsupported is None:
                continue
            cluster = f"{row.domain}:{row.example_id}"
            sentence = f"{row.domain}:{row.example_id}:{row.sentence_key}"
            bucket = by_example.setdefault(cluster, [])
            match = next((x for x in bucket if x["_sentence"] == sentence), None)
            if match is None:
                match = {"_sentence": sentence}
                bucket.append(match)
            match[condition] = row.predicted_unsupported
    clusters = sorted(by_example)
    complete = {
        cluster: [row for row in by_example[cluster] if all(c in row for c in weights)]
        for cluster in clusters
    }
    clusters = [c for c in clusters if complete[c]]
    if not clusters:
        return None

    def statistic(sampled):
        rows = [row for cluster in sampled for row in complete[cluster]]
        return float(np.mean([sum(weights[c] * float(row[c]) for c in weights) for row in rows]))

    point = statistic(clusters)
    rng = np.random.default_rng(seed)
    samples = [statistic([clusters[int(i)] for i in rng.integers(0, len(clusters), len(clusters))]) for _ in range(iterations)]
    return ConfidenceInterval(point_estimate=point, lower=float(np.quantile(samples, .025)),
                              upper=float(np.quantile(samples, .975)), iterations=iterations, seed=seed)


def run_decomposition_evidence_experiment(
    records: Sequence[RAGBenchEvaluationRecord], embedding_model,
    deterministic_decomposer: ClaimDecomposer, verifier: EntailmentVerifier,
    threshold: float, review: ClaimReviewArtifact, review_sha256: str,
    bootstrap_iterations: int = 2000, seed: int = 42,
    metadata: ExperimentMetadata | None = None,
) -> DecompositionEvidenceReport:
    if review.status != "frozen":
        raise ValueError("claim review must be frozen before running the experiment")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1")
    eligible, _, excluded = _eligible_records(records)
    if review.population_sha256 != population_sha256(eligible):
        raise ValueError("claim review population provenance does not match eligible records")
    if (review.decomposer_name, review.decomposer_version) != (
        deterministic_decomposer.name,
        deterministic_decomposer.version,
    ):
        raise ValueError("claim review decomposer provenance does not match the experiment")
    expected = {_key(r.domain, r.example_id, s.key) for r in eligible for s in r.response_sentences}
    reviewed = {_key(x.domain, x.example_id, x.sentence_key): x for x in review.items}
    if set(reviewed) != expected:
        raise ValueError("claim review population does not exactly match eligible records")

    deterministic = run_oracle_evidence_diagnostic(
        eligible, embedding_model, deterministic_decomposer, verifier, threshold,
        bootstrap_iterations, seed,
    )
    a = [x.selected for x in deterministic.predictions]
    b = [x.oracle for x in deterministic.predictions]
    b_pairs = {
        f"{x.domain}:{x.example_id}:{x.sentence_key}": x.oracle_pairs
        for x in deterministic.predictions
    }
    # Sentence keys are only record-local, so run each record with its exact reviewed map.
    c, d = [], []
    d_pairs = {}
    for record in eligible:
        local = {}
        for sentence in record.response_sentences:
            item = reviewed[_key(record.domain, record.example_id, sentence.key)]
            local[sentence.key] = (
                item.deterministic_claims if item.accept_deterministic else item.reviewed_claims
            )
        result = run_oracle_evidence_diagnostic(
            [record], embedding_model, FixtureClaimDecomposer(local), verifier, threshold,
            bootstrap_iterations, seed,
        )
        c.extend(x.selected for x in result.predictions)
        d.extend(x.oracle for x in result.predictions)
        d_pairs.update({
            f"{x.domain}:{x.example_id}:{x.sentence_key}": x.oracle_pairs
            for x in result.predictions
        })
    predictions = {"A": a, "B": b, "C": c, "D": d}
    condition_results = {}
    for name, rows in predictions.items():
        values = [x.predicted_unsupported for x in rows if x.predicted_unsupported is not None]
        condition_results[name] = ConditionResult(
            condition=name, false_unsupported_rate=float(np.mean(values)) if values else None,
            evaluated=len(values), predictions=rows,
            pair_evaluations=b_pairs if name == "B" else d_pairs if name == "D" else {},
        )
    contrasts = {
        "evidence_at_deterministic": {"B": 1, "A": -1},
        "decomposition_at_selected": {"C": 1, "A": -1},
        "evidence_at_reviewed": {"D": 1, "C": -1},
        "decomposition_at_oracle": {"D": 1, "B": -1},
        "interaction": {"D": 1, "C": -1, "B": -1, "A": 1},
    }
    effects = {name: PairedEffect(contrast=name, interval=_interval(predictions, weights, bootstrap_iterations, seed))
               for name, weights in contrasts.items()}
    residual = [f"{x.domain}:{x.example_id}:{x.sentence_key}" for x in d if x.predicted_unsupported is True]
    return DecompositionEvidenceReport(
        schema_version="decomposition-evidence-report.v1", population_size=len(expected),
        metadata=metadata,
        exclusions=dict(excluded), claim_review_sha256=review_sha256,
        conditions=condition_results, effects=effects, residual_condition_d_keys=residual,
    )


def make_residual_review_template(
    report: DecompositionEvidenceReport,
    report_sha256: str,
    records: Sequence[RAGBenchEvaluationRecord],
    review: ClaimReviewArtifact,
) -> ResidualReviewArtifact:
    record_index = {(x.domain, x.example_id): x for x in records}
    review_index = {_key(x.domain, x.example_id, x.sentence_key): x for x in review.items}
    predictions = {
        _key(x.domain, x.example_id, x.sentence_key): x
        for x in report.conditions["D"].predictions
    }
    items = []
    for value in report.residual_condition_d_keys:
        domain, example_id, sentence_key = value.split(":", 2)
        key = _key(domain, example_id, sentence_key)
        record = record_index[(domain, example_id)]
        claim_review = review_index[key]
        outcomes = report.conditions["D"].pair_evaluations[value]
        evidence = []
        seen = set()
        for outcome in outcomes:
            if outcome.evidence and outcome.evidence.sentence_key not in seen:
                seen.add(outcome.evidence.sentence_key)
                evidence.append(outcome.evidence.text)
        items.append(ResidualReviewItem(
            domain=domain,
            example_id=example_id,
            sentence_key=sentence_key,
            question=record.question,
            full_response=record.response,
            sentence=predictions[key].sentence,
            reviewed_claims=(
                claim_review.deterministic_claims
                if claim_review.accept_deterministic
                else claim_review.reviewed_claims
            ),
            annotated_evidence=evidence,
            verifier_outcomes=outcomes,
        ))
    return ResidualReviewArtifact(
        schema_version="decomposition-evidence-residuals.v1",
        reviewer_instructions=(
            "Assign exactly one declared category after inspecting the reviewed claims, annotated "
            "evidence, and verifier scores. Use undetermined when evidence does not distinguish causes."
        ),
        experiment_report_sha256=report_sha256, items=items,
    )
