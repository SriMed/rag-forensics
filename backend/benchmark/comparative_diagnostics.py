"""Provenance-preserving comparison schema for issue #30.

This module defines, but does not populate, the frozen shared schema for comparing
RAG Forensics against RAGChecker, RAGVue, and a RAGAS baseline on the same public
cases. Native per-system outputs are always preserved verbatim; mapped fields are a
best-effort projection into a common vocabulary and must never force an unlike
construct into a shared score. Dataset labels, model judgments, reviewer judgments,
and intervention evidence are kept as separate fields so a reader can tell which
kind of claim they are looking at.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmark.experiment_cli import DATASET_REVISION, DOMAINS, _load_records
from benchmark.oracle_evidence import _eligible_records
from models import RAGBenchEvaluationRecord, VerdictSignal

# The candidate pool for issue #30 is, by decision, the distinct parent examples of issue #29's
# 188-eligible-sentence population — not a fresh sample. `_eligible_records` already deduplicates
# to one entry per example that has at least one eligible sentence (see
# `benchmark.decomposition_evidence.make_claim_review_template`, which iterates the same
# `eligible` list's `response_sentences` to reach 188 sentence-level review items from a smaller
# number of distinct examples). These parameters must match #29's population exactly, or the pool
# is a different population under the same name.
CANDIDATE_POOL_DOMAINS = DOMAINS
CANDIDATE_POOL_SPLIT = "test"
CANDIDATE_POOL_LIMIT = 100
CANDIDATE_POOL_SEED = 42
CANDIDATE_POOL_REVISION = DATASET_REVISION


def load_case_candidate_pool() -> list[RAGBenchEvaluationRecord]:
    records, _skipped = _load_records(
        list(CANDIDATE_POOL_DOMAINS), CANDIDATE_POOL_SPLIT, CANDIDATE_POOL_LIMIT,
        CANDIDATE_POOL_SEED, CANDIDATE_POOL_REVISION,
    )
    eligible, _total, _excluded = _eligible_records(records)
    return eligible

SystemName = Literal["rag_forensics", "ragchecker", "ragvue", "ragas_baseline"]

# An observation can be present and healthy, absent from the framework's coverage
# (missing), present in scope but not returned this run (unavailable, e.g. an
# evaluator error), or attempted and returned an explicit failure (failed).
AvailabilityState = Literal["healthy", "missing", "unavailable", "failed"]

DECLARED_STRATA = (
    "systems_agree_labels_support",
    "systems_agree_labels_contradict",
    "component_diagnoses_disagree",
    "evidence_attributions_disagree",
    "single_system_exposes_failure",
    "intervention_discriminates_hypotheses",
    "intervention_fails_to_localize",
    "qualifier_negation_numerical_tabular_multisource_or_granularity",
    "counterexample_to_preferred_interpretation",
)


class NativeSystemOutput(BaseModel):
    """A system's own output, unmodified, plus its provenance and run state."""

    model_config = ConfigDict(frozen=True)

    system: SystemName
    system_version: str
    availability: AvailabilityState
    raw_output: dict | list | None
    error: str | None = None

    @model_validator(mode="after")
    def validate_error_matches_availability(self):
        if self.availability == "failed" and not self.error:
            raise ValueError("a failed native output requires an error message")
        if self.availability == "healthy" and self.error:
            raise ValueError("a healthy native output must not carry an error message")
        return self


class SystemDiagnosticRecord(BaseModel):
    """One system's native output plus its projection into the shared vocabulary.

    Mapped fields are optional: a system that does not produce an equivalent
    construct leaves the field ``None`` and names the gap in
    ``no_equivalent_fields`` rather than approximating it.
    """

    model_config = ConfigDict(frozen=True)

    system: SystemName
    native: NativeSystemOutput
    suspected_component: str | None
    supporting_observation: str | None
    evidence_attribution: list[str] = Field(default_factory=list)
    method: str | None
    reliability: Literal["unvalidated", "partially_calibrated", "model_judged"] | None
    causal_strength_language: str | None
    proposed_intervention: str | None
    no_equivalent_fields: list[str] = Field(default_factory=list)


class CaseJudgments(BaseModel):
    """Kept separate on purpose: a dataset label is not a model judgment, a model
    judgment is not a reviewer judgment, and none of those is evidence that an
    intervention actually discriminated between hypotheses.
    """

    model_config = ConfigDict(frozen=True)

    dataset_label: str | None
    model_judgments: dict[SystemName, str] = Field(default_factory=dict)
    reviewer_judgment: str | None = None
    intervention_evidence: str | None = None


class ComparativeCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    dataset: Literal["ragbench", "ragtruth"]
    domain: str
    dataset_revision: str
    stratum: Literal[DECLARED_STRATA]  # type: ignore[valid-type]  # a tuple of literals; pydantic resolves it
    selection_rationale: str
    systems: dict[SystemName, SystemDiagnosticRecord]
    judgments: CaseJudgments


class ComparativeCaseSet(BaseModel):
    schema_version: Literal["comparative-diagnostics.v1"]
    status: Literal["draft", "frozen"] = "draft"
    reviewer_identity: str | None = None
    population_sha256: str
    cases: list[ComparativeCase]

    @model_validator(mode="after")
    def validate_case_set(self):
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("comparative case set contains duplicate case_id values")
        if self.status == "frozen":
            if not self.reviewer_identity:
                raise ValueError("a frozen case set requires reviewer_identity")
            if not self.cases:
                raise ValueError("a frozen case set requires at least one case")
        return self


# Every RAGChecker metric except `faithfulness` is computed from `answer2response`,
# `response2answer`, or `retrieved2answer` (see ragchecker.metrics.METRIC_REQUIREMENTS in the
# installed 0.1.9 package), all of which are derived from `gt_answer`. RAGBench's adapted schema
# (`models.RAGBenchEvaluationRecord`) has no reference answer distinct from the response being
# evaluated, so only `faithfulness` can be computed on this project's committed populations without
# fabricating a gt_answer RAGChecker was not designed to receive.
RAGCHECKER_METRICS_REQUIRING_GT_ANSWER = frozenset(
    {
        "precision",
        "recall",
        "f1",
        "claim_recall",
        "context_precision",
        "context_utilization",
        "noise_sensitivity_in_relevant",
        "noise_sensitivity_in_irrelevant",
        "hallucination",
        "self_knowledge",
    }
)


def assert_ragchecker_metrics_computable(metrics: list[str], *, has_gt_answer: bool) -> None:
    if has_gt_answer:
        return
    blocked = sorted(set(metrics) & RAGCHECKER_METRICS_REQUIRING_GT_ANSWER)
    if blocked:
        raise ValueError(
            f"metrics {blocked} require gt_answer, which this population does not provide; "
            "only 'faithfulness' is computable without a reference answer"
        )


def dataset_label_for_record(
    record: RAGBenchEvaluationRecord,
) -> Literal["fully_supported", "contains_unsupported"]:
    return "contains_unsupported" if record.unsupported_response_sentence_keys else "fully_supported"


def ragchecker_result_input(record: RAGBenchEvaluationRecord) -> dict:
    """Map a record into RAGChecker's native `RAGResult` shape, unmodified in intent.

    `gt_answer` is left as an empty string rather than populated from `response` or any chunk:
    inventing a reference answer would let RAGChecker compute metrics under a false premise. Callers
    must restrict `metrics` to those in `assert_ragchecker_metrics_computable`'s allowance for an
    empty `gt_answer`.
    """
    return {
        "query_id": record.example_id,
        "query": record.question,
        "gt_answer": "",
        "response": record.response,
        "retrieved_context": [{"doc_id": chunk.chunk_id, "text": chunk.text} for chunk in record.chunks],
    }


def ragvue_item_from_record(record: RAGBenchEvaluationRecord) -> dict:
    return {
        "question": record.question,
        "answer": record.response,
        "contexts": [chunk.text for chunk in record.chunks],
    }


_HEURISTIC_PRIORITY_CAVEAT = (
    "RAG Forensics reports a heuristic priority ranking, not a calibrated severity or causal proof."
)


def map_rag_forensics_native(
    *, case_id: str, verdict_signals: list[VerdictSignal], raw_output: dict
) -> SystemDiagnosticRecord:
    if not verdict_signals:
        return SystemDiagnosticRecord(
            system="rag_forensics",
            native=NativeSystemOutput(
                system="rag_forensics",
                system_version="0.1.0",
                availability="unavailable",
                raw_output=raw_output,
                error=None,
            ),
            suspected_component=None,
            supporting_observation=None,
            evidence_attribution=[],
            method=None,
            reliability=None,
            causal_strength_language=None,
            proposed_intervention=None,
            no_equivalent_fields=["no ranked verdict signals were produced for this case"],
        )
    top = verdict_signals[0]
    return SystemDiagnosticRecord(
        system="rag_forensics",
        native=NativeSystemOutput(
            system="rag_forensics",
            system_version="0.1.0",
            availability="healthy",
            raw_output=raw_output,
            error=None,
        ),
        suspected_component=top.name,
        supporting_observation=top.description,
        evidence_attribution=[],
        method="verdict_generator",
        reliability=top.reliability,
        causal_strength_language=_HEURISTIC_PRIORITY_CAVEAT,
        proposed_intervention=None,
        no_equivalent_fields=[],
    )


def map_ragchecker_native(
    *,
    case_id: str,
    metrics_for_item: dict | None,
    requested_metrics: list[str],
    error: str | None = None,
) -> SystemDiagnosticRecord:
    if error is not None or metrics_for_item is None:
        return SystemDiagnosticRecord(
            system="ragchecker",
            native=NativeSystemOutput(
                system="ragchecker",
                system_version="0.1.9",
                availability="failed",
                raw_output=None,
                error=error or "no metrics returned",
            ),
            suspected_component=None,
            supporting_observation=None,
            evidence_attribution=[],
            method=None,
            reliability=None,
            causal_strength_language=None,
            proposed_intervention=None,
            no_equivalent_fields=[],
        )
    no_equivalent = [
        "RAGChecker's retriever/generator diagnostics beyond faithfulness "
        f"({sorted(RAGCHECKER_METRICS_REQUIRING_GT_ANSWER)}) require gt_answer, "
        "which this population does not provide"
    ]
    faithfulness = metrics_for_item.get("faithfulness")
    return SystemDiagnosticRecord(
        system="ragchecker",
        native=NativeSystemOutput(
            system="ragchecker",
            system_version="0.1.9",
            availability="healthy",
            raw_output=metrics_for_item,
            error=None,
        ),
        suspected_component=None,
        supporting_observation=(
            f"faithfulness={faithfulness}" if faithfulness is not None else None
        ),
        evidence_attribution=[],
        method="ragchecker_faithfulness" if faithfulness is not None else None,
        reliability="model_judged" if faithfulness is not None else None,
        causal_strength_language=None,
        proposed_intervention=None,
        no_equivalent_fields=no_equivalent,
    )


def map_ragvue_native(
    *,
    case_id: str,
    raw_result: dict,
    configured_model: str,
    error: str | None = None,
) -> SystemDiagnosticRecord:
    if error is not None:
        return SystemDiagnosticRecord(
            system="ragvue",
            native=NativeSystemOutput(
                system="ragvue",
                system_version="0.6.1",
                availability="failed",
                raw_output=raw_result,
                error=error,
            ),
            suspected_component=None,
            supporting_observation=None,
            evidence_attribution=[],
            method=None,
            reliability=None,
            causal_strength_language=None,
            proposed_intervention=None,
            no_equivalent_fields=[],
        )
    metrics = raw_result.get("metrics", [])
    # RAGVue can return score 0.0 with an embedded per-metric "error" instead of raising (observed
    # live: an anthropic-SDK version break surfaced this way rather than as an exception) — that is
    # exactly the "healthy-looking zero" this project's own README argues a diagnostic must not
    # produce, so a metric-level error is never folded into a numeric score here.
    healthy_metrics = [m for m in metrics if not m.get("details", {}).get("error")]
    failed_metrics = [m for m in metrics if m.get("details", {}).get("error")]
    no_equivalent_fields = [
        "RAGVue's self-reported judge model in native.raw_output may be incorrect; "
        f"{configured_model} is what this run actually configured"
    ]
    for m in failed_metrics:
        no_equivalent_fields.append(f"{m['name']} failed: {m['details']['error']}")
    availability: Literal["healthy", "unavailable"] = "unavailable" if failed_metrics else "healthy"
    supporting_observation = (
        "; ".join(f"{m['name']}={m['score']}" for m in healthy_metrics) or None
    )
    return SystemDiagnosticRecord(
        system="ragvue",
        native=NativeSystemOutput(
            system="ragvue",
            system_version="0.6.1",
            availability=availability,
            raw_output=raw_result,
            error=None,
        ),
        suspected_component=None,
        supporting_observation=supporting_observation,
        evidence_attribution=[],
        method=configured_model,
        reliability="model_judged" if healthy_metrics else None,
        causal_strength_language=None,
        proposed_intervention=None,
        no_equivalent_fields=no_equivalent_fields,
    )


def make_population_sha256(case_ids: list[str]) -> str:
    """A deterministic, order-sensitive fingerprint of a case-id sequence.

    Order-sensitivity is intentional: selection order can itself carry information
    about how a purposive sample was assembled, and silently normalizing it would
    hide that.
    """
    payload = json.dumps(list(case_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
