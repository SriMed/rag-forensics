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
    stratum: Literal[DECLARED_STRATA]
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


def make_population_sha256(case_ids: list[str]) -> str:
    """A deterministic, order-sensitive fingerprint of a case-id sequence.

    Order-sensitivity is intentional: selection order can itself carry information
    about how a purposive sample was assembled, and silently normalizing it would
    hide that.
    """
    payload = json.dumps(list(case_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
