"""Frozen stratum-eligibility and selection rules for issue #30 (comparative-diagnostics.v1).

These rules implement CASE-SELECTION-PROTOCOL.md steps 2-3 and are documented, with the owner's
decisions behind them, in evals/comparative_diagnostics/v1/STRATUM-RULES.md. They read the
JSON-serialized `SystemDiagnosticRecord`s written by the pool runner, the RAGBench record, and the
oracle-evidence predictions for the case's eligible sentences. Changing any rule after looking at
pool outputs requires a new protocol version, not an edit here.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from benchmark.comparative_diagnostics import BATCH_CRASH_PREFIX

Direction = Literal["retrieval", "generation"]
FLAGGED_SYSTEMS = ("rag_forensics", "ragas_baseline", "ragchecker", "ragvue")

# RAG Forensics' top-ranked signal counts as "flags unsupported content" only for these names.
RAG_FORENSICS_UNSUPPORTED_SIGNALS = frozenset({"low_faithfulness", "unattributed_content", "overconfidence"})
# RAGAS failures RAG Forensics passes through as its own signals; counted once, under ragas_baseline.
RAGAS_PASSTHROUGH_SIGNALS = frozenset({"faithfulness_unavailable", "context_utilization_unavailable"})
RAG_FORENSICS_COMPONENTS: dict[str, Direction] = {"retriever": "retrieval", "answer generator": "generation"}
RAGVUE_LOW_RELEVANCE = 0.5
QUALIFIER_PATTERN = re.compile(
    r"\b(not|no|never|without|except|only|unless|neither|nor|none)\b|\w+n't\b", re.IGNORECASE,
)
QUALIFIER_DOMAINS = frozenset({"finqa"})

SELECTION_SEED = 30
# Narrow strata first, so broad strata cannot use up their cases.
SELECTION_ORDER = (
    "single_system_exposes_failure",
    "counterexample_to_preferred_interpretation",
    "intervention_discriminates_hypotheses",
    "intervention_fails_to_localize",
    "component_diagnoses_disagree",
    "evidence_attributions_disagree",
    "qualifier_negation_numerical_tabular_multisource_or_granularity",
    "systems_agree_labels_contradict",
    "systems_agree_labels_support",
)
STRATUM_TARGETS = {
    "single_system_exposes_failure": 2,
    "counterexample_to_preferred_interpretation": 2,
    "intervention_discriminates_hypotheses": 2,
    "intervention_fails_to_localize": 2,
    "component_diagnoses_disagree": 3,
    "evidence_attributions_disagree": 0,  # declared infeasible: no other system attributes evidence
    "qualifier_negation_numerical_tabular_multisource_or_granularity": 3,
    "systems_agree_labels_contradict": 3,
    "systems_agree_labels_support": 3,
}

Entry = Mapping[str, Mapping[str, Any]]


def _native(entry: Entry, system: str) -> Mapping[str, Any]:
    return entry[system]["native"]


def _healthy_raw(entry: Entry, system: str) -> Any:
    native = _native(entry, system)
    return native["raw_output"] if native["availability"] == "healthy" else None


def _rf_raw(entry: Entry) -> Mapping[str, Any] | None:
    return _native(entry, "rag_forensics")["raw_output"]


def _ragvue_scores(entry: Entry) -> dict[str, float]:
    native = _native(entry, "ragvue")
    if native["availability"] != "healthy" or not native["raw_output"]:
        return {}
    return {
        m["name"]: m["score"] for m in native["raw_output"].get("metrics", [])
        if not m.get("details", {}).get("error") and m.get("score") is not None
    }


def unsupported_flags(entry: Entry) -> dict[str, bool | None]:
    flags: dict[str, bool | None] = dict.fromkeys(FLAGGED_SYSTEMS)
    rf = _healthy_raw(entry, "rag_forensics")
    if rf and rf.get("verdict_signals"):
        flags["rag_forensics"] = rf["verdict_signals"][0]["name"] in RAG_FORENSICS_UNSUPPORTED_SIGNALS
    raw = _rf_raw(entry)
    faith = (raw or {}).get("ragas", {}).get("faithfulness", {})
    if faith.get("status") == "ok" and faith.get("score") is not None:
        flags["ragas_baseline"] = faith["score"] < 1.0
    rc = _healthy_raw(entry, "ragchecker")
    if rc and rc.get("faithfulness") is not None:
        flags["ragchecker"] = rc["faithfulness"] < 1.0
    strict = _ragvue_scores(entry).get("strict_faithfulness")
    if strict is not None:
        flags["ragvue"] = strict < 1.0
    return flags


def rag_forensics_direction(entry: Entry) -> Direction | None:
    test = ((_rf_raw(entry) or {}).get("verdict_reasoning") or {}).get("test") or {}
    return RAG_FORENSICS_COMPONENTS.get(test.get("component", ""))


def ragvue_direction(entry: Entry) -> Direction | None:
    scores = _ragvue_scores(entry)
    if scores.get("strict_faithfulness") is not None and scores["strict_faithfulness"] < 1.0:
        return "generation"
    if scores.get("retrieval_relevance") is not None and scores["retrieval_relevance"] < RAGVUE_LOW_RELEVANCE:
        return "retrieval"
    return None


def failure_exposures(entry: Entry) -> set[str]:
    exposed = {s for s in ("rag_forensics", "ragchecker", "ragvue") if _native(entry, s)["availability"] != "healthy"}
    raw = _rf_raw(entry) or {}
    if any(
        sig["name"].endswith("_unavailable") and sig["name"] not in RAGAS_PASSTHROUGH_SIGNALS
        for sig in raw.get("verdict_signals", [])
    ):
        exposed.add("rag_forensics")
    ragas = raw.get("ragas", {})
    if raw and any(ragas.get(m, {}).get("status") != "ok" for m in ("faithfulness", "context_utilization")):
        exposed.add("ragas_baseline")
    return exposed


def intervention_outcomes(predictions: Iterable[Any]) -> set[str]:
    outcomes = set()
    for p in predictions:
        if p.selected.predicted_unsupported is True and p.oracle.predicted_unsupported is False:
            outcomes.add("intervention_discriminates_hypotheses")
        if p.selected.predicted_unsupported is True and p.oracle.predicted_unsupported is True:
            outcomes.add("intervention_fails_to_localize")
    return outcomes


def exclusion_reason(entry: Entry) -> str | None:
    for system, record in entry.items():
        if (record["native"].get("error") or "").startswith(BATCH_CRASH_PREFIX):
            return f"{system} batch crashed and was not rerun"
    if _native(entry, "rag_forensics")["availability"] == "failed":
        return "rag_forensics produced no output, so neither it nor the RAGAS baseline can be compared"
    return None


def eligible_strata(entry: Entry, record: Any, oracle_predictions: Iterable[Any]) -> set[str]:
    strata = set(intervention_outcomes(oracle_predictions))
    label_unsupported = bool(record.unsupported_response_sentence_keys)
    flags = {s: f for s, f in unsupported_flags(entry).items() if f is not None}
    if len(flags) >= 2 and len(set(flags.values())) == 1:
        agrees = next(iter(flags.values())) == label_unsupported
        strata.add("systems_agree_labels_support" if agrees else "systems_agree_labels_contradict")
    rf_flag = flags.get("rag_forensics")
    others = [f for s, f in flags.items() if s != "rag_forensics"]
    if rf_flag is not None and rf_flag != label_unsupported and others and all(f == label_unsupported for f in others):
        strata.add("counterexample_to_preferred_interpretation")
    rf_dir, rv_dir = rag_forensics_direction(entry), ragvue_direction(entry)
    if rf_dir and rv_dir and rf_dir != rv_dir:
        strata.add("component_diagnoses_disagree")
    if len(failure_exposures(entry)) == 1:
        strata.add("single_system_exposes_failure")
    if record.domain in QUALIFIER_DOMAINS or QUALIFIER_PATTERN.search(record.response):
        strata.add("qualifier_negation_numerical_tabular_multisource_or_granularity")
    return strata


def select_cases(eligibility: Mapping[str, set[str]], seed: int) -> list[tuple[str, str]]:
    """Seeded draw per stratum in SELECTION_ORDER; each case is selected into at most one stratum."""
    rng = random.Random(seed)
    taken: set[str] = set()
    selected: list[tuple[str, str]] = []
    for stratum in SELECTION_ORDER:
        pool = sorted(c for c, strata in eligibility.items() if stratum in strata and c not in taken)
        for case_id in rng.sample(pool, min(STRATUM_TARGETS[stratum], len(pool))):
            taken.add(case_id)
            selected.append((case_id, stratum))
    return selected
