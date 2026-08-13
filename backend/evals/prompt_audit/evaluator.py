"""Load, render, and deterministically score prompt-audit cases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from models import RetrievedChunk
from prompts.generation_prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt
from prompts.hedging_prompts import CLAIM_EXTRACTION_PROMPT, ENTAILMENT_PROMPT
from prompts.query_fit_prompts import build_question_generation_prompt
from prompts.verdict_prompts import RANKED_SIGNALS_PROMPT
from services.verdict_generator import RankedSignal, build_verdict_reasoning, reasoning_payload

DATASET_PATH = Path(__file__).parent / "v1" / "cases.json"
Split = Literal["development", "held_out"]
ExecutionMode = Literal["production_path", "counterfactual_capability"]


@dataclass(frozen=True)
class RenderedCase:
    case: dict[str, Any]
    prompt: str
    system_prompt: str | None


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    """Load the versioned dataset and reject malformed or duplicate cases."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != "prompt-eval.v1":
        raise ValueError("Unsupported prompt evaluation schema")
    if not isinstance(dataset.get("dataset_version"), str):
        raise ValueError("dataset_version must be a string")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise ValueError("every case must have a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    for case in cases:
        if case.get("split") not in {"development", "held_out"}:
            raise ValueError(f"invalid split for {case['id']}")
        if case.get("execution_mode") not in {
            "production_path",
            "counterfactual_capability",
        }:
            raise ValueError(f"invalid execution_mode for {case['id']}")
        if case.get("boundary") not in {
            "generation",
            "claim_extraction",
            "entailment",
            "query_fit",
            "verdict_rendering",
        }:
            raise ValueError(f"invalid boundary for {case['id']}")
        expectations = case.get("expectations")
        if not isinstance(expectations, dict) or not expectations.get("scorers"):
            raise ValueError(f"case {case['id']} needs deterministic scorers")
        if not expectations.get("human_review"):
            raise ValueError(f"case {case['id']} needs human-review criteria")
    return dataset


def select_cases(
    dataset: dict[str, Any],
    *,
    split: Split | Literal["all"] = "development",
    execution_mode: ExecutionMode | Literal["all"] = "production_path",
    case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select development cases by default so held-out cases are deliberate."""
    cases = dataset["cases"]
    if case_ids is not None:
        known = {case["id"] for case in cases}
        missing = sorted(case_ids - known)
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(missing)}")
        cases = [case for case in cases if case["id"] in case_ids]
    if split != "all":
        cases = [case for case in cases if case["split"] == split]
    if execution_mode != "all":
        cases = [case for case in cases if case["execution_mode"] == execution_mode]
    return cases


def render_case(case: dict[str, Any]) -> RenderedCase:
    """Render a dataset case through the production prompt builder/constant."""
    inputs = case["inputs"]
    boundary = case["boundary"]
    system_prompt: str | None = None
    if boundary == "generation":
        chunks = [RetrievedChunk(**chunk) for chunk in inputs["chunks"]]
        prompt = build_generation_prompt(inputs["question"], chunks)
        system_prompt = GENERATION_SYSTEM_PROMPT
    elif boundary == "claim_extraction":
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=inputs["answer"])
    elif boundary == "entailment":
        prompt = ENTAILMENT_PROMPT.format(
            chunk_text=inputs["context"], claim=inputs["claim"]
        )
    elif boundary == "query_fit":
        prompt = build_question_generation_prompt(
            inputs["chunks_text"], inputs["original_question"]
        )
    elif boundary == "verdict_rendering":
        # Preserve frozen v1 inputs and adapt their ranked-signal text to the current
        # deterministic reasoning boundary. The dataset itself remains unchanged.
        signal_names = {
            "no semantically close source": "unattributed_content",
            "Faithfulness score": "low_faithfulness",
            "geometrically distant": "query_isolation",
            "definitively but unsupported": "overconfidence",
            "Hedging analysis is unavailable": "hedging_analysis_unavailable",
            "Retrieval relevance score": "low_context_utilization",
        }
        signals = []
        for line in inputs["signals_text"].splitlines():
            description = re.sub(r"^\d+\.\s*", "", line)
            metadata = re.search(
                r"\s*\(heuristic priority: ([0-9.]+); reliability: ([a-z_]+)\)$",
                description,
            )
            if metadata is None:
                raise ValueError(f"invalid frozen verdict signal: {line}")
            description = description[:metadata.start()]
            name = next((value for key, value in signal_names.items() if key in description), None)
            if name is None:
                raise ValueError(f"unmapped frozen verdict signal: {line}")
            signals.append(RankedSignal(name, float(metadata.group(1)), description, metadata.group(2)))
        reasoning = build_verdict_reasoning(signals)
        prompt = RANKED_SIGNALS_PROMPT.format(
            reasoning_json=json.dumps(reasoning_payload(reasoning), indent=2)
        )
    else:  # Protected by load_dataset; retained for direct callers.
        raise ValueError(f"unsupported boundary: {boundary}")
    return RenderedCase(case=case, prompt=prompt, system_prompt=system_prompt)


def _strip_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _json_strings(response: str) -> tuple[list[str] | None, str | None]:
    try:
        parsed = json.loads(_strip_fence(response))
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None, "expected a JSON array containing only strings"
    return parsed, None


def _sentence_count(response: str) -> int:
    return len(re.findall(r"[^.!?]+(?:[.!?]+|$)", response.strip())) if response.strip() else 0


def score_response(case: dict[str, Any], response: str) -> dict[str, Any]:
    """Apply declared deterministic scorers; do not infer semantic correctness."""
    results: list[dict[str, Any]] = []
    lower = response.lower()
    cached_items: list[str] | None = None
    cached_error: str | None = None

    for spec in case["expectations"]["scorers"]:
        scorer = spec["type"]
        passed = False
        detail = ""
        if scorer == "non_empty":
            passed = bool(response.strip())
            detail = "response is non-empty" if passed else "response is empty"
        elif scorer == "contains_all_ci":
            missing = [value for value in spec["values"] if value.lower() not in lower]
            passed = not missing
            detail = "all required strings found" if passed else f"missing: {missing}"
        elif scorer == "contains_any_ci":
            found = [value for value in spec["values"] if value.lower() in lower]
            passed = bool(found)
            detail = f"matched: {found}" if passed else "none of the accepted strings found"
        elif scorer == "excludes_all_ci":
            found = [value for value in spec["values"] if value.lower() in lower]
            passed = not found
            detail = "no forbidden strings found" if passed else f"forbidden: {found}"
        elif scorer == "exact_normalized":
            actual = _strip_fence(response).strip().lower().rstrip(".!")
            expected = spec["value"].strip().lower()
            passed = actual == expected
            detail = f"expected {expected!r}; got {actual!r}"
        elif scorer == "max_words":
            count = len(response.split())
            passed = count <= spec["value"]
            detail = f"{count} words; maximum {spec['value']}"
        elif scorer == "sentence_count":
            count = _sentence_count(response)
            passed = spec["min"] <= count <= spec["max"]
            detail = f"{count} sentences; expected {spec['min']}–{spec['max']}"
        elif scorer.startswith("json_") or scorer == "unique_json_items":
            if cached_items is None and cached_error is None:
                cached_items, cached_error = _json_strings(response)
            if cached_error:
                passed = False
                detail = cached_error
            else:
                assert cached_items is not None
                joined = "\n".join(cached_items).lower()
                if scorer == "json_array_strings":
                    minimum = spec["min_items"]
                    maximum = spec["max_items"]
                    passed = minimum <= len(cached_items) <= maximum
                    detail = f"{len(cached_items)} string items; expected {minimum}–{maximum}"
                elif scorer == "unique_json_items":
                    normalized = [item.strip().lower() for item in cached_items]
                    passed = len(normalized) == len(set(normalized))
                    detail = "items are unique" if passed else "duplicate items found"
                elif scorer == "json_items_contain_all_ci":
                    missing = [value for value in spec["values"] if value.lower() not in joined]
                    passed = not missing
                    detail = "all required strings found" if passed else f"missing: {missing}"
                elif scorer == "json_items_contain_any_ci":
                    found = [value for value in spec["values"] if value.lower() in joined]
                    passed = bool(found)
                    detail = f"matched: {found}" if passed else "none of the accepted strings found"
                elif scorer == "json_items_exclude_all_ci":
                    found = [value for value in spec["values"] if value.lower() in joined]
                    passed = not found
                    detail = "no forbidden strings found" if passed else f"forbidden: {found}"
                else:
                    raise ValueError(f"unsupported scorer: {scorer}")
        else:
            raise ValueError(f"unsupported scorer: {scorer}")
        results.append({"scorer": scorer, "passed": passed, "detail": detail})

    return {
        "case_id": case["id"],
        "passed": all(result["passed"] for result in results),
        "score": sum(result["passed"] for result in results) / len(results),
        "scorer_results": results,
        "human_review": case["expectations"]["human_review"],
    }


def score_records(dataset: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score saved runner records and report aggregate results without hiding failures."""
    cases = {case["id"]: case for case in dataset["cases"]}
    scored = []
    for record in records:
        case_id = record["case_id"]
        if case_id not in cases:
            raise ValueError(f"record references unknown case: {case_id}")
        result = score_response(cases[case_id], record.get("response", ""))
        result["returncode"] = record.get("returncode")
        result["execution_succeeded"] = record.get("returncode") == 0
        if not result["execution_succeeded"]:
            result["passed"] = False
        scored.append(result)
    passed = sum(result["passed"] for result in scored)
    return {
        "dataset_version": dataset["dataset_version"],
        "record_count": len(scored),
        "passed_count": passed,
        "pass_rate": passed / len(scored) if scored else None,
        "results": scored,
    }


def _records_by_case(
    dataset: dict[str, Any], records: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    """Validate a comparison side and index its one record per case."""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("dataset_version") != dataset["dataset_version"]:
            raise ValueError(
                f"{label} record {record.get('case_id')} uses dataset version "
                f"{record.get('dataset_version')!r}, expected {dataset['dataset_version']!r}"
            )
        case_id = record.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError(f"{label} record is missing case_id")
        if case_id in indexed:
            raise ValueError(f"{label} contains duplicate case_id: {case_id}")
        indexed[case_id] = record
    return indexed


def compare_record_sets(
    dataset: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce a paired deterministic comparison over exactly matching case IDs."""
    baseline = _records_by_case(dataset, baseline_records, "baseline")
    candidate = _records_by_case(dataset, candidate_records, "candidate")
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "paired comparison requires identical case IDs; "
            f"missing from candidate={missing_candidate}, missing from baseline={missing_baseline}"
        )

    case_order = {case["id"]: index for index, case in enumerate(dataset["cases"])}
    rows = []
    transitions = {
        "improved": 0,
        "regressed": 0,
        "unchanged_pass": 0,
        "unchanged_fail": 0,
    }
    for case_id in sorted(baseline, key=lambda value: case_order[value]):
        baseline_result = score_records(dataset, [baseline[case_id]])["results"][0]
        candidate_result = score_records(dataset, [candidate[case_id]])["results"][0]
        if not baseline_result["passed"] and candidate_result["passed"]:
            transition = "improved"
        elif baseline_result["passed"] and not candidate_result["passed"]:
            transition = "regressed"
        elif baseline_result["passed"]:
            transition = "unchanged_pass"
        else:
            transition = "unchanged_fail"
        transitions[transition] += 1
        baseline_scorers = {
            item["scorer"]: item["passed"] for item in baseline_result["scorer_results"]
        }
        candidate_scorers = {
            item["scorer"]: item["passed"] for item in candidate_result["scorer_results"]
        }
        rows.append(
            {
                "case_id": case_id,
                "baseline_passed": baseline_result["passed"],
                "candidate_passed": candidate_result["passed"],
                "transition": transition,
                "changed_scorers": [
                    scorer
                    for scorer in baseline_scorers
                    if baseline_scorers[scorer] != candidate_scorers[scorer]
                ],
            }
        )
    return {
        "schema_version": "prompt-eval-comparison.v1",
        "dataset_version": dataset["dataset_version"],
        "case_count": len(rows),
        "transitions": transitions,
        "rows": rows,
        "interpretation": "Deterministic contract comparison only; human semantic review is separate.",
    }
