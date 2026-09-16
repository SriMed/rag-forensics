"""Small, real, live pilot for issue #30 — proves the comparative pipeline end to end.

This is deliberately scoped down from the issue's 12-20 case, multi-stratum target, the same way
issue #29 scoped its full three-domain population down to a TechQA-only pilot: it runs RAG
Forensics, RAGChecker (faithfulness only — see README.md), and RAGVue against a handful of live
TechQA test examples, freezes the result as a ComparativeCaseSet, and reports what genuinely ran.
It is not the frozen 12-20 case, multi-stratum selection CASE-SELECTION-PROTOCOL.md describes; that
requires running this same pipeline over the full candidate pool and applying the seeded
per-stratum draw, which is follow-up work.

Usage (from backend/):
    poetry run python evals/comparative_diagnostics/v1/run_pilot.py \
        --limit 3 --reviewer-identity you@example.com \
        --output evals/comparative_diagnostics/v1/pilot-results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # backend/

from dotenv import load_dotenv

load_dotenv()

import benchmark.comparative_diagnostics as cd
from benchmark.experiment_cli import DATASET_REVISION, _load_records
from models import CustomAnalyzeRequest, CustomChunk

ENVS = Path(__file__).parent / "envs"
DRIVERS = Path(__file__).parent / "drivers"
RAGCHECKER_MODEL = "anthropic/claude-haiku-4-5-20251001"
RAGVUE_MODEL = "claude-haiku-4-5-20251001"


def run_rag_forensics(records) -> dict[str, cd.SystemDiagnosticRecord]:
    from routers.analyze import analyze_custom

    out = {}
    for record in records:
        custom_chunks = [
            CustomChunk(chunk_id=c.chunk_id, text=c.text, score=c.score) for c in record.chunks
        ]
        request = CustomAnalyzeRequest(
            question=record.question, answer=record.response, chunks=custom_chunks,
            score_semantics="normalized_similarity",
        )
        try:
            response = analyze_custom(request)
            out[record.example_id] = cd.map_rag_forensics_native(
                case_id=record.example_id,
                verdict_signals=response.verdict_signals,
                raw_output=response.model_dump(),
            )
        except Exception as exc:
            out[record.example_id] = cd.SystemDiagnosticRecord(
                system="rag_forensics",
                native=cd.NativeSystemOutput(
                    system="rag_forensics", system_version="0.1.0", availability="failed",
                    raw_output=None, error=f"{type(exc).__name__}: {exc}",
                ),
                suspected_component=None, supporting_observation=None, evidence_attribution=[],
                method=None, reliability=None, causal_strength_language=None,
                proposed_intervention=None, no_equivalent_fields=[],
            )
    return out


def run_ragchecker(records) -> dict[str, cd.SystemDiagnosticRecord]:
    payload = {"results": [cd.ragchecker_result_input(r) for r in records]}
    in_path = ENVS / "_ragchecker_in.json"
    out_path = ENVS / "_ragchecker_out.json"
    in_path.write_text(json.dumps(payload))
    subprocess.run(
        [
            str(ENVS / ".venv-ragchecker/bin/python"), str(DRIVERS / "run_ragchecker.py"),
            "--input", str(in_path), "--output", str(out_path), "--model", RAGCHECKER_MODEL,
        ],
        check=True,
    )
    raw = json.loads(out_path.read_text())
    out = {}
    for record in records:
        if raw.get("error"):
            out[record.example_id] = cd.map_ragchecker_native(
                case_id=record.example_id, metrics_for_item=None,
                requested_metrics=["faithfulness"], error=raw["error"],
            )
        else:
            out[record.example_id] = cd.map_ragchecker_native(
                case_id=record.example_id,
                metrics_for_item=raw["per_item"].get(record.example_id),
                requested_metrics=["faithfulness"],
            )
    return out


def run_ragvue(records) -> dict[str, cd.SystemDiagnosticRecord]:
    import os

    payload = [
        {"case_id": r.example_id, **cd.ragvue_item_from_record(r)} for r in records
    ]
    in_path = ENVS / "_ragvue_in.json"
    out_path = ENVS / "_ragvue_out.json"
    in_path.write_text(json.dumps(payload))
    env = {**os.environ, "RAGVUE_JUDGE_PROVIDER": "anthropic", "RAGVUE_ANTHROPIC_MODEL": RAGVUE_MODEL}
    subprocess.run(
        [
            str(ENVS / ".venv-ragvue/bin/python"), str(DRIVERS / "run_ragvue.py"),
            "--input", str(in_path), "--output", str(out_path),
        ],
        check=True, env=env,
    )
    raw = json.loads(out_path.read_text())
    configured_model = raw.get("configured_model", f"anthropic:{RAGVUE_MODEL}")
    out = {}
    for record in records:
        if raw.get("error"):
            out[record.example_id] = cd.map_ragvue_native(
                case_id=record.example_id, raw_result={}, configured_model=configured_model,
                error=raw["error"],
            )
        else:
            out[record.example_id] = cd.map_ragvue_native(
                case_id=record.example_id,
                raw_result=raw["per_item"].get(record.example_id, {}),
                configured_model=configured_model,
            )
    return out


def infer_stratum(systems: dict[str, cd.SystemDiagnosticRecord]) -> str:
    availabilities = {name: r.native.availability for name, r in systems.items()}
    if any(a in ("failed", "missing", "unavailable") for a in availabilities.values()) and any(
        a == "healthy" for a in availabilities.values()
    ):
        return "single_system_exposes_failure"
    return "systems_agree_labels_support"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--reviewer-identity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records, skipped = _load_records(["techqa"], "test", args.limit, 42, DATASET_REVISION)
    if skipped:
        print(f"skipped rows: {skipped}", file=sys.stderr)

    rag_forensics = run_rag_forensics(records)
    ragchecker = run_ragchecker(records)
    ragvue = run_ragvue(records)

    cases = []
    for record in records:
        systems = {
            "rag_forensics": rag_forensics[record.example_id],
            "ragchecker": ragchecker[record.example_id],
            "ragvue": ragvue[record.example_id],
        }
        cases.append(
            cd.ComparativeCase(
                case_id=record.example_id,
                dataset="ragbench",
                domain=record.domain,
                dataset_revision=f"galileo-ai/ragbench@{DATASET_REVISION}",
                stratum=infer_stratum(systems),
                selection_rationale="Pilot: first N shuffled techqa test examples, not stratified selection.",
                systems=systems,
                judgments=cd.CaseJudgments(
                    dataset_label=cd.dataset_label_for_record(record),
                    model_judgments={},
                    reviewer_judgment=None,
                    intervention_evidence=None,
                ),
            )
        )

    case_set = cd.ComparativeCaseSet(
        schema_version="comparative-diagnostics.v1",
        status="frozen",
        reviewer_identity=args.reviewer_identity,
        population_sha256=cd.make_population_sha256([c.case_id for c in cases]),
        cases=cases,
    )
    Path(args.output).write_text(case_set.model_dump_json(indent=2))
    print(f"wrote {len(cases)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
