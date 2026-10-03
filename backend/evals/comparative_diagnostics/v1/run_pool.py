"""Run every feasible system over the full issue #30 candidate pool (CASE-SELECTION-PROTOCOL.md step 1).

This records native and mapped outputs for all candidate-pool cases before any stratum is assigned;
it does not select cases. Systems run in batches and the output is saved after each batch, so an
interrupted run resumes where it stopped. A batch whose runner crashes (e.g. a driver subprocess
exiting non-zero) is recorded as `failed` with a BATCH_CRASH_PREFIX error and retried on the next
invocation; failures a system reports itself are data for the strata and are never retried.

Usage (from backend/):
    poetry run python evals/comparative_diagnostics/v1/run_pool.py \
        --output evals/comparative_diagnostics/v1/pool-run.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # backend/

import benchmark.comparative_diagnostics as cd

BATCH_CRASH_PREFIX = "runner batch failed: "

Runner = Callable[[list], Mapping[str, cd.SystemDiagnosticRecord]]


def _crashed(system: cd.SystemName, error: str) -> cd.SystemDiagnosticRecord:
    return cd.SystemDiagnosticRecord(
        system=system,
        native=cd.NativeSystemOutput(
            system=system, system_version="unknown", availability="failed",
            raw_output=None, error=BATCH_CRASH_PREFIX + error,
        ),
        suspected_component=None, supporting_observation=None, evidence_attribution=[],
        method=None, reliability=None, causal_strength_language=None,
        proposed_intervention=None, no_equivalent_fields=[],
    )


def _needs_run(entry: dict | None) -> bool:
    if entry is None:
        return True
    return (entry["native"].get("error") or "").startswith(BATCH_CRASH_PREFIX)


def run_pool(
    records: list, output: Path, *, runners: Mapping[cd.SystemName, Runner], batch_size: int, metadata: dict,
) -> None:
    doc = json.loads(output.read_text()) if output.exists() else {"cases": {}}
    doc["metadata"] = metadata
    cases: dict = doc["cases"]

    for system, runner in runners.items():
        pending = [r for r in records if _needs_run(cases.get(r.example_id, {}).get(system))]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            try:
                results = runner(batch)
            except Exception as exc:  # whole-batch crash: record it, keep going
                results = {r.example_id: _crashed(system, f"{type(exc).__name__}: {exc}") for r in batch}
            for record in batch:
                cases.setdefault(record.example_id, {})[system] = results[record.example_id].model_dump(mode="json")
            output.write_text(json.dumps(doc, indent=2))
            print(f"{system}: {start + len(batch)}/{len(pending)}", file=sys.stderr, flush=True)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    from config import CLAUDE_HAIKU, CLAUDE_SONNET
    from evals.comparative_diagnostics.v1 import run_pilot

    records = cd.load_case_candidate_pool()
    case_ids = [r.example_id for r in records]
    metadata = {
        "protocol": "evals/comparative_diagnostics/v1/CASE-SELECTION-PROTOCOL.md step 1",
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "dataset_revision": f"galileo-ai/ragbench@{cd.CANDIDATE_POOL_REVISION}",
        "pool_size": len(case_ids),
        "population_sha256": cd.make_population_sha256(case_ids),
        "models": {
            "rag_forensics": {"haiku": CLAUDE_HAIKU, "sonnet": CLAUDE_SONNET},
            "ragchecker": run_pilot.RAGCHECKER_MODEL,
            "ragvue": run_pilot.RAGVUE_MODEL,
        },
    }
    run_pool(
        records, Path(args.output), batch_size=args.batch_size, metadata=metadata,
        runners={
            "rag_forensics": run_pilot.run_rag_forensics,
            "ragchecker": run_pilot.run_ragchecker,
            "ragvue": run_pilot.run_ragvue,
        },
    )
    print(f"wrote {len(case_ids)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
