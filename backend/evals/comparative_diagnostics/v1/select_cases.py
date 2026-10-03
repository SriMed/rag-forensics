"""Apply the frozen issue #30 stratum rules to a pool run and write a draft case manifest.

CASE-SELECTION-PROTOCOL.md steps 2-4. The output is a draft: the owner reviews it and freezes it by
setting `status="frozen"` and `reviewer_identity`.

Usage (from backend/):
    poetry run python evals/comparative_diagnostics/v1/select_cases.py \
        --pool-run evals/comparative_diagnostics/v1/pool-run.json \
        --oracle-report output/comparative-pool-oracle-evidence.json \
        --output evals/comparative_diagnostics/v1/selected-cases.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # backend/

import benchmark.comparative_diagnostics as cd
from benchmark.comparative_strata import build_case_set
from models import OracleEvidenceDiagnosticReport, OracleEvidenceSentenceResult


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-run", required=True)
    parser.add_argument("--oracle-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pool = json.loads(Path(args.pool_run).read_text())["cases"]
    records = {r.example_id: r for r in cd.load_case_candidate_pool()}
    if set(records) != set(pool):
        raise SystemExit("pool run does not cover exactly the candidate pool")
    report = OracleEvidenceDiagnosticReport.model_validate_json(Path(args.oracle_report).read_text())
    predictions: dict[str, list[OracleEvidenceSentenceResult]] = defaultdict(list)
    for prediction in report.predictions:
        predictions[prediction.example_id].append(prediction)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    case_set = build_case_set(
        {case_id: pool[case_id] for case_id in records}, records, predictions,
        rules=f"evals/comparative_diagnostics/v1/STRATUM-RULES.md@{commit}",
    )
    Path(args.output).write_text(case_set.model_dump_json(indent=2))
    assert case_set.selection is not None
    print(f"wrote {len(case_set.cases)} draft cases to {args.output}")
    for stratum, eligible in case_set.selection.eligible_counts.items():
        print(f"  {stratum}: {case_set.selection.selected_counts[stratum]} selected of {eligible} eligible")
    print(f"  excluded: {len(case_set.selection.exclusions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
