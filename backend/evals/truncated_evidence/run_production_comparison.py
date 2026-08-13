"""Compare the pre-#27 baseline with the source-metadata contract on production model."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

from evals.truncated_evidence.run_comparison import assess
from models import RetrievedChunk
from prompts.generation_prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt


MODEL = "claude-haiku-4-5-20251001"
CASES = Path(__file__).parent / "v1" / "cases.json"
BASELINE_SYSTEM_PROMPT = (
    "You are a precise answer generator for a RAG system. "
    "Answer the question using ONLY information from the provided context chunks. "
    "Do not add any information that is not present in the chunks. "
    "Be concise and factual."
)


def _baseline_prompt(question: str, evidence: str) -> str:
    return (
        f"Context:\n[Chunk 1]: {evidence}\n\nQuestion: {question}\n\n"
        "Answer based solely on the context above:"
    )


def render(case: dict, variant: str, condition: str) -> tuple[str, str]:
    evidence = case[variant]
    if condition == "pre_issue_27":
        return BASELINE_SYSTEM_PROMPT, _baseline_prompt(case["question"], evidence)
    chunk = RetrievedChunk(
        chunk_id="evidence", text=evidence, score=1.0,
        completeness="complete" if variant == "complete" else "truncated",
        completeness_source="source",
    )
    return GENERATION_SYSTEM_PROMPT, build_generation_prompt(case["question"], [chunk])


def run(system: str, prompt: str) -> dict:
    try:
        completed = subprocess.run(
            [
                "claude", "-p", "--model", MODEL, "--no-session-persistence",
                "--permission-mode", "dontAsk", "--system-prompt", system, prompt,
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        return {"status": "ok", "response": completed.stdout.strip(), "error": None}
    except subprocess.CalledProcessError as exc:
        return {
            "status": "unavailable", "response": "",
            "error": f"CalledProcessError: {exc.stderr.strip()[-500:]}",
        }
    except Exception as exc:
        return {"status": "unavailable", "response": "", "error": type(exc).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    results = []
    for case in cases:
        for variant in ("complete", "truncated"):
            for condition in ("pre_issue_27", "source_metadata_contract"):
                system, prompt = render(case, variant, condition)
                for repetition in range(1, args.repetitions + 1):
                    output = run(system, prompt)
                    results.append({
                        "case_id": case["id"], "variant": variant,
                        "condition": condition, "repetition": repetition, **output,
                        "assessment": assess(case, variant, output["response"]),
                        "human_review": None,
                    })
    artifact = json.dumps({
        "schema_version": "truncated-evidence-production-results.v2",
        "run_date": date.today().isoformat(), "runner": "Claude CLI",
        "model": MODEL, "repetitions": args.repetitions, "results": results,
    }, indent=2)
    if args.output:
        args.output.write_text(artifact + "\n", encoding="utf-8")
    else:
        print(artifact)


if __name__ == "__main__":
    main()
