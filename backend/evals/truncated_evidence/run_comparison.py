"""Run issue #25's paired complete/truncated generation comparison.

Run from ``backend/`` with an authenticated Claude CLI:
    poetry run python -m evals.truncated_evidence.run_comparison --repetitions 2

The CLI is proxy-model evidence: it does not exercise the production API model ID.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import date
from pathlib import Path

from models import RetrievedChunk
from prompts.generation_prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt


CASES = Path(__file__).parent / "v1" / "cases.json"
CLAUDE_MODEL = "haiku"
CONDITIONS = ("baseline", "metadata", "qualification", "hybrid")
QUALIFICATION = (
    " If a context chunk visibly ends mid-sentence, do not complete its thought or infer "
    "the missing words. State what the visible text supports and explicitly qualify that "
    "the relevant evidence is truncated."
)


def detect_incomplete_terminal_sentence(text: str) -> bool:
    """Conservatively flag prose whose last non-space character is not terminal punctuation."""
    stripped = text.rstrip()
    if not stripped:
        return False
    stripped = re.sub(r'[\]\)\}"\'”’]+$', "", stripped).rstrip()
    return bool(stripped) and stripped[-1] not in ".!?;:"


def render(question: str, evidence: str, condition: str) -> tuple[str, str, bool]:
    detected = detect_incomplete_terminal_sentence(evidence)
    prompt = build_generation_prompt(
        question, [RetrievedChunk(chunk_id="evidence", text=evidence, score=1.0)]
    )
    system = GENERATION_SYSTEM_PROMPT
    if condition == "metadata":
        prompt += (
            "\n\nMetadata: the final context chunk ends mid-sentence."
            if detected
            else "\n\nMetadata: the final context chunk is complete."
        )
    elif condition == "hybrid" and detected:
        prompt += "\n\nMetadata: the final context chunk ends mid-sentence."
    if condition in {"qualification", "hybrid"}:
        system += QUALIFICATION
    return system, prompt, detected


def run_proxy(system: str, prompt: str, runner: str) -> dict:
    try:
        command = (
            [
                "claude", "-p", "--model", CLAUDE_MODEL, "--no-session-persistence",
                "--permission-mode", "dontAsk", "--system-prompt", system, prompt,
            ]
            if runner == "claude"
            else [
                "codex", "exec", "--ephemeral", "--sandbox", "read-only",
                "--skip-git-repo-check", system + "\n\n" + prompt,
            ]
        )
        completed = subprocess.run(
            command,
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


def assess(case: dict, variant: str, response: str) -> dict:
    lower = response.lower()
    forbidden = case["forbidden_additions_when_truncated"] if variant == "truncated" else []
    required = case["required_complete"] if variant == "complete" else []
    return {
        "forbidden_phrase_present": any(value.lower() in lower for value in forbidden),
        "explicit_truncation": any(value in lower for value in ("truncat", "incomplete", "cut off", "ends mid")),
        "qualified": any(value in lower for value in ("may", "suggest", "cannot", "not enough", "incomplete", "truncat", "visible context")),
        "complete_answer_useful": (
            all(value.lower() in lower for value in required) if variant == "complete" else None
        ),
        "human_review": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runner", choices=("claude", "codex"), default="claude")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--variants", nargs="+", choices=("complete", "truncated"), default=("complete", "truncated"))
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    results = []
    for case in cases:
        for variant in args.variants:
            evidence = case[variant]
            for condition in args.conditions:
                system, prompt, detected = render(case["question"], evidence, condition)
                for repetition in range(1, args.repetitions + 1):
                    output = run_proxy(system, prompt, args.runner)
                    results.append({
                        "case_id": case["id"], "variant": variant,
                        "condition": condition, "repetition": repetition,
                        "detected_incomplete": detected, **output,
                        "assessment": assess(case, variant, output["response"]),
                    })
    payload = {
        "schema_version": "truncated-evidence-results.v1",
        "runner": f"{args.runner.title()} CLI",
        "model_alias": CLAUDE_MODEL if args.runner == "claude" else "configured default",
        "run_date": date.today().isoformat(),
        "production_model": "claude-haiku-4-5-20251001",
        "proxy_model_caveat": "The CLI alias is proxy evidence and may not resolve to the production model ID.",
        "repetitions": args.repetitions, "results": results,
    }
    if args.merge_existing and args.output and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        replacements = {
            (item["case_id"], item["variant"], item["condition"], item["repetition"]): item
            for item in results
        }
        merged = [
            replacements.pop(
                (item["case_id"], item["variant"], item["condition"], item["repetition"]),
                item,
            )
            for item in existing["results"]
        ] + list(replacements.values())
        cases_by_id = {case["id"]: case for case in cases}
        for item in merged:
            item["assessment"] = assess(
                cases_by_id[item["case_id"]], item["variant"], item["response"]
            )
        payload["results"] = merged
    artifact = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(artifact + "\n", encoding="utf-8")
    else:
        print(artifact)


if __name__ == "__main__":
    main()
