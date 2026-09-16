"""Run RAGVue, unmodified, inside its own isolated environment.

Invoke with the RAGVue venv's own python:
    RAGVUE_JUDGE_PROVIDER=anthropic ANTHROPIC_API_KEY=... \
        envs/.venv-ragvue/bin/python drivers/run_ragvue.py --input IN.json --output OUT.json

Input is a JSON list of `{"question", "answer", "contexts", "case_id"}` items (RAGVue's own
question/answer/contexts shape, plus a case_id we thread through for joining results back).

RAGVue's own `raw.model` metadata is known to mislabel the judge actually used (see ../README.md
and ADR-044) — this driver does not correct it; it is preserved verbatim in the output alongside
the model this script actually configured.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["retrieval_relevance", "strict_faithfulness", "context_similarity"],
    )
    args = parser.parse_args()

    import ragvue

    with open(args.input) as f:
        items = json.load(f)

    configured_model = (
        f"anthropic:{os.environ.get('RAGVUE_ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001')}"
        if os.environ.get("RAGVUE_JUDGE_PROVIDER") == "anthropic"
        else f"openai:{ragvue.DEFAULT_MODEL}"
    )

    output = {"configured_model": configured_model, "per_item": {}, "error": None}
    try:
        for item in items:
            case_id = item["case_id"]
            payload = {k: v for k, v in item.items() if k != "case_id"}
            result = ragvue.evaluate([payload], metrics=args.metrics)
            output["per_item"][case_id] = result["results"][0]
    except Exception as exc:  # native tool failure, recorded rather than hidden
        output["error"] = f"{type(exc).__name__}: {exc}"

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
