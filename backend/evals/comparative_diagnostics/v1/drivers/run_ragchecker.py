"""Run RAGChecker, unmodified, inside its own isolated environment.

Invoke with the RAGChecker venv's own python:
    envs/.venv-ragchecker/bin/python drivers/run_ragchecker.py --input IN.json --output OUT.json

Input is RAGChecker's own native `{"results": [...]}` shape (see RAGChecker's
`examples/checking_inputs.json`), with `gt_answer` left as `""` — see ../README.md for why. Only
`faithfulness` is requested: every other RAGChecker metric requires `gt_answer`.

Output preserves RAGChecker's own per-item and overall metrics verbatim, plus an `error` entry per
item if extraction/checking raised.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="anthropic/claude-haiku-4-5-20251001",
        help="litellm model string passed to RAGChecker's extractor and checker.",
    )
    args = parser.parse_args()

    from ragchecker.container import RAGResults
    from ragchecker.evaluator import RAGChecker

    with open(args.input) as f:
        raw = f.read()
    results = RAGResults.from_json(raw)

    checker = RAGChecker(extractor_name=args.model, checker_name=args.model, batch_size_extractor=1, batch_size_checker=1)

    output: dict[str, Any] = {"per_item": {}, "overall": None, "error": None}
    try:
        checker.evaluate(results, metrics=["faithfulness"])
        output["overall"] = results.metrics
        for item in results.results:
            output["per_item"][item.query_id] = item.metrics
    except Exception as exc:  # native tool failure, recorded rather than hidden
        output["error"] = f"{type(exc).__name__}: {exc}"

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
