"""Run the issue #20 labeled before/after comparison.

Run from backend with an Anthropic credential:
    poetry run python evals/context_utilization/run_comparison.py
"""
import json
import subprocess
from pathlib import Path

from ragas.metrics._context_precision import QAC, context_utilization


MODEL = "haiku"
CASES = Path(__file__).parent / "v1/cases.json"


def _parse_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(stripped)


def _score(question: str, answer: str, contexts: list[str]) -> dict:
    verdicts = []
    reasons = []
    try:
        for context in contexts:
            prompt = context_utilization.context_precision_prompt.to_string(
                QAC(question=question, answer=answer, context=context)
            )
            completed = subprocess.run(
                [
                    "claude",
                    "-p",
                    "--model",
                    MODEL,
                    "--no-session-persistence",
                    "--permission-mode",
                    "dontAsk",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            parsed = _parse_json(completed.stdout)
            verdicts.append(1 if parsed["verdict"] else 0)
            reasons.append(parsed["reason"])
    except Exception as exc:
        return {"score": None, "status": "unavailable", "error": type(exc).__name__}
    relevant = sum(verdicts)
    score = 0.0 if relevant == 0 else sum(
        (sum(verdicts[: index + 1]) / (index + 1)) * verdict
        for index, verdict in enumerate(verdicts)
    ) / relevant
    return {
        "score": score,
        "status": "ok",
        "error": None,
        "context_verdicts": verdicts,
        "reasons": reasons,
    }


def main() -> None:
    results = []
    for case in json.loads(CASES.read_text()):
        before = _score(case["question"], "N/A", case["contexts"])
        after = _score(case["question"], case["answer"], case["contexts"])
        results.append(
            {
                "id": case["id"],
                "human_context_labels": case["human_context_labels"],
                "before_context_precision_with_sentinel": before,
                "after_context_utilization": after,
                "review": None,
            }
        )
    print(json.dumps({"ragas_version": "0.4.3", "model": MODEL, "results": results}, indent=2))


if __name__ == "__main__":
    main()
