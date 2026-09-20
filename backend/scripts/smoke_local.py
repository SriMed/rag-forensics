"""Check a running local backend; model calls require the explicit --analyze flag."""
import argparse
import json
from pathlib import Path

import httpx

from models import AnalyzeResponse, CustomAnalyzeRequest


def run_smoke(client: httpx.Client, *, analyze: bool = False) -> dict:
    health = client.get("/health", timeout=10)
    health.raise_for_status()
    if health.json().get("status") != "ok":
        raise ValueError("Unexpected /health response")
    ready = client.get("/ready", timeout=10)
    if ready.status_code == 503 and ready.json().get("checks", {}).get("anthropic_api_key") is False:
        raise ValueError("Set ANTHROPIC_API_KEY in backend/.env and restart the backend")
    ready.raise_for_status()
    body = ready.json()
    summary = {
        "health": "ok", "ready": body["status"] == "ready",
        "bundled_corpus_domains": body["bundled_corpus_domains"], "analysis": "not_requested",
    }
    if analyze:
        example = Path(__file__).resolve().parents[1] / "examples/custom-analysis.json"
        payload = CustomAnalyzeRequest.model_validate_json(example.read_text())
        response = client.post("/analyze/custom", json=payload.model_dump(), timeout=600)
        response.raise_for_status()
        result = AnalyzeResponse.model_validate(response.json())
        if result.ragas.context_utilization.status != "ok" or result.ragas.faithfulness.status != "ok":
            raise ValueError("RAGAS evaluation unavailable; inspect backend logs")
        if (
            result.hedging_mismatch.status != "ok"
            or result.hedging_mismatch.unavailable_claim_count
            or result.query_corpus_fit.status == "error"
        ):
            raise ValueError("Forensics unavailable; inspect backend logs")
        summary["analysis"] = "ok"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--analyze", action="store_true", help="Send the public example to Anthropic via the backend")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url) as client:
        print(json.dumps(run_smoke(client, analyze=args.analyze), indent=2))


if __name__ == "__main__":
    main()
