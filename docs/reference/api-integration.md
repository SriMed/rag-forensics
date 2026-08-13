# Using RAG Forensics with your own RAG system

The forensics layer accepts your own retrieved chunks and generated answer directly
via the `/analyze/custom` endpoint — no RAGBench or ChromaDB required.

## Minimal example

```python
import requests

# Your existing retrieval results (e.g. from OpenSearch KNN)
chunks = [
    {"chunk_id": "doc_42_chunk_3", "text": "...", "score": 0.87},
    {"chunk_id": "doc_17_chunk_1", "text": "...", "score": 0.74},
]

response = requests.post("https://your-deployment.railway.app/analyze/custom", json={
    "question": "What is the refund policy?",
    "answer": "Refunds are processed within 5-7 business days.",
    "score_semantics": "normalized_similarity",
    "chunks": chunks
})

print(response.json())
```

## Chunk format

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | Any unique identifier for the chunk in your system |
| `text` | string | The raw text content of the chunk |
| `score` | float (0–1) | Normalized similarity where higher means more relevant |

`score_semantics` is currently required to be `normalized_similarity`. BM25 values, distances,
reranker logits, and vendor-specific scores must be converted to a meaningful 0–1 similarity
scale before use. Distribution shape is not comparable across retrievers unless their score
calibration is comparable.

## Response

Same `AnalyzeResponse` shape as the demo endpoint. Its RAGAS portion uses explicit result objects:

```json
{
  "ragas": {
    "context_utilization": {"score": 0.82, "status": "ok", "error": null},
    "faithfulness": {"score": null, "status": "unavailable", "error": "evaluation_failed"},
    "utilization_context_excerpts": ["..."],
    "faithfulness_context_excerpts": ["..."]
  }
}
```

`context_utilization` is answer-conditioned: it asks whether higher-ranked retrieved contexts were
useful for producing the supplied answer. It is not a direct measure of question–context relevance,
retriever quality, or a calibrated probability. A metric exception produces `evaluation_failed`;
a `NaN` or infinite output produces `non_finite_score`. Both use `score: null` and are skipped by
numeric downstream triggers rather than coerced to zero. See the main README for the remaining
response schema.
Priority scores in `verdict_signals` are heuristic ordering indices, not probabilities or
calibrated severities. Check each signal's `reliability` field and treat recommendations as
experiments to test rather than established root causes.

The `hedging_mismatch` result also carries explicit availability semantics. A validated empty claim
array returns `status: "ok"`, `total_claims: 0`, and `error: null`. Extraction failures return
`status: "error"` and one of `claim_extraction_failed`, `claim_extraction_parse_failed`, or
`claim_extraction_schema_failed`; downstream verdicts treat those zero-valued metric placeholders as
unavailable rather than healthy.

Each extracted claim includes `entailment_checks`, one record for every attempted top-three chunk.
After trimming surrounding whitespace, the only valid model outputs are the exact lowercase enum
values `supported` and `not_supported`. Commentary, punctuation, capitalization, alternate spacing,
and arbitrary prose produce `status: "invalid_format"`; request or response failures produce
`status: "error"`. Both have `verdict: null` and are distinct from an evaluated
`not_supported` judgment. Evaluation continues to later chunks and still stops at the first valid
`supported` result.

Claims with no valid chunk verdict expose `supported: null` and `mismatch_type: null`.
`overconfident_fraction` and `underconfident_fraction` use only `evaluated_claim_count` as their
denominator, while `total_claims`, `unavailable_claim_count`, and the per-chunk records preserve
coverage. Consequently, a zero mismatch fraction is not evidence of a healthy answer when
`unavailable_claim_count` is nonzero; the verdict layer emits a separate unavailable-judgment
signal in that case. `evaluated_chunk_count` counts valid chunk verdicts, not attempted chunks.
