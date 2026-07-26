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

Same `AnalyzeResponse` shape as the demo endpoint. See the main README for full schema.
Priority scores in `verdict_signals` are heuristic ordering indices, not probabilities or
calibrated severities. Check each signal's `reliability` field and treat recommendations as
experiments to test rather than established root causes.
