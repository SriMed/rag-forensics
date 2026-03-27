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
    "chunks": chunks
})

print(response.json())
```

## Chunk format

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | Any unique identifier for the chunk in your system |
| `text` | string | The raw text content of the chunk |
| `score` | float (0–1) | Similarity score from your retrieval system |

## Response

Same `AnalyzeResponse` shape as the demo endpoint. See the main README for full schema.
