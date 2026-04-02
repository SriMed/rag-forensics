# RAG Forensics

> Did your RAG system actually use the retrieved information — or just hallucinate a confident-sounding answer?

RAG Forensics is a diagnostic layer for Retrieval-Augmented Generation systems. It accepts a question, the RAG-generated answer, and the retrieved chunks, then runs five independent forensics analyses to produce a structured verdict explaining *why* an answer succeeded or failed — not just that it did.

## What it does

Standard RAG evaluation gives you scores: faithfulness 0.62, answer relevance 0.71. It doesn't tell you which failure mode you're looking at or what to do about it. RAG Forensics layers five independent analyses on top of those scores to pinpoint the problem:

| Module | Question it answers |
|---|---|
| **RAGAS Metrics** | What are the raw faithfulness and context precision scores? |
| **Score Distribution** | What is the shape of the retrieval score distribution? A flat distribution at low absolute scores signals weak retrieval — no chunk stood out. A steep decay from a high top score means one document genuinely matched. |
| **Embedding Analysis** | Is the query geometrically coherent with the retrieved chunks in embedding space, or is it an outlier? |
| **Chunk Attribution** | Is the answer actually grounded in the retrieved chunks sentence-by-sentence, or did the model hallucinate while citing them? |
| **Hedging Mismatch** | Does the model's language ("definitely", "may", "it appears") match the evidence strength in the chunks? |
| **Query-Corpus Fit** | *(Conditional)* When mismatch signals are present: what questions would the retrieved chunks actually answer well? Distinguishes between a poorly-phrased query (user was close) and a genuine coverage gap (corpus doesn't have it). |

A verdict generator synthesizes all analyses into a single readable diagnosis.

---

## Forensics examples

Each module runs independently. Here is a concrete example of the output each produces.

### RAGAS Metrics

```json
"ragas": {
  "retrieval_relevance_score": 0.41,
  "faithfulness_score": 0.87,
  "relevance_evidence": [
    "The Federal Reserve raised interest rates by 25 basis points in March.",
    "Inflation as measured by CPI reached 3.2% in February."
  ],
  "faithfulness_evidence": [
    "The Federal Reserve raised interest rates by 25 basis points in March.",
    "Inflation as measured by CPI reached 3.2% in February."
  ]
}
```

Low `retrieval_relevance_score` (0.41) with high `faithfulness_score` (0.87) is a specific failure pattern: the model answered faithfully *from the chunks it got*, but the chunks themselves weren't very relevant to the question. The retriever is the weak link, not the generator.

---

### Score Distribution

Treats the retrieval score vector as a probability distribution and measures its shape.

```json
"retrieval_distribution": {
  "top_score": 0.91,
  "score_gap": 0.43,
  "score_entropy": 0.61,
  "decay_rate": 2.8,
  "tail_mass": 0.09,
  "n_chunks": 5
}
```

High `score_gap` (0.43): one chunk dominates by a large margin. Low `tail_mass` (0.09): almost no score weight in the tail. High `decay_rate` (2.8): scores drop off steeply after the first chunk. This profile — one dominant chunk, steep decay — is associated with narrow retrieval. The answer is likely grounded in a single source even if five chunks were returned.

Contrast with a flat distribution: `score_gap` near 0, `score_entropy` near 1.0, `decay_rate` near 0. That pattern signals the retriever had no strong signal and returned five weakly-relevant chunks uniformly — a noise retrieval.

---

### Embedding Analysis

Geometric view of retrieval coherence in the 384-dimensional embedding space.

```json
"embedding_space": {
  "centroid_distance": 0.18,
  "chunk_spread": 0.34,
  "query_isolation": 0.72,
  "projection": [
    {"id": "query", "x": 0.12, "y": -0.45},
    {"id": "chunk_0", "x": 0.08, "y": -0.31},
    {"id": "chunk_1", "x": 0.21, "y": -0.38},
    {"id": "chunk_2", "x": -0.14, "y": -0.52},
    {"id": "chunk_3", "x": 0.33, "y": -0.28},
    {"id": "chunk_4", "x": -0.05, "y": -0.61}
  ]
}
```

`centroid_distance` (0.18): query is close to the centroid of retrieved chunks — good geometric alignment. `chunk_spread` (0.34): chunks are moderately spread (not identical, not wildly scattered). `query_isolation` (0.72): query sits closer to the centroid than chunks do on average — it's well inside the cluster.

Compare to a bad case: `centroid_distance` near 0.8, `query_isolation` > 2.0. `centroid_distance` of 0.8 is a cosine distance — approaching orthogonal. A brief note on what that means: sentence transformers encode meaning as *direction* in high-dimensional space, not position. Cosine distance measures the angle between two vectors, not their absolute separation. Distance 0 means the vectors point the same direction (semantically identical). Distance 1 means they are at 90° — orthogonal — which indicates no shared semantic structure that the model captured. They are not opposites, just unrelated. So `centroid_distance` of 0.8 means the query and the chunk centroid are nearly 90° apart: the query is asking about something the retrieved chunks collectively don't point toward. The chunks form their own coherent cluster, but the query is not a member of it. `query_isolation` > 2.0 means the query sits more than twice as far from the centroid as the average chunk does, i.e. it's an outlier relative to the cluster's own internal spread.

This matters because ChromaDB always returns N nearest neighbors — there is no minimum similarity threshold. If no genuinely relevant documents exist in the index, it returns the least-bad matches anyway. The retrieval scores (converted from L2 distances) might read 0.61, 0.58, 0.55 and look passable. The embedding geometry reveals what the scores obscure: the retrieved chunks have more in common with each other than any of them do with the query.

---

### Chunk Attribution

*(Implementation in progress — #7)*

Sentence-level grounding map: for each sentence in the generated answer, which chunk supports it, and how strongly?

```json
"chunk_attribution": {
  "verdict": "warn",
  "explanation": "2 of 4 answer sentences are well-grounded. 1 sentence has weak grounding (score < 0.5). 1 sentence has no supporting chunk (score < 0.3) — possible hallucination.",
  "evidence": [
    "The vaccine was authorized by the FDA on December 11, 2020."
  ],
  "attribution_map": [
    {
      "sentence": "The vaccine was authorized by the FDA on December 11, 2020.",
      "chunk_id": "covidqa_doc_14_chunk_2",
      "similarity_score": 0.91
    },
    {
      "sentence": "Clinical trials enrolled over 44,000 participants.",
      "chunk_id": "covidqa_doc_14_chunk_5",
      "similarity_score": 0.78
    },
    {
      "sentence": "The vaccine showed 95% efficacy across all age groups.",
      "chunk_id": "covidqa_doc_14_chunk_3",
      "similarity_score": 0.44
    },
    {
      "sentence": "No long-term side effects have been reported in any demographic.",
      "chunk_id": null,
      "similarity_score": 0.21
    }
  ]
}
```

The last sentence — "No long-term side effects have been reported in any demographic" — has no supporting chunk. That's a hallucination marker. The third sentence is weakly grounded (0.44), meaning the model may have extrapolated beyond what the chunk actually said.

---

### Hedging Mismatch

*(Implementation in progress — #6)*

Detects misalignment between the confidence of the model's language and the strength of the supporting evidence.

```json
"hedging_verification_mismatch": {
  "verdict": "fail",
  "explanation": "Answer uses high-confidence language ('conclusively shows', 'it is established') but retrieval scores are weak (top score 0.61) and chunk spread is high (0.71), indicating the model is more certain than the evidence warrants.",
  "evidence": [
    "Recent studies suggest a possible correlation between sleep deprivation and cardiovascular risk."
  ]
}
```

The chunk uses hedged language ("suggest", "possible correlation"). The answer asserts it as established fact. This is the mismatch: the model stripped the epistemic qualifiers from the source material and presented a softer finding as settled science.

---

### Query-Corpus Fit

*(Implementation in progress — #8)*

Conditional module — only runs when upstream signals indicate a query-corpus mismatch (`query_isolation > 1.2`, `retrieval_relevance_score < 0.5`, or both `score_entropy > 1.5` and `faithfulness_score < 0.5`). When not triggered, returns immediately with no Claude API call.

When triggered, it prompts Claude to generate 3–5 questions the retrieved chunks would actually answer well, then computes the cosine similarity between each suggested question and the original query embedding. That similarity score — `relevance_to_original` — is what distinguishes the two failure modes.

**Example: query mismatch** (`mean_question_similarity: 0.71` — suggested questions are semantically close to what the user asked)

```json
"query_corpus_fit": {
  "triggered": true,
  "mismatch_type": "query_mismatch",
  "mean_question_similarity": 0.71,
  "suggested_questions": [
    {
      "question": "What were the key monetary policy decisions made by the Federal Reserve in Q1?",
      "source_chunk_ids": ["finqa_doc_31_chunk_0", "finqa_doc_31_chunk_2"],
      "relevance_to_original": 0.78
    },
    {
      "question": "How did the Fed's rate changes affect bond yields in early 2023?",
      "source_chunk_ids": ["finqa_doc_31_chunk_1"],
      "relevance_to_original": 0.69
    },
    {
      "question": "What inflation indicators did the Federal Reserve cite in its March statement?",
      "source_chunk_ids": ["finqa_doc_31_chunk_2", "finqa_doc_31_chunk_4"],
      "relevance_to_original": 0.65
    }
  ]
}
```

The user asked something like "What did the Fed do about inflation?" The suggested questions are adjacent — same domain, same documents. `mismatch_type: "query_mismatch"` means the corpus has what the user needs, but the phrasing didn't land in the right part of embedding space. The fix is on the query side.

**Example: coverage gap** (`mean_question_similarity: 0.19` — suggested questions bear little resemblance to what the user asked)

```json
"query_corpus_fit": {
  "triggered": true,
  "mismatch_type": "coverage_gap",
  "mean_question_similarity": 0.19,
  "suggested_questions": [
    {
      "question": "What are the eligibility requirements for Medicare Part B?",
      "source_chunk_ids": ["covidqa_doc_08_chunk_1"],
      "relevance_to_original": 0.21
    },
    {
      "question": "How do I appeal a Medicare coverage denial?",
      "source_chunk_ids": ["covidqa_doc_08_chunk_3"],
      "relevance_to_original": 0.18
    },
    {
      "question": "What preventive services are covered under Medicare Advantage?",
      "source_chunk_ids": ["covidqa_doc_08_chunk_0", "covidqa_doc_08_chunk_2"],
      "relevance_to_original": 0.17
    }
  ]
}
```

The user asked something about vaccine clinical trial design. The corpus is Medicare policy documents. The suggested questions are completely unrelated to the original query — `mismatch_type: "coverage_gap"`. The fix is on the data side: this corpus cannot answer what the user needs.

---

## Relation to prior work

Two recent papers are worth situating this against.

**RAGXplain** (Abbasiantaeb et al., May 2025, arXiv:2505.xxxxx) independently arrived at a similar thesis: raw evaluation scores need LLM-powered reasoning to become actionable. It focuses on translating RAGAS scores into configuration recommendations. RAG Forensics was built without knowledge of RAGXplain. The shared insight is that scores alone are insufficient — the forensics signal set is different. RAGXplain does not describe score distribution shape or hedging mismatch as diagnostic signals.

**RAGSmith** (Kartal et al., arXiv:2511.01386) addresses the adjacent problem of *pipeline optimization* — treating RAG design as an architecture search over 46,080 configurations using genetic search. It answers "which pipeline is best for this domain?". RAG Forensics asks a different question: "why did this specific answer succeed or fail?" The two are complementary: RAGSmith finds a good configuration; RAG Forensics diagnoses what went wrong at inference time.

**On score distribution as a signal.** The RAG evaluation literature has been almost entirely focused on output quality — did the answer contain the right information. Retrieval is typically treated as a black box that either worked or didn't. Analyzing score distribution shape requires treating retrieval as a probabilistic process worth diagnosing in its own right. That framing is absent from the standard eval literature, which is why the signal hasn't been formalized before.

## Architecture

```
POST /example or /analyze/custom
  → routers/
  → services/
    → retriever.py         (ChromaDB similarity search, returns embeddings)
    → generator.py         (Claude answer generation)
    → ragas_scorer.py      (faithfulness + context_precision → float scores, no verdict)
    → forensics/
      → retrieval_distribution.py  (entropy, decay rate, score gap — pure numpy)
      → embedding_analysis.py      (centroid distance, spread, PCA projection — pure sklearn)
      → chunk_attribution.py       (sentence grounding map — in progress)
      → hedging_mismatch.py        (language vs evidence alignment — in progress)
      → query_corpus_fit.py        (conditional: suggested questions + mismatch type — in progress)
    → verdict_generator.py         (match_rule() maps signal combinations → RecommendationRule → final verdict)
```

RAGAS scores and the two pure-numeric forensics modules produce no verdict on their own. `query_corpus_fit` only makes Claude API calls when triggered. All verdict logic is deferred to the verdict generator, which runs `match_rule()` over the full signal set and maps the result to a structured `RecommendationRule` — including R08 (query mismatch) and R09 (coverage gap) from query-corpus fit.

## Endpoints

**`POST /example`** — Picks a question from the embedded RAGBench dataset, runs the full pipeline, returns a diagnostic report.

**`POST /analyze/custom`** — Accepts your own question, answer, and chunks. See [README_INTEGRATION.md](./README_INTEGRATION.md).

## Setup

```bash
cd backend
poetry install
cp .env.example .env   # add ANTHROPIC_API_KEY

# Seed the ChromaDB store from RAGBench (one-time, ~2–5 min)
poetry run python scripts/bootstrap_data.py

# Start the server
poetry run uvicorn main:app --reload
```

## Running tests

```bash
cd backend
poetry run pytest --asyncio-mode=auto
```

All external API calls (Anthropic, RAGAS) are mocked in tests. No API key is needed to run the test suite.
