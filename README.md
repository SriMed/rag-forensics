# RAG Forensics

> What observable signals can help a developer investigate why a RAG answer went wrong?

RAG Forensics is a transparent hypothesis-generation layer for Retrieval-Augmented Generation systems. It accepts a question, the RAG-generated answer, and the retrieved chunks, then produces an inspectable diagnostic record: observations, heuristic priorities, reliability labels, source candidates, and concrete follow-up tests. Its outputs narrow an investigation; they do not prove a root cause.

## What it does

Standard RAG evaluation gives you scores such as faithfulness 0.62 and answer relevance 0.71. Those scores rarely identify a unique failure mode. RAG Forensics layers five complementary analyses on top of them to rank plausible diagnostic hypotheses:

| Module | Question it answers |
|---|---|
| **RAGAS Metrics** | What are the raw faithfulness and context precision scores? |
| **Score Distribution** | What is the shape of the retrieval score distribution? Flatness and decay are observations that must be interpreted jointly with absolute relevance and score semantics. |
| **Embedding Analysis** | Is the query geometrically coherent with the retrieved chunks in embedding space, or is it an outlier? |
| **Semantic Attribution** | Which retrieved chunk is the closest semantic source candidate for each answer sentence? Similarity is not treated as proof of entailment. |
| **Hedging Mismatch** | Does the model's language ("definitely", "may", "it appears") match the evidence strength in the chunks? |
| **Retrieved-Context Fit** | *(Conditional)* What questions would the retrieved chunks answer well, and are those questions near or far from the original query? This describes the retrieved set, not the full corpus. |

A verdict generator synthesizes all analyses into a ranked list of heuristic priority signals (`verdict_signals`) and a readable investigation recommendation. Priority scores make ordering deterministic, but are not probabilities or cross-signal calibrated severities. Every signal states whether it is unvalidated, partially calibrated, or model-judged.

---

## Forensics examples

Each module can be called in isolation, although several signals share inputs, embeddings, or model judges. Here is a concrete example of each output.

### RAGAS Metrics

```json
"ragas": {
  "retrieval_relevance_score": 0.41,
  "faithfulness_score": 0.87,
  "relevance_context_excerpts": [
    "The Federal Reserve raised interest rates by 25 basis points in March.",
    "Inflation as measured by CPI reached 3.2% in February."
  ],
  "faithfulness_context_excerpts": [
    "The Federal Reserve raised interest rates by 25 basis points in March.",
    "Inflation as measured by CPI reached 3.2% in February."
  ]
}
```

Low `retrieval_relevance_score` (0.41) with high `faithfulness_score` (0.87) is a specific failure pattern: the model answered faithfully *from the chunks it got*, but the chunks themselves weren't very relevant to the question. The retriever is the weak link, not the generator.

---

### Score Distribution

Normalizes the retrieval score vector and measures its shape. `normalized_entropy = entropy / log(n_chunks)` makes entropy comparable across different top-k values.

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

High `score_gap` (0.43), low `tail_mass` (0.09), and high `decay_rate` (2.8) describe a narrow retrieved set dominated by one chunk. Whether that chunk is genuinely relevant requires an absolute relevance check.

Contrast with a flat distribution: `score_gap` near 0, normalized entropy near 1.0, and `decay_rate` near 0. This shows that scores are similarly distributed. It does **not** distinguish uniformly strong retrieval from uniformly weak retrieval, so shape must be interpreted jointly with absolute relevance and documented score semantics.

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

Sentence-level grounding map: for each sentence in the generated answer, which chunk supports it, and how strongly?

```json
"chunk_attribution": {
  "unattributed_fraction": 0.25,
  "mean_attribution_score": 0.61,
  "weak_match_fraction": 0.25,
  "attribution_map": [
    {
      "sentence": "The vaccine was authorized by the FDA on December 11, 2020.",
      "chunk_id": "covidqa_doc_14_chunk_2",
      "similarity_score": 0.91,
      "attribution_strength": "strong"
    },
    {
      "sentence": "Clinical trials enrolled over 44,000 participants.",
      "chunk_id": "covidqa_doc_14_chunk_5",
      "similarity_score": 0.78,
      "attribution_strength": "strong"
    },
    {
      "sentence": "The vaccine showed 95% efficacy across all age groups.",
      "chunk_id": "covidqa_doc_14_chunk_3",
      "similarity_score": 0.44,
      "attribution_strength": "weak"
    },
    {
      "sentence": "No long-term side effects have been reported in any demographic.",
      "chunk_id": null,
      "similarity_score": 0.21,
      "attribution_strength": "unattributed"
    }
  ]
}
```

The last sentence has no semantically close source candidate under the current threshold. That makes it useful for review, but does not by itself establish hallucination. Topic similarity can miss valid synthesis and can falsely accept contradictions, incorrect numbers, or changed qualifiers.

---

### Hedging Mismatch

Detects misalignment between the confidence of the model's language and the strength of the supporting evidence.

```json
"hedging_mismatch": {
  "overconfident_fraction": 0.5,
  "underconfident_fraction": 0.0,
  "total_claims": 4,
  "claim_breakdown": [
    {
      "claim": "Recent studies conclusively show a link between sleep deprivation and cardiovascular risk.",
      "confidence_class": "definitive",
      "supported": false,
      "mismatch_type": "overconfident",
      "source_chunk_id": "healthqa_doc_07_chunk_1"
    },
    {
      "claim": "It is established that poor sleep increases cortisol levels.",
      "confidence_class": "definitive",
      "supported": false,
      "mismatch_type": "overconfident",
      "source_chunk_id": "healthqa_doc_07_chunk_3"
    },
    {
      "claim": "Some researchers suggest this may be a bidirectional relationship.",
      "confidence_class": "hedged",
      "supported": true,
      "mismatch_type": "matched",
      "source_chunk_id": "healthqa_doc_07_chunk_2"
    },
    {
      "claim": "Further studies are needed to confirm causality.",
      "confidence_class": "hedged",
      "supported": true,
      "mismatch_type": "matched",
      "source_chunk_id": "healthqa_doc_07_chunk_4"
    }
  ]
}
```

The chunk uses hedged language ("suggest", "possible correlation"). The answer asserts it as established fact. This is the mismatch: the model stripped the epistemic qualifiers from the source material and presented a softer finding as settled science.

---

### Retrieved-Context Fit

Conditional module — only runs when upstream signals indicate a retrieved-context mismatch. The three trigger conditions are checked in order: `query_isolation > 1.2`, `retrieval_relevance_score < 0.5`, or `normalized_entropy > 0.9 AND faithfulness_score < 0.5`. The `trigger_reason` field names which condition fired. These thresholds are heuristic.

When triggered, it prompts Claude to generate 3–5 questions the retrieved chunks would answer well, then computes cosine similarity between each suggested question and the original query. This distinguishes an observed retrieved-context near miss from an observed retrieved-context topic gap. It cannot determine whether an unretrieved document elsewhere in the corpus contains the answer.

**Example: retrieved-context near miss** (`mean_question_similarity: 0.71`)

```json
"query_corpus_fit": {
  "triggered": true,
  "observed_fit": "retrieved_context_near_miss",
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

The suggested questions are adjacent to the original question. This supports testing a query rewrite; it does not prove that phrasing caused the failure or that the full answer exists in the corpus.

**Example: retrieved-context topic gap** (`mean_question_similarity: 0.19`)

```json
"query_corpus_fit": {
  "triggered": true,
  "observed_fit": "retrieved_context_topic_gap",
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

The suggested questions are distant from the original query, showing that the retrieved set is off-topic. Establishing a corpus coverage gap requires a corpus-level answerability check or counterfactual retrieval experiment.

---

## Contribution and limitations

### What this adds

The contribution is an **inspectable instance-level debugging record**. Instead of collapsing evaluation into one score, RAG Forensics exposes complementary observations, their method and reliability, a deterministic heuristic ordering, and follow-up tests. This makes the diagnostic reasoning contestable without claiming that the observations identify a unique cause.

The retrieved-context fit module adds a useful intervention split: semantically near retrieved content motivates testing query reformulation, while distant content motivates testing retrieval and corpus coverage. These remain hypotheses until the proposed test is run.

**Confidence in this contribution: moderate.** The implementation establishes inspectability and deterministic ordering. Its diagnostic accuracy and developer utility have not yet been established by a labeled benchmark or user study.

**What would change this assessment:** failure to improve root-cause classification, time-to-diagnosis, or intervention success over RAGAS-only baselines would weaken the contribution. A blinded injected-failure benchmark is the intended test.

### Honest limitations

**Semantic attribution thresholds are uncalibrated.** The `"strong"` (> 0.75) and `"unattributed"` (< 0.40) cosine-similarity thresholds were chosen by convention. They identify source candidates; they have not been validated as entailment or hallucination classifiers.

**Hedging mismatch confidence classification is lexicon-bounded.** The confidence classifier uses a priority-ordered lexicon of uncertainty markers. Constructions outside that lexicon are classified as `definitive`, which can bias `overconfident_fraction` upward.

**Underconfidence is not currently inferred.** Binary entailment does not establish that a hedge was unnecessary; doing so requires comparing the source's epistemic strength with the answer's. The compatibility field remains zero until that analysis exists.

**The entailment check covers only the top 3 chunks.** A claim grounded in chunk 4 or 5 will be classified as overconfident regardless of whether it is actually supported. The fix (increase `_ENTAILMENT_TOP_K`) trades cost for accuracy.

**The verdict uses only the top 3 signals.** The full ranked list is available in `verdict_signals` in the API response. The recommendation prose is generated from the top 3 only; signals ranked 4th and below are not sent to Claude. Consult `verdict_signals` directly for the full picture.

**The query_corpus_fit trigger thresholds are heuristic.** `query_isolation > 1.2`, `retrieval_relevance_score < 0.5`, and `normalized_entropy > 0.9` were chosen without outcome-label calibration. The `trigger_reason` field exposes which condition fired.

**Priority scores are heuristic indices.** They combine quantities with different semantics and reliability. Their ordering is deterministic but not calibrated as probability, severity, or expected intervention value. Each signal exposes a reliability class.

**Retrieved-context fit is not corpus answerability.** The module observes only retrieved chunks. Its near-miss and topic-gap labels deliberately avoid claiming that the full corpus contains or lacks the answer.

**Analysis failures are explicit.** LLM-backed modules return `status="error"` with an error code rather than silently representing failure as a clean zero. Consumers must display unknown separately from healthy.

**The embedding model is an implicit shared dependency.** Semantic attribution and retrieved-context fit use the same internal MiniLM model. `/analyze/custom` re-embeds the supplied text in that space rather than accepting caller embeddings, which keeps comparisons coherent but can differ from the caller's production retriever.

### Relation to prior work

**RAGXplain** (Abbasiantaeb et al., May 2025) independently arrived at a similar thesis: raw evaluation scores need reasoning to become actionable. It focuses on translating RAGAS scores into configuration recommendations. The shared insight is that scores alone are insufficient. RAGXplain does not describe score distribution shape or hedging mismatch as diagnostic signals; those signals are not, as far as I can determine from the standard RAG eval literature (RAGAS, RAGTruth, ARES), previously formalized — though I have not done a systematic review.

**RAGSmith** (Kartal et al., arXiv:2511.01386) addresses *pipeline optimization* — architecture search over 46,080 RAG configurations. It answers "which pipeline is best for this domain?". RAG Forensics asks "why did this specific answer fail?" The two are complementary.

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
      → chunk_attribution.py       (semantic source-candidate map — pure numpy/sklearn/nltk)
      → hedging_mismatch.py        (language vs evidence alignment — LLM entailment)
      → query_corpus_fit.py        (conditional: suggested questions + observed retrieved-context fit)
    → verdict_generator.py         (heuristic priority ranking → test-oriented recommendation)
```

RAGAS scores and the numeric forensics modules produce observations rather than verdicts. `query_corpus_fit` only makes Claude API calls when triggered. The verdict generator orders heuristic priorities deterministically, then asks Claude to express the highest-ranked observations as hypotheses and falsifiable follow-up tests.

## Endpoints

**`POST /example`** — Picks a question from the embedded RAGBench dataset, runs the full pipeline, returns a diagnostic report.

**`POST /analyze/custom`** — Accepts your own question, answer, and chunks. See [README_INTEGRATION.md](./README_INTEGRATION.md).

## Label-preserving RAGBench evaluation

The interactive demo re-retrieves documents and generates a new answer, so its output cannot
be compared directly with RAGBench's original sentence-support labels. The offline benchmark
runner instead preserves each row's original:

- question, documents, response, and response-sentence segmentation;
- unsupported response-sentence keys;
- sentence-to-source support mappings;
- adherence, relevance, utilization, and completeness labels.

It evaluates the semantic-attribution module as an unsupported-sentence detector. A sentence
is predicted unsupported when its attribution strength is `unattributed`. The report contains
per-sentence raw similarity, source candidate, gold label, prediction, confusion counts,
precision, recall, F1, coverage, explicit skipped-row reasons, and run configuration.

```bash
cd backend

# Writes JSON to stdout. Dataset order is deterministically shuffled by seed.
poetry run python -m benchmark.cli \
  --domain techqa \
  --split test \
  --limit 100 \
  --seed 42

# Or save the machine-readable report.
poetry run python -m benchmark.cli \
  --domain covidqa \
  --split validation \
  --limit 250 \
  --seed 42 \
  --output output/ragbench-covidqa.json
```

The default benchmark makes no Anthropic or RAGAS calls. It does use the local
`sentence-transformers/all-MiniLM-L6-v2` embedding model and may download that model or the
dataset if they are not already cached.

This benchmark validates unsupported-sentence screening against RAGBench outcome labels. It
does not establish that similarity proves entailment, and it cannot identify configuration-level
causes such as top-k, chunking, or corpus coverage.

### Current benchmark status

The first reproducible smoke benchmark used 100 seeded examples from the RAGBench TechQA test
split (`seed=42`). All 100 records were evaluated without skips, covering 946 response sentences.

| Metric | Result |
|---|---:|
| Gold unsupported sentences | 330 |
| Predicted unsupported sentences | 349 |
| Precision | 0.381 |
| Recall | 0.403 |
| F1 | 0.392 |
| Coverage | 1.000 |
| AUROC (lower similarity predicts unsupported) | 0.563 |

At the current `0.4` unattributed threshold, whole-sentence MiniLM similarity is therefore not a
reliable standalone unsupported-claim detector. Supported and unsupported sentences overlap
substantially in similarity. An in-sample threshold sweep reached approximately `0.547` F1 at
`0.67`, largely by predicting almost every sentence as unsupported; an always-unsupported
classifier scores approximately `0.517` F1 on this sample. The sweep is diagnostic and is not a
held-out calibration result.

The supported conclusion is narrower: embedding similarity can rank source candidates for
investigation, but this experiment does not justify treating its threshold as proof of grounding
or hallucination. The next scientific comparison should evaluate the current method against
claim decomposition plus entailment-aware verification across every RAGBench domain, followed
by external validation on a dataset such as RAGTruth.

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
