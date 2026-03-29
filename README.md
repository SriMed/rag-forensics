# RAG Forensics

> Did your RAG system actually use the retrieved information — or just hallucinate a confident-sounding answer?

RAG Forensics is a diagnostic layer for Retrieval-Augmented Generation systems. It accepts a question, the RAG-generated answer, and the retrieved chunks, then runs four forensics analyses plus RAGAS scoring to produce a structured verdict explaining *why* an answer succeeded or failed — not just that it did.

## What it does

Standard RAG evaluation gives you scores: faithfulness 0.62, answer relevance 0.71. It doesn't tell you which failure mode you're looking at or what to do about it. RAG Forensics layers four independent analyses on top of those scores to pinpoint the problem:

| Module | Question it answers |
|---|---|
| **Chunk Attribution** | Is the answer actually grounded in the retrieved chunks, or did the model hallucinate while citing them? |
| **Score Distribution** | Are the retrieval scores suspiciously clustered? Low variance often signals a retriever returning low-confidence filler rather than genuine matches. |
| **Hedging Mismatch** | Does the model's language ("definitely", "may", "it appears") match the evidence strength in the chunks? Overconfident language against weak evidence is a separate failure mode from hallucination. |
| **Confidence Calibration** | Is the model's expressed certainty calibrated against what the chunks actually support? |

A verdict generator synthesizes all four analyses into a single readable diagnosis.

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
    → retriever.py         (ChromaDB similarity search)
    → generator.py         (Claude answer generation)
    → ragas_scorer.py      (faithfulness, answer relevance)
    → forensics/
      → chunk_attribution.py
      → confidence_calibration.py
      → hedging_mismatch.py
      → retrieval_distribution.py
    → verdict_generator.py
```

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
