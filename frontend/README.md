# RAG Forensics — Frontend

A Next.js interface for the RAG Forensics internal demo. It loads a seeded RAGBench example,
sends it to the backend for analysis, and renders the resulting diagnostic record — heuristic
priorities, reliability labels, and the underlying forensics evidence — for inspection.

This app is a viewer for the backend's diagnostic output. It does not run any forensics analysis
itself; all retrieval, generation, and scoring happen in the FastAPI backend documented in the
[repository root README](../README.md) and [`docs/`](../docs/).

## Backend dependency

This interface requires the backend running and reachable at `NEXT_PUBLIC_API_URL` (defaults to
`http://localhost:8000`). Start the backend first — see the root
[README's Quick start](../README.md#quick-start) — then point this app at it.

## Setup

```bash
npm install
cp .env.local.example .env.local   # edit NEXT_PUBLIC_API_URL if the backend isn't on localhost:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Pick a RAGBench domain, load a seeded
example, then run analysis to see the assembled diagnostic record.

## Commands

```bash
npm run dev     # start the dev server
npm run build   # production build
npm run start   # serve a production build
npm run lint    # eslint
npm test        # jest + testing-library
```

## Architecture

```
app/page.tsx
  → app/components/ExampleBrowser.tsx   (domain selection, example loading, triggers analysis)
    → lib/api.ts                        (typed fetch client: GET /example, POST /analyze)
    → app/components/DiagnosticCard.tsx (renders one forensics result)
```

`lib/api.ts` defines the response types this app consumes — `ChunkAttributionMetrics`,
`HedgingMismatchMetrics`, `RAGASMetrics`, and the other forensics shapes described in the backend's
architecture docs. Keep these types in sync with `backend/models.py` when the backend's response
shapes change.

## Evidence semantics

The values this app renders are heuristic observations and reliability labels, not proofs of root
cause. In particular:

- Priority scores from `verdict_signals` are not probabilities or calibrated severities.
- `ChunkAttributionMetrics` reports semantic similarity between an answer sentence and a chunk —
  it does not establish entailment.
- RAGAS scores (`retrieval_relevance_score`, `faithfulness_score`) are raw continuous values with
  no verdict attached; interpretation is left to the verdict generator's output, which this app
  also displays.

See [`docs/reference/methods.md`](../docs/reference/methods.md) for the full definition of each
metric and its evidentiary weight.
