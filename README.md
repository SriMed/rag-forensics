# RAG Forensics

> An inspectable diagnostic record for investigating why a RAG answer may have gone wrong.

RAG evaluations commonly collapse a response into scores such as faithfulness or relevance.
Those scores can identify concern, but they rarely identify a unique cause. A weak answer may
reflect retrieval, missing evidence, contradiction, changed qualifiers, generation, or simply a
failed evaluator. Treating one observed score as proof of one cause hides those alternatives.

RAG Forensics keeps them visible. Given a question, answer, and retrieved context, it records
observable signals, evidence candidates, method assumptions, reliability labels, ranked
hypotheses, and follow-up tests. It is a hypothesis-generation layer—not a root-cause oracle.

## The argument

Useful reasoning transparency requires more than displaying intermediate numbers. A diagnostic
system should distinguish:

1. **what it observed**;
2. **how that observation was produced**;
3. **what the observation can and cannot establish**;
4. **which competing explanations remain**; and
5. **what intervention could discriminate between them**.

RAG Forensics implements that contract at the level of individual answers.

| Observation | Supports investigating | Does not establish |
|---|---|---|
| Flat or sharply decaying retrieval scores | retriever uncertainty or concentration | whether any retrieved passage is correct |
| Query isolated from retrieved embeddings | retrieved-context mismatch | that the full corpus lacks an answer |
| Low answer-to-context similarity | a weak source candidate | contradiction, hallucination, or lack of entailment |
| Definitive language with weak evidence | possible epistemic overstatement | the correct confidence level |
| Retrieved content answers a different question | query/retrieval intervention | whether query wording caused the failure |

The verdict layer ranks these signals as heuristic priorities. Its values are not probabilities,
calibrated severities, or causal attributions.

## What the system produces

- retrieval-distribution and embedding-space observations;
- sentence- or claim-level source candidates with raw scores;
- hedging/evidence mismatch observations;
- conditional retrieved-context-fit hypotheses;
- explicit evaluator failures instead of healthy-looking zeroes;
- ranked investigation signals and falsifiable follow-up tests;
- label-preserving RAGBench reports and explicit span-to-sentence RAGTruth reports.

The API exposes two entry points:

- `POST /example` runs the demonstration pipeline over a stored RAGBench example.
- `POST /analyze/custom` accepts a caller-provided question, answer, and retrieved chunks.

See [Methods and architecture](docs/METHODS.md) and
[integration documentation](README_INTEGRATION.md) for the full contracts.

## What the evidence currently says

The project now evaluates its grounding methods without regenerating RAGBench answers or changing
their source documents. Thresholds are selected on validation data and evaluated on untouched
test data with clustered confidence intervals.

The principal comparison is:

- **B1:** whole-sentence embedding similarity;
- **B2:** deterministic claim decomposition plus similarity;
- **B3:** the same claims and evidence candidates scored by a pinned NLI cross-encoder.

On a seeded sample of up to 100 validation and 100 test examples from each RAGBench domain:

| Held-out macro metric | B1 | B3 |
|---|---:|---:|
| F1 | 0.301 | 0.278 |
| AUPRC | 0.215 | 0.247 |

The paired B3−B1 macro-F1 difference was `-0.022` with a 95% interval of
`[-0.066, 0.025]`. Macro AUPRC increased, but its interval also included no improvement, and the
direction varied by domain. A small RAGTruth external-validation run was similarly mixed.

Therefore the current evidence does **not** support either whole-sentence similarity or the
present claim-plus-NLI pipeline as a reliable standalone grounding detector. Similarity remains
useful for navigating to candidate evidence. The stronger contribution is the transparent,
label-preserving framework that makes this null result—and its remaining uncertainty—inspectable.

Detailed protocols, commands, revisions, results, and limitations are in
[Benchmarking and current evidence](docs/BENCHMARKS.md).

## Current research boundary

The next useful experiment is not another unstructured model swap. It should separate:

- claim-decomposition errors;
- evidence-selection errors;
- multi-sentence or numerical reasoning failures;
- verifier errors; and
- annotation-granularity mismatches.

An oracle-evidence condition using RAGBench’s annotated supporting sentences can distinguish
evidence-selection failure from verifier failure. Until such interventions are run, the project
should claim that it narrows an investigation—not that it explains the cause of a bad answer.

## Quick start

```bash
cd backend
poetry install
cp .env.example .env
poetry run python scripts/bootstrap_data.py
poetry run uvicorn main:app --reload
```

Run the offline test suite:

```bash
cd backend
poetry run pytest
```

External API calls are mocked in tests; no API key is required.

## Documentation

- [Methods, outputs, architecture, and limitations](docs/METHODS.md)
- [Benchmark protocol, results, and reproducible commands](docs/BENCHMARKS.md)
- [Custom API integration](README_INTEGRATION.md)
- [Architectural decisions](ADR.md)
