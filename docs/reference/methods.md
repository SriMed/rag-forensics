# Methods and architecture

This document describes how RAG Forensics produces its observations. None of the methods below
should be interpreted as independently proving a pipeline root cause.

## Diagnostic modules

| Module | Output | Interpretation constraint |
|---|---|---|
| RAGAS metrics | faithfulness and context-precision scores | model-judged evaluation, not a causal explanation |
| Retrieval distribution | top score, gap, normalized entropy, decay, tail mass | score shape cannot distinguish uniformly good from uniformly bad retrieval |
| Embedding analysis | centroid distance, chunk spread, query isolation, PCA projection | geometry describes the retrieved set, not corpus answerability |
| Semantic attribution | nearest source candidate and similarity for each answer unit | similarity is neither entailment nor contradiction detection |
| Hedging mismatch | confidence markers compared with available evidence | lexicon and evidence-model coverage bound the result |
| Retrieved-context fit | questions the retrieved passages appear able to answer | conditional hypothesis about retrieved content, not the full corpus |

The verdict generator converts these observations into a deterministic ordering of
`verdict_signals`. Each signal identifies its score as a heuristic priority and exposes a
reliability class. Recommendation prose uses the highest-ranked observations to propose
follow-up tests.

For a non-technical walkthrough of evidence candidates, competing hypotheses, and follow-up
tests, see [How RAG Forensics investigates an answer](../explainers/how-rag-forensics-works.md).

## Interpretation examples

### Retrieval shape

A large score gap and small tail mass describe retrieval dominated by one result. That result
may be highly relevant or merely the least-bad match. Absolute relevance and documented score
semantics are still required.

A flat distribution similarly has two competing explanations: several strong results or several
weak results. Entropy alone cannot choose between them.

### Source attribution

An answer sentence with a low maximum cosine similarity lacks a close semantic source candidate
under the configured model. This is useful for review, but it can miss valid synthesis and can
accept contradictions, changed numbers, or removed qualifiers.

### Retrieved-context fit

If the retrieved passages appear to answer questions adjacent to the user’s question, query
reformulation is a reasonable intervention to test. If they answer distant questions, retrieval
or corpus coverage deserves investigation. Neither observation proves that query wording caused
the failure or that the corpus lacks the answer.

## Architecture

```text
POST /example or /analyze/custom
  -> routers/
  -> services/
     -> retriever.py
     -> generator.py
     -> ragas_scorer.py
     -> forensics/
        -> retrieval_distribution.py
        -> embedding_analysis.py
        -> chunk_attribution.py
        -> hedging_mismatch.py
        -> query_corpus_fit.py
     -> verdict_generator.py

Offline evaluation
  -> benchmark/ragbench.py or benchmark/ragtruth.py
  -> benchmark/grounding.py
  -> benchmark/experiment.py
  -> machine-readable report
```

Numeric forensics modules return observations rather than verdicts. LLM-backed modules expose
errors explicitly. The offline benchmark defaults to local embedding and NLI models and does not
call Anthropic or RAGAS.

## Offline grounding methods are evaluation tools

B1, B2, and B3 belong to the offline benchmark path shown above. They test whether particular
grounding signals correspond to RAGBench labels; they are not additional interactive product
modules.

B3 performs four steps:

1. deterministically split each response sentence into smaller claims;
2. choose the most similar document sentence for each claim;
3. score each claim/evidence pair with a pinned pretrained NLI cross-encoder; and
4. mark the parent sentence supported only when all evaluated claims pass the frozen threshold.

The NLI cross-encoder is a third-party general-purpose model, not a verifier developed or trained
by RAG Forensics. Its suitability is still part of this project's measurement validity because B3
uses its scores to make grounding decisions. The benchmark therefore tests the assembled method,
preserves verifier errors as unknown states, and avoids treating its outputs as ground truth.

## Known limitations

- Similarity thresholds are useful operational cutoffs, not proof of grounding.
- The deterministic claim decomposer splits clauses syntactically and has not been validated as
  a semantic atomicity model.
- Claim verification currently selects the closest individual evidence sentence; genuine
  multi-source synthesis may require evidence aggregation.
- The general-purpose NLI model is not trained specifically for heterogeneous RAG documents,
  tables, or financial calculation.
- Hedging classification is lexicon-bounded; unseen constructions default toward definitive.
- Retrieved-context-fit trigger thresholds and verdict priorities remain heuristic.
- `/analyze/custom` re-embeds caller text with the project’s MiniLM model, which may not match the
  caller’s production retriever space.
- Model or evaluator failure is an unknown state and must not be rendered as a clean result.

## Prior-work boundary

The project’s intended contribution is instance-level diagnostic transparency: preserving
observations, provenance, competing hypotheses, and proposed interventions for one answer. It is
not architecture search, automatic RAG optimization, or a claim of state-of-the-art hallucination
detection.
