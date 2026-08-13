# Methods and architecture

This document describes how RAG Forensics produces its observations. None of the methods below
should be interpreted as independently proving a pipeline root cause.

## Diagnostic modules

| Module | Output | Interpretation constraint |
|---|---|---|
| RAGAS metrics | faithfulness and answer-conditioned context-utilization scores | model-judged evaluation, not a causal explanation; failures are explicitly unavailable |
| Retrieval distribution | top score, gap, normalized entropy, decay, tail mass | score shape cannot distinguish uniformly good from uniformly bad retrieval |
| Embedding analysis | centroid distance, chunk spread, query isolation, PCA projection | geometry describes the retrieved set, not corpus answerability |
| Semantic attribution | nearest source candidate and similarity for each answer unit | similarity is neither entailment nor contradiction detection |
| Hedging mismatch | confidence markers compared with available evidence | lexicon and evidence-model coverage bound the result |
| Retrieved-context fit | questions the retrieved passages appear able to answer | conditional hypothesis about retrieved content, not the full corpus |

The verdict generator converts these observations into a deterministic ordering of
`verdict_signals` and an inspectable `verdict_reasoning` object. The latter preserves up to three
observations with reliability labels, materially different hypotheses, a named component under
test, one test action, and outcome interpretations that identify which hypothesis each result
supports. When retrieval and generation signals coexist, they remain competing explanations
rather than being collapsed into a single asserted cause.

Claude receives only this constructed structure and may word it as recommendation prose; it is not
asked to invent causes or tests. If rendering fails, the API deterministically formats the complete
structure instead of falling back to one signal. Unavailable evaluations become missing-evidence
observations with a restore-and-rerun test, never healthy zeros.

Answer generation also receives an explicit source-boundary contract on every retrieved chunk.
Completeness is `complete`, `truncated`, or `unknown`, with separate provenance. Known-truncated
chunks instruct generation not to guess the missing continuation and to disclose material
incompleteness. Unknown is an unavailable metadata state, not evidence of either completeness or
truncation. RAGBench's stored document strings lack source-boundary provenance and therefore remain
unknown; punctuation heuristics do not change that state. See the
[truncated-evidence evaluation](truncated-evidence.md) for the exact-model comparison and limits.

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

When triggered, question generation returns three to five structured candidates with cited chunk
IDs. A separate structured model judgment checks whether each candidate is directly answerable
from its cited chunks and sufficiently specific. Citations outside the retrieved set are invalid, and accepted question
embeddings must remain below `0.90` pairwise cosine similarity to exclude semantic duplicates.
Only three or more accepted questions can produce a retrieved-context fit label. Otherwise the
result has `status="error"`, `error="insufficient_valid_questions"`, and no `observed_fit` or mean
similarity. Accepted and rejected candidates remain inspectable; rejection reasons distinguish
unsupported or nonspecific questions, semantic duplicates, and invalid chunk citations. These labels concern
only the retrieved passages and never establish full-corpus coverage.

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
- Verdict hypothesis and test templates are deterministic diagnostic defaults, not proof that the
  named component caused an observed failure.
- `/analyze/custom` re-embeds caller text with the project’s MiniLM model, which may not match the
  caller’s production retriever space.
- Model or evaluator failure is an unknown state and must not be rendered as a clean result.
- Hedging mismatch fractions exclude claims for which every entailment attempt is invalid or
  failed; inspect evaluated and unavailable claim counts before interpreting the fraction.

## Prior-work boundary

The project’s intended contribution is instance-level diagnostic transparency: preserving
observations, provenance, competing hypotheses, and proposed interventions for one answer. It is
not architecture search, automatic RAG optimization, or a claim of state-of-the-art hallucination
detection.
