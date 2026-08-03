# Understanding the oracle-evidence experiment

The oracle-evidence diagnostic is designed to answer a narrow but important question:

> When the grounding pipeline rejects a supported answer, did it choose the wrong evidence, or
> did the verifier misunderstand the right evidence?

## The two jobs inside the pipeline

Consider an answer containing this sentence:

> The company earned $20 million in 2024.

The source documents contain:

> Revenue for 2024 was $20 million.

The current B3 method performs two jobs:

1. **Evidence selection:** find the source sentence most relevant to the answer's claim.
2. **Verification:** decide whether the selected source sentence supports the claim.

If B3 incorrectly reports that the answer is unsupported, the final result alone does not reveal
which job failed. It may have selected an irrelevant sentence, or it may have selected suitable
evidence that the verifier failed to interpret.

## What “oracle evidence” means

RAGBench includes human-provided annotations identifying the source sentences that support many
correct response sentences. In the experiment, those annotated sentences are called **oracle
evidence**.

“Oracle” does not mean that the annotations are infallible or that the experiment has access to
magic information. It means that one intermediate task—choosing evidence—is temporarily supplied
by the benchmark so the following verification task can be tested separately.

For example:

| Condition | Evidence given to the verifier | Verifier result |
|---|---|---|
| Normal B3 | “Operating costs were $12 million.” | Unsupported |
| Oracle evidence | “Revenue for 2024 was $20 million.” | Supported |

If the verifier succeeds when given the annotated evidence, the original error is consistent
with an evidence-selection problem.

In a different case:

| Condition | Evidence given to the verifier | Verifier result |
|---|---|---|
| Normal B3 | The annotated supporting sentence | Unsupported |
| Oracle evidence | The same annotated supporting sentence | Unsupported |

Here, improved evidence selection does not resolve the error. The remaining explanations include
a verifier failure, an error in splitting the answer into smaller claims, or a mismatch between
the benchmark's annotation granularity and the verifier's task.

## What the experiment can establish

The experiment can help localize false rejections of supported response sentences:

- If annotated evidence resolves most errors, evidence selection is likely an important
  bottleneck.
- If errors persist with annotated evidence, verifier behavior, claim decomposition, or annotation
  granularity deserves more attention.
- If the result varies across domains or answer types, the project should not claim a single
  universal bottleneck.

This is a diagnostic intervention, not a new production method. A deployed RAG system would not
have RAGBench's human annotations available when analyzing a new answer.

## What the experiment cannot establish

The oracle condition is most interpretable for sentences labeled as supported. For a supported
sentence, an annotator can identify the source material that supports it. For an unsupported
sentence, there may be no corresponding “correct negative evidence” sentence: the necessary
information may be absent, distributed across several passages, or contradicted only indirectly.

Using labels to manufacture oracle evidence for unsupported sentences could therefore leak the
benchmark answer into the method being evaluated. It would make the method appear more capable
than a deployable system really is.

Accordingly, this experiment should not be described as:

- a new B4 grounding detector;
- a direct improvement to overall classification F1; or
- proof of the cause of every grounding error.

Its defensible purpose is narrower: **give the verifier known supporting material to determine
whether evidence selection or downstream checking better explains supported-sentence failures.**

## Why this matters

Without this intervention, replacing the retriever, claim splitter, or verifier would be an
undirected model change. The oracle comparison provides decision value:

- improve evidence selection when correct evidence fixes the failures;
- improve verification or claim decomposition when it does not; and
- investigate domain-specific behavior when the result is mixed.

This preserves the project's central boundary: RAG Forensics narrows an investigation by making
competing explanations testable; it does not turn a diagnostic signal into a causal conclusion.
