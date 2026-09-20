# Understanding the oracle-evidence experiment

The oracle-evidence diagnostic is designed to answer a narrow but important question:

> When the grounding pipeline rejects a supported answer, did it choose the wrong evidence, or
> do errors persist when it receives the annotated supporting evidence?

An **oracle condition** temporarily supplies the trusted answer to one intermediate step so that a
later step can be tested separately. A useful analogy is guiding a delivery driver to the correct
address: if the package is delivered only with guidance, navigation was part of the problem; if
delivery still fails, navigation was not the whole problem.

This is an offline experiment on the evaluator, not a diagnosis of the chatbot that produced the
answer. A faulty support judgment could direct a developer toward changing retrieval or generation
when the answer was already supported. For why the project compares text similarity with a model
that judges whether a claim follows from a source, see
[where the grounding evaluators fit](how-rag-forensics-works.md#where-the-grounding-evaluators-fit).

## The two jobs inside the pipeline

Consider an answer containing this sentence:

> The company earned $20 million in 2024.

The source documents contain:

> Revenue for 2024 was $20 million.

The **claim-entailment evaluator** splits an answer into claims, selects evidence by semantic
similarity, and uses an NLI verifier to judge support. After claim splitting, it performs two jobs
relevant to this experiment:

1. **Evidence selection:** find the source sentence most relevant to the answer's claim.
2. **Verification:** decide whether the selected source sentence supports the claim.

The verifier is a pinned third-party pretrained NLI model. RAG Forensics did not create it. The
project tests it because the claim-entailment evaluator uses its output to make a
supported/unsupported decision; validating an assembled diagnostic requires checking whether each
relied-upon component is suitable for its assigned role.

If the claim-entailment evaluator incorrectly reports that the answer is unsupported, the final
result alone does not reveal which job failed. It may have selected an irrelevant sentence, or it
may have selected suitable evidence that the verifier failed to interpret.

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
| Selected-evidence condition | “Operating costs were $12 million.” | Unsupported |
| Oracle evidence | “Revenue for 2024 was $20 million.” | Supported |

If the verifier succeeds when given the annotated evidence, the original error is consistent
with an evidence-selection problem.

In a different case:

| Condition | Evidence given to the verifier | Verifier result |
|---|---|---|
| Selected-evidence condition | The annotated supporting sentence | Unsupported |
| Oracle evidence | The same annotated supporting sentence | Unsupported |

Here, improved evidence selection does not resolve the error. The remaining explanations include
a verifier failure, an error in splitting the answer into smaller claims, or a mismatch between
the benchmark's annotation granularity and the verifier's task. Evidence may also need to be
presented together rather than one sentence at a time.

## Why the experiment checks source sentences separately

The grounding comparison starts with one source sentence selected for each claim. Both the
claim-similarity and claim-entailment methods receive that same pair, allowing the experiment to
compare their support judgments without changing evidence selection at the same time.

The oracle experiment then substitutes the human-annotated supporting sentences for automatically
selected evidence. It preserves the individual-sentence checks: each annotated sentence is scored
against each claim, and the highest support score is retained for that claim. This tests a change
in the evidence supplied while keeping the verifier's input format fixed. It does not test whether
the verifier can use several sentences together. See the
[exact oracle protocol](../reference/benchmarks.md#oracle-evidence-failure-localization).

For example, consider these invented source sentences:

> The policy covers temporary employees. They become eligible after six months.

Together they support “Temporary employees become eligible after six months.” Neither sentence
alone supplies both the group and the waiting period. Trying every annotated sentence separately
does not restore that connection. The single-sentence setup is therefore an experimental
simplification with a concrete limitation, not a claim that source sentences are self-contained.

The subsequent [decomposition-by-evidence pilot](decomposition-by-evidence.md#a-preliminary-finding-from-the-techqa-pilot)
raised this possibility when supported answers were still rejected after claim review and annotated
evidence were supplied. That review motivates a test; it does not show that combining evidence will
fix those errors. The proposed follow-up compares separate and joint presentation of the same
evidence, then checks whether any benefit survives automatic evidence selection and whether more
unsupported answers are incorrectly accepted. The question is whether this particular evaluator
becomes more reliable, not whether multi-sentence support exists.

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

- a new deployable grounding detector;
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

## What the completed result means in counts

The seeded cross-domain run contained 188 eligible supported sentences. With the claim-entailment
evaluator's selected evidence, 85 were falsely rejected (`45.2%`). With annotated evidence, 54 were falsely rejected
(`28.7%`). At the sentence level, annotated evidence corrected 37 previous false rejections but
introduced 6 new ones, for a net reduction of 31—or 16.5 percentage points. The
example-clustered 95% interval ranged from a 10.1- to 23.0-point reduction.

This supports evidence selection as a meaningful bottleneck for this eligible population. It is
not the sole bottleneck: the 54 oracle-condition false rejections still admit verifier,
claim-decomposition, multi-sentence reasoning, and annotation-granularity explanations.
