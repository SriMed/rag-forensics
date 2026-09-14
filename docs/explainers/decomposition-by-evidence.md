# Understanding the decomposition-by-evidence experiment

The oracle-evidence diagnostic showed that supplying annotated evidence reduces false rejections
of supported sentences, but errors persist even with perfect evidence. This experiment asks the
next question:

> When the claim-entailment evaluator still rejects a genuinely-supported sentence, is that
> because the claim it was asked to check was itself malformed?

## What a grounding check is checking

A **grounding check** is any automated method that tries to answer, per sentence or per claim, "is
this actually backed up by the retrieved evidence, or did the model just make it up?" This project
has tried three, of increasing precision — whole-sentence similarity, claim-decomposition with
similarity, and claim-decomposition with entailment — described in full in [How RAG Forensics
investigates an answer](how-rag-forensics-works.md#where-the-grounding-evaluators-fit). This
experiment is about the first stage of the most precise of the three: the claim-decomposition step
that happens before any evidence is ever selected.

Two terms carry a lot of weight in what follows, defined precisely in [CONTEXT.md](../../CONTEXT.md):
a **claim** is a piece of the model's own answer, decomposed to be independently checkable; **evidence**
is a piece of the retrieved source material a claim is checked against. They never come from the
same place — one is what's being verified, the other is what it's verified against.

## The test for whether a claim is well-formed

Every decomposition decision in this experiment answers one question: *handed just this claim's
text, and one evidence candidate — no other sentence, no surrounding paragraph — could a verifier
tell what's being asserted and check it?*

Take a minimal example, unrelated to this project's actual dataset:

> Response: "The library closes at 9pm on weekdays. This means students should plan visits
> accordingly."

Take the second sentence as a claim on its own: *"This means students should plan visits
accordingly."* Hand that, alone, to a verifier along with the evidence *"Library hours: Mon–Fri
9am–9pm."* The verifier cannot judge it — it never saw the first sentence, so it doesn't know what
"this" refers to. Whatever it outputs is noise, not signal: not because the evidence is bad (it's
actually a perfect match), but because the claim string doesn't say anything checkable in
isolation. If this gets marked unsupported, that's a **false-unsupported** result caused entirely
by decomposition, not by evidence quality.

The corrected claim — *"Because the library closes at 9pm on weekdays, students should plan visits
accordingly"* — names the thing it's about. Same evidence, same verifier, but now there's an actual
question being asked. [The reviewer README](../../backend/evals/decomposition_evidence/v1/README.md#common-decomposition-failure-patterns)
catalogs the five ways this project's response sentences fail that test in practice, each with a
real before/after example.

## Why the review can't look at evidence

It's tempting, mid-review, to reason about a specific piece of evidence: "well, if the evidence
says something generic like 'students should always plan visits carefully,' this vague claim would
probably pass anyway." That reasoning is exactly what the review is designed to exclude, for two
reasons.

First, it's usually wrong in a way that matters: a vague claim matching generic evidence by
lexical overlap is a **false-supported** result, not a safe pass — the specific fact the sentence
actually asserted (that the 9pm closing time drives the recommendation) was never checked against
anything. Decomposition that happens to survive by accident hides exactly the kind of unverified
assertion this diagnostic exists to catch.

Second, and more fundamentally: the same claim gets re-evaluated against a different retrieved
evidence pool every time the pipeline runs on it. A decision made by imagining "what the evidence
probably looks like" only holds for the pool in front of you at that moment — it doesn't generalize
to the next run, the next domain, or the next retrieval method. The job is to make the claim
well-formed enough that *any* evidence-matching outcome is a meaningful measurement, not to guess
whether one particular outcome would look fine.

## Why claim count changes the measured outcome

A response sentence is marked supported only when **every** claim it decomposes into passes
verification (see [Methods and architecture](../reference/methods.md)). That aggregation rule means
the number of claims a sentence produces — its **aggregation exposure** — affects the measured
outcome independent of whether the decomposition is more or less correct, through two distinct
mechanisms:

1. **Genuine partial coverage.** If a sentence asserts something about both DASH and JazzSM but the
   retrieved evidence only actually discusses DASH, splitting into independent per-topic claims
   correctly surfaces that the JazzSM half is unverified — a real finding, not an artifact. Before
   decomposition review, that partial failure was invisible, hidden inside one blunt whole-sentence
   score.
2. **Independent verifier and retrieval noise.** Even where evidence genuinely covers every claim,
   each one is a separate roll of an imperfect similarity search and an imperfect entailment model.
   More claims means more independent chances for that ordinary noise to strike at least once,
   dragging a genuinely well-grounded sentence's aggregate verdict down for reasons that have
   nothing to do with the sentence's actual grounding.

From the outside — one sentence, one supported/unsupported verdict — these two mechanisms are
indistinguishable. That ambiguity is exactly why [the interpretation constraint in Benchmarking and
current evidence](../reference/benchmarks.md#decomposition-by-evidence-protocol) requires the
report to publish claim-count distributions and aggregation exposure alongside the headline effect,
rather than letting one clean-looking percentage stand for a decomposition-quality finding on its
own.

## How the four conditions isolate decomposition from evidence selection

The experiment holds the verifier, threshold, and evaluation population fixed and crosses two
independent manipulations — which claims are used, and which evidence is supplied:

| | similarity-selected evidence | oracle evidence |
|---|---|---|
| deterministic claims | A | B |
| reviewed claims | C | D |

Condition A and C are identical except for which claims feed evidence-matching and entailment.
Comparing their false-unsupported rates isolates the causal effect of the decomposition step alone:
if C's rate is meaningfully lower than A's, the reviewed claims caught sentences that were being
wrongly rejected specifically because of how they were split — decomposition quality was a real
contributor to the earlier mixed benchmark result. If C looks like A, decomposition wasn't the
bottleneck, and the earlier result is better explained by the verifier or by reasoning that needs
more than one sentence at a time. B versus D repeats the same isolation under oracle evidence, to
check whether decomposition still matters once evidence is no longer the limiting factor.

See [Benchmarking and current evidence](../reference/benchmarks.md#decomposition-by-evidence-protocol)
for the full protocol, the frozen threshold, and the blinded two-stage review process that produces
the claims used in conditions C and D.

## How this differs from a regular RAG Forensics request

None of the above happens when someone actually uses the product.

| | A live `/example` or `/analyze/custom` request | This experiment |
|---|---|---|
| Data | A real question, real retrieved chunks, a live-generated answer | RAGBench's pre-built questions, context, and answers |
| Ground truth | None — nobody knows if the answer is actually correct | RAGBench's human-annotated supported/unsupported labels |
| What's produced | Observations and heuristic priorities for a person to investigate | A measured error rate (false-unsupported rate, confidence intervals) |
| Aggregation exposure | Not tracked — there's no population to aggregate over | Central to interpreting the result |

The live product never scores itself against a known-correct answer, because none exists for a
real user's question — that's precisely why RAG Forensics produces inspectable hypotheses rather
than verdicts (see [How RAG Forensics investigates an answer](how-rag-forensics-works.md)). This
experiment exists to sanity-check, offline and against known labels, whether the claim-decomposition
step inside one of RAG Forensics' evaluation tools is trustworthy enough to keep relying on — before
that method's assumptions are extended anywhere near a live request.

## A preliminary finding from the TechQA pilot

A 39-item TechQA-only pilot run (not the full population — see the companion `v1-techqa`
protocol) surfaced a mechanism worth naming even at this small scale. Of the sentences still
false-unsupported under oracle evidence (condition D), the great majority were categorized
`multi_sentence_support` in the residual review — and in this pilot, **all** of them traced back to
one of two specific corrections: resolving a cross-sentence pronoun (failure pattern 4) or
resolving list-position context (failure pattern 5). Zero traced to `verifier_error`, numerical
reasoning, or an annotation-granularity mismatch.

The likely mechanism: the entailment step is only ever given one evidence sentence per claim.
Correctly resolving a reference or a list-header dependency — which atomicity requires — produces
a claim whose truth now depends on two facts stated in two different sentences, but the pipeline
never supplies a second evidence candidate to check the second half. That reads as a structural
ceiling in a single-evidence-sentence grounding check, not as evidence that the underlying RAG
answers require genuine multi-document reasoning, and not as a decomposition-quality finding either
— a correctly atomized claim is exactly what produces this failure mode.

This is a 15-item, single-domain, single-pilot observation. It should be treated as a hypothesis
to test at larger scale (does the pattern hold outside TechQA? does it hold with a top-k evidence
candidate set instead of top-1?), not as an established result.

## What this experiment does not establish

This experiment analyzes the measurement validity of the claim-entailment grounding evaluator — one
evaluation tool among several this project has benchmarked. It does not establish the effectiveness
of the complete RAG Forensics diagnostic record, and it does not differentiate that record from
related systems such as RAGChecker or RAGVUE (see [Related work in RAG evaluation and
debugging](../reference/related-work.md)). This is evaluator-component localization, not validation
of the diagnostic record a user actually receives.

- It does not prove that any single sentence's rejection was "caused by" decomposition — only that,
  in aggregate and holding evidence-selection fixed, corrected claims shift the measured rate.
- It does not produce a deployable improvement to the claim decomposer; the reviewed claims are a
  frozen, blinded research artifact, not a new production component.
- A result on one domain's population does not generalize to the others without also running them
  — domain composition should always be reported alongside the effect.
