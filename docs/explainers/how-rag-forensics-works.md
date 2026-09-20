# How RAG Forensics investigates an answer

RAG Forensics helps a developer investigate one RAG answer. It does not decide the true root
cause automatically. Instead, it organizes what the system observed, which explanations remain
plausible, and which intervention would best distinguish among them.

## A running example

Suppose a user asks:

> What were Acme's 2024 revenue and operating costs?

The RAG system retrieves these passages:

1. “Acme reported 2024 revenue of $20 million.”
2. “The company issued new shares in May.”
3. “Operating costs remained flat at $12 million.”

It answers:

> Acme earned $20 million in revenue, and operating costs doubled to $24 million.

The answer contains one supported claim and one changed number. A single score may flag the
answer, but it does not explain whether retrieval, generation, or evaluation produced the
problem.

## What RAG Forensics contributes

For this answer, RAG Forensics can produce three kinds of investigative output:

### 1. Evidence candidates

An evidence candidate is a retrieved passage that appears related to an answer sentence or claim.
For example:

- “Acme reported 2024 revenue of $20 million” is a strong candidate for the revenue claim.
- “Operating costs remained flat at $12 million” is a candidate for the costs claim.

“Candidate” is deliberate: semantic similarity can help a reviewer find relevant text, but it
does not prove that the text entails the answer. A contradiction can use nearly identical words.

### 2. Competing failure hypotheses

The observations may support several explanations at once:

- **Generation error:** the correct cost passage was retrieved, but the answer changed $12 million
  to $24 million.
- **Evidence-selection error:** an evaluator paired the cost claim with the unrelated share-issue
  passage and rejected it for the wrong reason.
- **Verifier error:** an evaluator received the correct cost passage but failed to recognize the
  relationship.
- **Multi-source reasoning limitation:** a claim may require combining several passages rather
  than checking one passage in isolation.

These are hypotheses, not diagnoses. The same low score can arise from more than one mechanism.

### 3. Suggested follow-up tests

A useful follow-up changes one part of the pipeline while holding the others fixed:

- Give the evaluator the known cost passage. If its judgment changes, evidence selection was part
  of the failure.
- Regenerate the answer with the same retrieved passages. If the number is corrected, generation
  instability deserves attention.
- Retrieve more passages or rewrite the query. If the needed evidence appears only then,
  retrieval coverage deserves attention.
- Ask a reviewer to inspect the claim and its candidate evidence when automated evaluators
  disagree or fail.

RAG Forensics ranks such leads using heuristic priorities and reliability labels. Those priorities
are investigation orderings—not probabilities, calibrated severities, or causal attributions.

## Where the grounding evaluators fit

Support checking matters because it helps a developer decide what to investigate. If the retrieved
text contains the needed information but the answer contradicts it, answer generation deserves
attention. If the needed information was not retrieved, retrieval deserves attention. If the answer
accurately reflects a source that is itself wrong, source quality deserves attention. A support
judgment is one piece of evidence for these decisions; it cannot establish a cause by itself.

The evaluator can also be wrong. An incorrect rejection can send someone investigating an answer
that was already supported, while an incorrect acceptance can hide a real problem. The offline
grounding experiments test whether a proposed support signal is dependable enough for this role.

Why compare similarity with entailment? Consider this illustrative pair:

> Source: “The treatment did not improve survival.”
>
> Answer: “The treatment improved survival.”

The texts are very similar, but the answer reverses the source's meaning. Similarity can help find
the relevant passage without establishing that it supports the answer. **Entailment** asks whether
the claim follows from the supplied text. A pretrained natural language inference (**NLI**) model
attempts that judgment. Its training objective makes it a plausible tool to test, not a guarantee
that it will judge support reliably on this project's data.

One offline benchmark compares:

- two prevalence checks that always predict supported or always predict unsupported;
- a **whole-sentence similarity evaluator**;
- a **claim-similarity evaluator** using deterministic claim splitting; and
- a **claim-entailment evaluator** using the same claims and evidence candidates with an NLI verifier.

This sequence isolates what changes: the second evaluator adds claim splitting, while the third
changes the scoring method from similarity to entailment-aware verification. These are evaluation
conditions, not successive product versions.

The two claim-based methods receive the same claim and the same selected source sentence, so their
comparison tests the added support judgment without also changing the supplied evidence. This is a
bounded experimental setup, not an assumption that every claim can be supported by one sentence.

The claim-entailment evaluator is an offline benchmark method used to test one possible grounding
signal. It is not the whole RAG Forensics product. It:

1. splits a response sentence into smaller claims;
2. selects the most semantically similar source sentence for each claim;
3. sends each claim and selected source sentence to a pretrained NLI verifier; and
4. combines the claim judgments into a supported/unsupported sentence decision.

The NLI verifier is a pinned third-party pretrained model. RAG Forensics did not train or invent
it. The project evaluates it because the claim-entailment evaluator relies on its output: before presenting a component's score
as a useful diagnostic signal, the project must test whether that component is suitable for the
assigned job.

The held-out benchmark did not show that the claim-entailment evaluator was a reliable standalone grounding detector. Its
value in the project is experimental: its preserved intermediate steps make evidence-selection,
claim-decomposition, and verifier failures easier to distinguish.

## Where the oracle condition fits

Normally, the claim-entailment evaluator both chooses evidence and verifies it. If it rejects a supported answer, the final
decision does not reveal which step failed.

An **oracle condition** is an experimental diagnostic in which the benchmark supplies the trusted
answer to one intermediate step so the next step can be tested separately. In this oracle
condition, RAGBench's human-annotated supporting sentences temporarily replace the evidence chosen
by the claim-entailment evaluator; each is still checked separately. This is like guiding a delivery driver to the correct address
so you can test whether the driver can complete the delivery. If the verifier succeeds only after
receiving annotated evidence, evidence selection contributed to the original failure. If it still
fails, downstream explanations remain.

The annotations are unavailable for new production answers, so this is a diagnostic experiment,
not a deployable feature. See [Understanding the oracle-evidence experiment](oracle-evidence.md)
for the full design, result, and limitations.

## The responsible product claim

RAG Forensics makes a RAG failure easier to inspect by preserving observations, candidate
evidence, method assumptions, reliability, competing explanations, and discriminating follow-up
tests. It narrows an investigation; it does not prove why an answer failed.
