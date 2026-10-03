# Understanding the joint-evidence experiment

The decomposition-by-evidence pilot left 15 supported TechQA sentences that the claim-entailment evaluator still rejected even with annotated evidence. A residual review categorized 10 of them as `multi_sentence_support`: the claim seemed to need facts from more than one source sentence. This experiment asks:

> If the verifier sees the relevant source sentences together instead of one at a time, does it stop rejecting claims that really are supported, without starting to accept claims that are not?

The exact protocol, numbers, and commands live in [Benchmarking and current evidence](../reference/benchmarks.md#joint-evidence-result) and the frozen [`protocol.md`](../../backend/evals/joint_evidence/v1-techqa/protocol.md). This page explains the design calls behind them and how to read the result.

## Separate versus joint: what the verifier sees

The verifier is a sentence-pair model: it receives one premise (source text) and one hypothesis (the claim) and returns entailment, neutral, and contradiction probabilities. Take this invented passage from the experiment's public synthetic fixture:

> (1) Fix pack 4.2.0.1 resolves this problem. (2) It restores the previous heap default and adds a startup check.

and the claim *"Fix pack 4.2.0.1 restores the previous heap default."*

Scored **separately**, the verifier sees sentence (1) with the claim, which names the fix pack but says nothing about heap defaults, and then sentence (2) with the claim, which mentions heap defaults but only says "It". Neither pair entails the claim on its own. The earlier oracle condition keeps the best of the separate scores, so the claim can be rejected even though the passage supports it.

Scored **jointly**, the verifier sees "(1) (2)" as one premise. Now "It" can be resolved to the fix pack, and the claim follows. That is the hypothesis: some false rejections come from the one-sentence-at-a-time presentation, not from the verifier's judgment or the claim.

## Why the planned comparison could not test the hypothesis

Issue #32 originally planned to compare the annotated sentences scored separately (A) with the same annotated sentences given together (B). Before running anything, counting the inputs showed a problem:

- 28 of the 39 sentences have exactly one annotated evidence sentence. With one sentence, "separately" and "together" are the same input, so A and B cannot differ.
- All 10 `multi_sentence_support` cases are among those 28. In examples like the one above, RAGBench's annotation marks only one sentence; the reviewer judged that a second one was also needed.

Joint presentation of the *annotated* set therefore cannot help the cases that raised the question. The experiment added a condition, B⁺, that supplies the annotated sentences plus their immediate neighbors, because cross-sentence references ("it", "this problem", "step 3") usually point to an adjacent sentence. A new human review to add the missing sentences was considered and rejected: the only available reviewer had already seen the residual cases, so that review would not have been blind. B−A was kept but reported only descriptively. ADR-055 records this change.

## The calls behind the evidence

Each choice was made from input properties, such as sentence counts and token lengths, before any verifier outcome was seen. Choosing them after looking at results would have been tuning on the pilot.

- **One neighbor on each side (w = 1).** It is the smallest window that can resolve a reference to the previous sentence, and it adds the least text, which keeps the "more text" confound small. Token limits did not distinguish w = 1, 2, or 3: the same 3 sentences overflowed at every width. A wider window is a possible declared follow-up, not a retry.
- **Top-3 for the similarity-selected joint condition (C).** B⁺ supplies a median of 3 sentences (1 annotated plus 2 neighbors), so k = 3 makes C and B⁺ differ mainly in *how* evidence is chosen rather than *how much*.
- **Document order, single-space separator, no duplicates.** Every joint condition formats its premise the same way, so formatting cannot explain differences between them.
- **Never truncate.** The verifier's input limit is 512 tokens, and over-long inputs would otherwise be cut silently, dropping arbitrary evidence. The runner measures every pair first; a sentence with any over-budget pair is marked unavailable in that condition and reported separately.

## Telling better evidence apart from more text

Fewer false rejections alone would not show that joint evidence helps. Supplying more text can make an entailment model more willing to say "entailed" to anything. The design includes two controls for that.

**A size-matched distractor control (S).** For each supported sentence, S supplies the annotated sentences plus exactly as many extra sentences as B⁺ added, drawn at random from the same document but away from the evidence: more than 2 sentences from any annotated sentence, and never annotated themselves. If B⁺ beats A but S does just as well, the gain looks like "more text" rather than "the right neighboring context". Distractors come from the same document so their topic and style match the neighbors; low-similarity distractors would have been too easy for the verifier to ignore. A random sentence can still help by accident, so this control bounds the confound but cannot eliminate it.

**An unsupported-sentence control.** The same population has 330 sentences that RAGBench labels unsupported. If joint evidence makes the verifier more permissive, these should be accepted more often. They have no annotated evidence, so the B⁺ construction is applied around the similarity-selected sentence instead (W: top-1 plus neighbors), and C is applied as is. These sentences use the deterministic claims, because no human claim review exists for them.

## How the result was judged

Everything below was frozen before the run, including what would count as success.

**Transition counts first.** For each comparison, the report counts sentences *corrected* (wrong before, right now), *newly wrong*, unchanged, and unavailable. With 39 sentences, these counts say more than a single rate.

**Few clusters.** The 39 sentences come from only 7 examples, and sentences from the same example are not independent. A bootstrap that resamples examples was kept for continuity with the earlier experiments, but with 7 examples it tends to give intervals that are too narrow. The confirmatory check is an exact sign-flip test over examples: it asks how often flipping the direction of each example's change would produce a result at least as extreme as the one observed.

**Three possible labels.** B⁺ and C each get one of: *confounded by leniency* (the false-supported rate on unsupported sentences rises by more than 5 percentage points, or its interval lies entirely above zero); *supported* (not confounded, more corrections than new errors with sign-flip p below 0.05, and for B⁺, better than the distractor control); or *inconclusive*. The 5-point margin was chosen because, with 330 sentences in 40 examples, smaller differences are hard to tell apart from noise.

## What happened

Both labels came out **inconclusive**. Compared with separate scoring, B⁺ corrected 6 sentences and introduced 3 new errors, but the distractor control did as well as B⁺, so the improvement cannot be credited to local context. Neither condition was confounded by leniency; adding neighbors actually *reduced* false acceptances of unsupported sentences. Within the 10 motivating cases, B⁺ corrected 4 and broke none, but those cases generated the hypothesis, so they cannot confirm it. The [reference section](../reference/benchmarks.md#joint-evidence-result) has the full tables.

## What the design got wrong

Two problems only became visible once results existed. Both are worth checking for in future protocols.

- **The confirmatory test could not succeed.** With 7 examples, the smallest possible sign-flip p-value is 2/128 ≈ 0.016. But examples in which no sentence changes contribute nothing to the test. Only 4 examples changed under B⁺ versus A, so the smallest attainable p-value was 2/16 = 0.125. Because most sentences have a single annotated evidence sentence and the verifier is deterministic, many examples were always likely to stay unchanged. Checking before the run how many examples *could* change would have shown that the test could not reach 0.05.
- **One rule filtered nothing.** The supported label required that treating unavailable sentences as unchanged would not alter the conclusion. Unchanged sentences add zero to both the counts and the test statistic, so that condition always holds. It did no harm, but it looked like a safeguard it was not.

## What this experiment does not establish

It does not show that multi-sentence support is absent, nor that the verifier can or cannot reason across sentences. An inconclusive result leaves claim representation, verifier behavior, and annotation ambiguity unresolved. The population is TechQA only and small; a separately frozen sample would be needed to test whether any pattern generalizes. The threshold was set for single-sentence premises and held fixed, so joint premises may score on a different scale. The supported-set and subgroup numbers depend on private review artifacts and cannot be reproduced without them; the unsupported-set numbers can.
