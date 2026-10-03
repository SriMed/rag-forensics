# Joint evidence verification protocol, v1 — TechQA-only

**Status:** frozen 2026-10-03, before any outcome of this experiment was computed. **Issue:** #32. **Amendment rationale:** ADR-055.

Any change to this file after freezing requires a new protocol version (`v2`), not an edit. Results produced under a changed protocol are not results of this protocol.

## Question

Does giving the verifier evidence sentences together, rather than scoring each sentence separately, reduce false "unsupported" judgments on supported sentences, and does any gain survive when evidence is selected by similarity instead of taken from the annotations?

## Amendment to the issue's design

Issue #32 names joint annotated evidence (B) against separate annotated evidence (A) as the primary comparison. Input statistics computed before freezing, with no verifier outcomes inspected, show that this comparison cannot test the cases that motivated the issue:

- 28 of the 39 supported sentences have exactly one annotated evidence sentence, so B's input is identical to A's for them. The 11 multi-sentence evidence sets (sizes 2, 3, 6, and 12) come from 3 of the 7 examples.
- All 10 sentences categorized `multi_sentence_support` in the #29 residual review have exactly one annotated evidence sentence. The review judged that support needed further sentences, but RAGBench annotates only one, so B cannot change their outcome.

The primary condition is therefore **B⁺**: the annotated sentences plus their immediate neighbors, presented jointly. B−A is kept and reported descriptively. The owner confirmed this amendment on 2026-10-03.

## Frozen inputs

| Item | Value |
|---|---|
| Dataset | `galileo-ai/ragbench`, revision `97808f3e5fd16ede40bbff6c2949af8139b2eb7b` |
| Population | TechQA, `test` split, `--evaluation-limit 100`, seed 42 (the #29 v1-techqa population) |
| Eligible-population hash | `population_sha256` `0512b68ad8d2c08564f0001dc76c5ba570289bbc2657ca163fb8ab0868acc977` |
| Reviewed claims (supported set) | `../../decomposition_evidence/v1-techqa/claim-review.json`, revision 1, sha256 `049cb1f15687be26661195f811694ece72e7deb2866e231288bb85586b9e7c39` (private) |
| Exploratory subgroup membership | `../../decomposition_evidence/v1-techqa/residual-review.json`, sha256 `5d7a83de0a775e0ebb5152f340f5c0d0a9a2f4db6798b2026b4adc03b3fb6ad1` (private) |
| Deterministic claims (unsupported set) | `DeterministicClaimDecomposer`, name `deterministic_clause`, version `1` |
| Verifier and tokenizer | `cross-encoder/nli-deberta-v3-base`, revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` |
| Entailment threshold | `0.0017914474026707317` (the #29 value; not tuned on this experiment, per ADR-051) |
| Selection embedding | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Aggregation | A sentence is supported only if every claim is supported (`all_claims_supported`) |
| Seed | 42 for distractor draws and the bootstrap |
| Bootstrap | 2,000 iterations, clusters = `domain:example_id` |

The runner must reject inputs whose hashes, revisions, or population counts differ from this table.

## Populations

**Supported set.** The 39 eligible supported sentences in 7 examples (sentences per example: 3, 3, 3, 4, 7, 9, 10), with the frozen reviewed claims. Every sentence is reported, including those unavailable in a condition. No sentence is selected or dropped based on the #29 results.

**Unsupported control set.** Every sentence in the same 100-record population that RAGBench labels unsupported (330 sentences in 40 examples at freezing), with deterministic claims. There is no oracle evidence for these sentences, and none is constructed.

**Exploratory subgroup.** The 10 supported sentences categorized `multi_sentence_support` in the #29 residual review. Identified from the cases that generated the hypothesis, so it is not an independent confirmatory sample.

## Evidence construction

Common rules for every joint condition:

- **Order:** selected sentences are presented in document order (record order of `document_sentences`), never rank order.
- **Separator:** a single space between sentences.
- **Deduplication:** each document sentence appears at most once in a premise.
- **Document boundaries:** neighbor windows never cross a `document_id` boundary.
- **Verifier input:** `(premise, claim)`, as in the existing verifier adapter.

### Supported set

| Condition | Premise per claim | Role |
|---|---|---|
| A | Each annotated sentence scored separately; maximum entailment per claim | Existing oracle baseline (#29 condition D) |
| B | All annotated sentences, jointly | Descriptive only |
| B⁺ | Annotated sentences plus ±1 neighboring sentence of each (w = 1), merged, jointly | **Primary** |
| S | Annotated sentences plus *n* distractors, jointly, where *n* = sentences B⁺ added for this response sentence | Evidence-size control for B⁺ |
| T1 | Top-1 document sentence by claim cosine similarity | Selected reference (#29 condition C) |
| C | Top-3 document sentences by claim cosine similarity, jointly | Selected joint evidence |

**Selection (T1, C).** Cosine similarity between the claim embedding and every document sentence of the record; that same full pool is used for T1 and C. Ties are broken by earlier document position (stable sort). k = 3 matches B⁺'s median evidence size (1 annotated + 2 neighbors), so C and B⁺ differ in how evidence is chosen rather than how much.

**Distractors (S).** Eligible distractors come from the documents containing the annotated sentences, excluding:

- every sentence annotated as supporting *any* response sentence of the record;
- every sentence within ±2 positions of an annotated sentence for this response sentence;
- empty or whitespace-only sentences.

*n* distractors are drawn without replacement from the eligible list (in document order) by a NumPy `default_rng` seeded with the first 8 bytes, big-endian, of `sha256("42:{domain}:{example_id}:{sentence_key}")`. The draw is therefore independent of processing order. If fewer than *n* eligible distractors exist, the sentence is `control_unavailable` in S; it is not padded from other documents. Limitation: absence from the support annotations does not prove a sentence irrelevant, so a distractor can still help by accident.

### Unsupported control set

| Condition | Premise per claim |
|---|---|
| T1 | Top-1 selected sentence (as above) |
| C | Top-3 selected sentences, jointly (as above) |
| W | Top-1 selected sentence plus its ±1 neighbors, jointly (the B⁺ construction anchored on the selected sentence) |

W is the false-supported check for B⁺; C is the false-supported check for C.

## Token budget and unavailable outcomes

- **Budget:** 512 tokens for the tokenized `(premise, claim)` pair including special tokens, measured with the pinned verifier tokenizer.
- **Over budget:** if any claim pair of a sentence exceeds the budget in a condition, the sentence is `over_budget` in that condition. It is never truncated and never sent to the verifier in truncated form. At freezing, 3 supported sentences (the 12-sentence evidence sets) exceed the budget under B and B⁺.
- **Verifier failure:** an exception becomes `verifier_error` for that claim.
- **Aggregation with unavailable claims:** any unavailable claim makes the sentence unavailable (`predicted_unsupported = None`), never unsupported.
- **Paired contrasts** use only sentences available in both conditions; unavailable sentences are counted by reason (`over_budget`, `verifier_error`, `control_unavailable`) for each condition.

Every evaluation records the verifier input text, evidence sentence keys, NLI scores, status, and the sentence-level outcome.

## Analysis

### Contrasts

| Contrast | Population | Status |
|---|---|---|
| B⁺ − A | Supported | Primary |
| B⁺ − S | Supported | Primary (evidence-size control) |
| C − T1 | Supported | Primary (practical) |
| B − A | Supported | Descriptive; only 11 sentences can differ |
| W − T1 (false supported) | Unsupported | Leniency check for B⁺ |
| C − T1 (false supported) | Unsupported | Leniency check for C |
| All supported contrasts | Exploratory subgroup | Transition counts only |

### Reported for each contrast

1. **Transition counts (primary result):** corrected (unsupported → supported on supported sentences; supported → unsupported on unsupported sentences), newly introduced errors, unchanged, and unavailable by reason. Plus a per-example table of paired differences.
2. **Example-clustered bootstrap interval** of the paired difference in error rate (the #29 implementation; 2,000 iterations, seed 42). With 7 supported-set clusters, percentile bootstrap intervals are likely too narrow, and the report must say so.
3. **Exact cluster sign-flip permutation test** (supported-set primary contrasts only). For each example *c*, let *D_c* be the sum over its paired sentences of (condition error − reference error). The statistic is Σ *D_c* / *N* over the *N* paired sentences. All 2^K sign assignments of the *D_c* are enumerated; the two-sided p-value is the share whose absolute statistic is at least the observed absolute statistic (with a 1e-12 tolerance). With K = 7, the smallest attainable p-value is 2/128 ≈ 0.016.

### Interpretation rules

Each of B⁺ and C receives exactly one label, decided in this order:

1. **Confounded by leniency:** on the unsupported set, the increase in false-supported rate for its leniency check (W − T1 for B⁺; C − T1 for C) exceeds 5 percentage points as a point estimate, or its clustered 95% interval lies entirely above zero.
2. **Supports a local-context bottleneck** (B⁺) or **supports a practical follow-up** (C): not confounded, and
   - for B⁺: B⁺ − A shows more corrected than introduced rejections with sign-flip p < 0.05; B⁺ − S points in the same direction with more corrected than introduced rejections; and treating every unavailable sentence as unchanged would not change these conclusions.
   - for C: C − T1 shows more corrected than introduced rejections with sign-flip p < 0.05, under the same unavailable-sentence condition.
3. **Inconclusive:** anything else. An inconclusive result leaves claim representation, verifier behavior, and annotation ambiguity unresolved; it does not show that multi-sentence support is absent.

## Known limitations

- The threshold was set for single-sentence premises. Joint premises may shift the entailment-score distribution; the threshold is held fixed regardless (ADR-051), and that is part of what is tested.
- Supported sentences use reviewed claims and unsupported sentences use deterministic claims, so the false-unsupported and false-supported rates are not one accuracy measure.
- RAGBench's unsupported labels are model-generated and may be noisy.
- The population is TechQA-only with 7 supported-set clusters. A separately frozen sample is required to test generalization.

## Privacy and publication

`results.json` for this protocol contains the private reviewed claims and is kept local, like the #29 artifacts. Published: this protocol, the code, a runnable synthetic fixture, and aggregate results with uncertainty in `docs/reference/benchmarks.md`. Supported-set and subgroup results cannot be reproduced independently without the private claim and residual reviews; unsupported-set results can, since they depend only on public data and the deterministic decomposer.
