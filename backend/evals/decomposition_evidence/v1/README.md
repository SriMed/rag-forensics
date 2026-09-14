# Decomposition-by-evidence review artifacts, v1

This directory contains the human-review inputs for issue #29. The artifacts are review records,
not ground truth. They target the same 188 eligible supported RAGBench sentences as the seeded
oracle-evidence diagnostic.

## Claim review

`claim-review.json` is deliberately committed as a draft. Each item contains only the original
question, full response, target sentence, current deterministic split, and review fields. It does
not expose selected evidence, annotated evidence, verifier scores, condition assignments, or
downstream outcomes.

For each item, decide whether every proposed clause is already a smallest independently
verifiable claim that preserves the sentence's entities, relations, quantities, negation, and
qualifiers. Use no evidence or experiment outputs while making this decision. Set
`accept_deterministic` to `true` when it is. Otherwise set it to `false` and write
the corrected claims in `reviewed_claims`. Do not decide whether the claims are supported and do
not inspect verifier scores. Put unresolved cases in the artifact-level `unresolved_judgments`.

After all 188 items have decisions, fill `reviewer_identity`, increment `revision` if the draft was
previously circulated, record any remaining uncertainty in `unresolved_judgments`, and set `status`
to `frozen`. An unresolved item still needs the reviewer's best executable decomposition decision.
The runner validates these conditions and refuses drafts or a population mismatch.

## Common decomposition failure patterns

Five patterns account for nearly all corrections observed reviewing this population so far. For
each, the test is the same: could a verifier, handed just this claim's text and one evidence
candidate — no other sentence, no surrounding paragraph — tell what's being asserted? See
[Understanding the decomposition-by-evidence experiment](../../../../docs/explainers/decomposition-by-evidence.md)
for why that test is the right one and what happens mechanically when a claim fails it.

**1. Narration artifact.** Framing about the response itself, not a fact about the subject, riding
at either end of the sentence. Strip it; the hedge inside stays.
- Deterministic: *"In the context provided, it is mentioned that when installing IBM WTX Design
  Studio 8.4.1.1 over an existing installation, the COBOL Copybook importer may be missing after
  the upgrade."* (`techqa_DEV_Q243`, sentence b)
- Reviewed: *"When installing IBM WTX Design Studio 8.4.1.1 over an existing installation, the
  COBOL Copybook importer may be missing after the upgrade."*

**2. Non-independent fragment.** The deterministic splitter cut a sentence into pieces that can't
be verified without a sibling fragment's subject or object.
- Deterministic: *"This will provide a detailed listing of the prerequisites for both DASH"* /
  *"JazzSM, showing which items have passed"* / *"which items have failed."* (`techqa_DEV_Q093`,
  sentence e)
- Reviewed: four independent claims, one per product/outcome combination — *"The prerequisite
  scanner script will provide a detailed listing of the prerequisites for JazzSM, showing which
  items have failed,"* and likewise for DASH-passed, DASH-failed, JazzSM-passed. Splitting further
  than the deterministic split, not merging, was correct here: each combination is independently
  checkable, so smallest means the finer decomposition.

**3. Dangling connective.** A clause opening with "then" or "and" has no subject of its own and
also drops the ordering relation ("do X, then Y") if split away from what precedes it.
- Deterministic: *"To resolve this problem, it is recommended to install WTX Design Studio
  8.4.1.1 in an empty directory"* / *"then perform any installation customization."*
  (`techqa_DEV_Q243`, sentence d)
- Reviewed: one merged claim preserving both the referent and the sequencing — *"To resolve the
  COBOL Copybook importer being missing after upgrading IBM WTX Design Studio 8.4.1.1 over an
  existing installation, it is recommended to install WTX Design Studio 8.4.1.1 in an empty
  directory and then perform any installation customization."*

**4. Unresolved reference.** A complete, grammatical sentence whose pronoun only resolves against
a *different* sentence. Substitute the referent using the full response — that's not a new fact,
just re-attaching one already stated elsewhere in the same answer.
- Deterministic: *"This issue occurs because some files are not correctly overwritten or modified
  during the upgrade process."* (`techqa_DEV_Q243`, sentence c)
- Reviewed: *"The COBOL Copybook importer being missing after the upgrade occurs because some
  files are not correctly overwritten or modified during the upgrade process."* Resolve to the
  minimum antecedent that makes the claim checkable — not to everything upstream that happens to
  be related; re-importing an upstream sentence's own separate claim just bundles two facts into
  one.

**5. List-position context.** A list entry can be a complete, grammatical claim and still mean
nothing outside its list — its meaning comes from the header/label above it, not from anything
inside the sentence.
- Deterministic: *"- CVSS Base Score: 9.3"* (`techqa_DEV_Q074`, sentence f)
- Reviewed: *"CVSS Base Score: 9.3, for CVE-2015-1920."*

General rules across all five: never add a fact the sentence doesn't state; keep every negation,
quantity, and qualifier attached to its claim; and when genuinely torn, still make a call, then
record it in `unresolved_judgments` rather than leaving the decision blank.

**A downstream note on patterns 4 and 5.** In a 39-item TechQA pilot, every residual condition-D
failure categorized `multi_sentence_support` traced back to a pattern-4 or pattern-5 correction —
resolving a cross-sentence reference or a list-position dependency correctly produces a claim that
needs two evidence sentences, but the pipeline only ever supplies one. That's a property of the
single-evidence-sentence verifier design surfacing once decomposition is done carefully, not a sign
the correction was wrong. See [Understanding the decomposition-by-evidence
experiment](../../../../docs/explainers/decomposition-by-evidence.md#a-preliminary-finding-from-the-techqa-pilot).

## Interpretation constraint

This experiment cannot isolate "decomposition quality" by itself — see the **Interpretation
constraint** in
[`docs/reference/benchmarks.md`](../../../../docs/reference/benchmarks.md#decomposition-by-evidence-protocol)
before reading results out of a run. The report schema does not yet expose the additional counts
that constraint requires.

## Commands

Regenerate the blinded review packet directly from the pinned dataset population:

```bash
cd backend
poetry run python -m benchmark.decomposition_evidence_cli prepare-claims \
  --output evals/decomposition_evidence/v1/claim-review.json \
  --domains techqa finqa covidqa \
  --evaluation-split test \
  --evaluation-limit 100 \
  --seed 42
```

After the claim review is frozen, run all four conditions and create the residual-review draft:

```bash
poetry run python -m benchmark.decomposition_evidence_cli run \
  --claims evals/decomposition_evidence/v1/claim-review.json \
  --entailment-threshold 0.0017914474026707317 \
  --domains techqa finqa covidqa \
  --evaluation-split test \
  --evaluation-limit 100 \
  --seed 42 \
  --bootstrap-iterations 2000 \
  --output evals/decomposition_evidence/v1/results.json \
  --residual-review-output evals/decomposition_evidence/v1/residual-review.json
```

The residual packet is generated only after the frozen claims have produced the primary report.
It contains only condition-D false-unsupported sentences and, unlike the blinded claim packet,
includes annotated evidence and verifier outcomes. A reviewer assigns one of the six predeclared
categories, using `undetermined` whenever the record cannot distinguish mechanisms.
