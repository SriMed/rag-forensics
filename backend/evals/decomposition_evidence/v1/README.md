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
