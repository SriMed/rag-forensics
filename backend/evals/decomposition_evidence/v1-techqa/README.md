# Decomposition-by-evidence review artifacts, v1 — TechQA-only

This directory is a deliberately narrower variant of
[`v1/`](../v1/README.md): the same protocol, frozen against a **TechQA-only** population instead
of the full 188-item, three-domain one.

## Why a separate population

The full `v1` population is 73% `covidqa` by construction — `covidqa` examples are eligible for
this diagnostic at a much higher rate than `techqa` or `finqa`, and raising the shared sampling
limit only adds more `covidqa` items; `techqa` and `finqa` stay thin regardless (see
[Understanding the decomposition-by-evidence experiment](../../../../docs/explainers/decomposition-by-evidence.md)).
TechQA's list-structured, multi-step procedural answers also produce the highest concentration of
the five failure patterns catalogued in `v1/README.md` — reviewing a smaller, domain-mixed sample
instead would have meant mostly reviewing `covidqa` and losing most of that signal.

This population is the exact TechQA subset of `v1`'s 188 items — same records, same deterministic
splits, same `population_sha256`-verified provenance, just domain-scoped at generation time
(`--domains techqa`) rather than truncated after the fact.

## Claim review

`claim-review.json` is **frozen** — 39 items, all reviewed, `reviewer_identity` set. It follows the
same reviewer instructions and the same five failure patterns as `v1/README.md`; see that file for
the pattern catalog. Nine items are flagged in `unresolved_judgments`, mostly barely-a-claim list
entries (e.g. a bare CVSS score line, or the string `"- None"` under an unpopulated "workarounds"
heading) where a reviewer's best executable decision is still recorded but the reviewer is not
confident it carves the material at its natural joint.

## Scope of any result from this population

A result run against this population supports a claim about **TechQA only**. It does not
generalize to `finqa` or `covidqa` without separately running and reporting on those domains — see
the companion tiny three-domain sample (planned, not yet run) for that.

## Commands

Run all four conditions and create the residual-review draft:

```bash
cd backend
poetry run python -m benchmark.decomposition_evidence_cli run \
  --claims evals/decomposition_evidence/v1-techqa/claim-review.json \
  --entailment-threshold 0.0017914474026707317 \
  --domains techqa \
  --evaluation-split test \
  --evaluation-limit 100 \
  --seed 42 \
  --bootstrap-iterations 2000 \
  --output evals/decomposition_evidence/v1-techqa/results.json \
  --residual-review-output evals/decomposition_evidence/v1-techqa/residual-review.json
```

`--domains techqa` and `--evaluation-limit 100` must match how this population was generated, or
the runner rejects the population mismatch.
