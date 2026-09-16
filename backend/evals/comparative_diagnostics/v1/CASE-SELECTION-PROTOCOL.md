# Case-selection protocol (frozen before any case is selected)

This protocol must be frozen — committed, unmodified from this point — before any case is chosen
based on system outputs. Selecting cases first and writing the protocol to match would let the
result shape the method; freezing first prevents that, per issue #30's explicit ordering.

## Candidate pool

Candidates are drawn only from populations already committed in this repository or directly
derivable from them without new human judgment. **Decision, made explicitly rather than left
implicit:** the primary pool is the *distinct parent examples* of issue #29's 188-eligible-sentence
population, not the sentences themselves and not a fresh sample. RAGChecker and RAGVue compare
whole answers, not isolated sentences, so the unit of comparison must be the parent example; using
a fresh independent sample instead would have severed the link to a population this project has
already reviewed once.

- **Primary pool**: `benchmark.comparative_diagnostics.load_case_candidate_pool()` — reuses
  `benchmark.experiment_cli._load_records` and `benchmark.oracle_evidence._eligible_records` with
  the exact parameters issue #29 used (`domains=["techqa","finqa","covidqa"]`, `split="test"`,
  `limit=100`, `seed=42`, `revision=DATASET_REVISION`), then takes `_eligible_records`'s
  already-deduplicated output — one entry per example with at least one eligible sentence, not one
  per sentence. This never touches the private `claim-review.json`/`results.json` review artifacts;
  eligibility is a deterministic function of RAGBench's own labels.
- **Secondary pool (not yet used)**: the frozen 39-item TechQA decomposition-by-evidence population
  (`backend/evals/decomposition_evidence/v1-techqa/`) and the committed RAGTruth adapter population
  (`benchmark/ragtruth.py`, `benchmark/ragtruth_cli.py`) remain available if the primary pool proves
  too thin for a given stratum.

No new RAGBench/RAGTruth examples are pulled in ad hoc; every candidate must already have a stable
example id, a domain, and a dataset revision recorded somewhere in the repository.

## Target size and strata

12–20 cases total, covering these declared strata (matching `ComparativeCase.stratum` in the
schema — see `backend/benchmark/comparative_diagnostics.py`):

| Stratum | Approx. target | Selection basis |
|---|---:|---|
| `systems_agree_labels_support` | 2–3 | All runnable systems agree; dataset label matches |
| `systems_agree_labels_contradict` | 2–3 | All runnable systems agree; dataset label disagrees |
| `component_diagnoses_disagree` | 2–3 | Systems name different suspected failing components |
| `evidence_attributions_disagree` | 1–2 | Systems attribute the answer to different evidence |
| `single_system_exposes_failure` | 1–2 | Only one system reports `unavailable`/`failed` for the case |
| `intervention_discriminates_hypotheses` | 1–2 | A proposed intervention (e.g. oracle evidence) resolves competing hypotheses |
| `intervention_fails_to_localize` | 1–2 | The same kind of intervention does not resolve them |
| `qualifier_negation_numerical_tabular_multisource_or_granularity` | 2–3 | Drawn from decomposition-by-evidence's five cataloged failure patterns and RAGTruth's span types |
| `counterexample_to_preferred_interpretation` | 1–2 | A case where RAG Forensics' own hypothesis ranking would mislead a reader |

Totals are approximate; the exact count and per-stratum split are recorded in the frozen manifest,
not fixed here, because pool availability (e.g., how many `single_system_exposes_failure` cases
actually occur) is only known after running the systems once.

## Selection procedure

1. Run all feasible systems (RAG Forensics, RAGChecker, RAGVue, RAGAS baseline) over the full
   candidate pool, recording every `NativeSystemOutput` — including `missing`/`unavailable`/
   `failed` states — before looking at which cases look interesting.
2. Compute, for the full pool, simple agreement/disagreement flags per stratum definition (e.g.,
   "do RAG Forensics' `suspected_component` and RAGChecker's failing-component signal name the
   same category"). This produces stratum-eligible candidate counts, not a ranking of "best"
   cases.
3. Within each stratum's eligible pool, select cases by a fixed, disclosed rule — a seeded random
   draw (documented seed) — not by hand-picking the most illustrative examples. Hand-picking is
   permitted only for the two strata that are inherently rare and definitional
   (`single_system_exposes_failure`, `counterexample_to_preferred_interpretation`); those
   selections must state the specific reason for the pick in `selection_rationale`.
4. Record candidate-pool counts, the seed, exclusions (e.g., a case dropped because a required
   system output could not be produced at all — not because it looked messy), and the reason each
   retained case was kept, in the frozen manifest itself (`ComparativeCaseSet`), not only in prose
   documentation.
5. Freeze the manifest (`status="frozen"`, `reviewer_identity` set,
   `population_sha256` recorded) before writing any aggregate comparative discussion. Do not
   re-open selection after seeing how the comparison reads, per this issue's out-of-scope item
   "tuning ... case selection after reviewing comparative aggregate results."

## What this protocol does not authorize

- Claiming the 12–20 selected cases represent prevalence of any failure type in RAGBench/RAGTruth
  as a whole, or in production use generally.
- Claiming one system is more accurate, faster, or more useful than another from this collection.
  That requires the separate calibrated comparison in issue #31.
- Inventing a gold root cause for a case where no dataset label or oracle-evidence result actually
  identifies one; such cases are recorded with `dataset_label: null` and an honest
  `reviewer_judgment`, not a fabricated cause.
