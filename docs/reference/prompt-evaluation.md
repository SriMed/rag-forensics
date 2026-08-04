# Prompt development evaluation set

The prompt audit uses a versioned development evaluation set rather than editing prompts from
isolated examples. Frozen version 1.0.0 contains 24 cases: the original 15 audit samples, three
synthetic domain-shaped cases, three additional multi-chunk reasoning cases, and three exact
repository-derived cases from the local TechQA, FinQA, and CovidQA Chroma collections. Repository
cases record collection names, embedding IDs, extracted fields, and SHA-256 content hashes.

The fixtures are stored in
`backend/evals/prompt_audit/v1/cases.json`. Synthetic domain-shaped fixtures are not benchmark
evidence. Repository-derived fixtures exercise real stored project inputs, but three selected cases
still cannot establish domain-level performance.

`manifest.json` freezes the case count, split/mode counts, and SHA-256 hashes of the dataset and
human-review artifacts. Any hash change requires a new version rather than an in-place baseline
rewrite.

## Evidence boundary

Each case declares two kinds of expectations:

- **Deterministic scorers** check inspectable contracts such as non-empty output, exact normalized
  labels, valid JSON arrays of strings, item count, uniqueness, required or forbidden facts,
  sentence count, and word limits.
- **Human-review criteria** cover semantic properties that string checks cannot establish, such as
  claim atomicity, actual entailment, question answerability, correct uncertainty, and whether a
  proposed diagnostic test is genuinely discriminating.

The automatic pass rate includes only deterministic scorers. It is a contract-regression signal,
not a complete measure of prompt quality. Human criteria remain visible in the case definition and
score report so they cannot be silently replaced by a weak proxy.

Human judgments use the versioned `human-review.schema.json`. Copy
`human-review.template.json` outside the frozen v1 directory, then record the run identifier,
reviewer, review time, a `pass`, `fail`, or `uncertain` judgment, and a rationale. Criterion-level
notes are optional. This version deliberately defines no multi-reviewer adjudication workflow.

## Splits

The `development` split is selected by default and may be inspected while revising a prompt. The
`held_out` split is excluded by default and should be run only after choosing a candidate revision.
The split is a development discipline, not a secrecy mechanism: all fixtures remain committed and
inspectable.

When the set changes substantively, create a new version directory rather than overwriting v1.
Corrections that change an expected judgment also require a dataset-version change and an
explanation in the prompt audit.

## Execution modes

Every case is labeled:

- `production_path` reproduces the prompt boundary's current input shape.
- `counterfactual_capability` asks what the prompt/model could do under a different input shape.

The two combined-context entailment cases are counterfactual because production checks each chunk
in a separate model call. They do not test the current per-chunk loop. Production-path cases are
selected by default; counterfactual cases require `--execution-mode counterfactual_capability`.

## Execution boundary

Model execution is intentionally outside the committed evaluation scaffold. The repository stores
the cases, production prompt rendering, deterministic scoring, paired comparison, and review
contracts; it does not prescribe or claim a particular model-execution workflow.

`load_dataset()`, `select_cases()`, and `render_case()` in
`backend/evals/prompt_audit/evaluator.py` produce the exact prompt inputs. Saved response records can
then be scored with `score_records()`. `compare_record_sets()` rejects duplicate case IDs,
mismatched dataset versions, and non-identical case sets before producing paired improvement and
regression rows.

The paired report compares deterministic contracts only. Baseline and candidate semantic quality
must be reviewed using separate human-review records conforming to the frozen schema.

## Comparison protocol

1. Run and preserve the current production prompt on the production-path development cases.
2. Inspect deterministic failures and complete the declared human review.
3. Change one prompt hypothesis at a time.
4. Rerun the same development cases and compare paired case results.
5. Select a candidate without inspecting its held-out outputs during revision.
6. Run the held-out split once, report every regression, and retain the raw provenance.
7. Do not claim a production-model improvement from a proxy CLI model. Proxy runs can reveal prompt
   ambiguity and contract weaknesses; model-specific reliability requires the deployed model.

This set is deliberately small. It supports development and regression discovery, not population
accuracy estimates or calibrated claims about TechQA, FinQA, CovidQA, or production traffic.
