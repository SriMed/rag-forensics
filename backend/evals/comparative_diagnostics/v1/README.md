# Comparative diagnostic disagreement set — feasibility and protocol (issue #30)

This directory is the feasibility check and frozen shared schema for issue #30, not the case
collection itself. Selecting cases is out of scope for this pass; it requires the schema below to
be frozen first, per the issue's own ordering.

## 1. Feasibility check: can RAGChecker and RAGVue run reproducibly on our inputs?

Both are real, installable, actively maintained packages (not the fictional-sounding names in
`docs/reference/related-work.md` might suggest at a glance — they exist on PyPI and GitHub and were
installed and exercised for this check). Findings below come from installing both in an isolated
virtual environment (never added to `backend/pyproject.toml`) and running small, live smoke tests
against toy question/answer/context inputs, not from reading documentation alone.

### RAGChecker (PyPI: `ragchecker` 0.1.9, depends on `refchecker` 0.2.17)

- **Input contract**: `RAGResult` requires `query_id`, `query`, `gt_answer`, `response`, and
  `retrieved_context` (a list of `{doc_id, text}`). **`gt_answer` — a reference answer — is
  required.** RAG Forensics' own `/analyze/custom` contract has no such field; it diagnoses an
  answer without assuming a gold reference exists. RAGBench supplies a reference answer per
  example, so this is satisfiable for RAGBench-sourced cases, but it is a real construct
  difference worth stating explicitly rather than treating RAGChecker's need for ground truth as
  incidental.
- **Claim extraction and checking is LLM-backed**, driven through `litellm`. It is not
  Bedrock/OpenAI-only, as the top-level quick-start examples suggest: a live smoke test using
  `model="anthropic/claude-haiku-4-5-20251001"` in `refchecker.LLMExtractor` succeeded and
  produced normal triplet claims, so RAGChecker can be pointed entirely at Claude, consistent with
  this project's Anthropic-only convention.
- **Real, load-bearing incompatibility: `refchecker` pins `anthropic<0.30,>=0.29`.** This
  project's own `pyproject.toml` pins `anthropic>=0.86,<0.87` (used directly by `generator.py`,
  `hedging_mismatch.py`, and `query_corpus_fit.py`). Installing RAGChecker into the same
  environment as the backend downgrades or conflicts with that pin — `pip` reported it explicitly
  (`refchecker 0.2.17 requires anthropic<0.30,>=0.29, but you have anthropic 0.86.0`). Forcing the
  newer SDK anyway did not break RAGChecker's own extractor call in this smoke test, but the
  declared constraint means `poetry add ragchecker` to `backend/pyproject.toml` is not viable
  without either violating CLAUDE.md's "do not upgrade/downgrade existing dependencies" rule or
  running two dependency-incompatible tools in one lockfile.
  - **Resolution**: run RAGChecker from its own isolated environment/subprocess, never as a
    `backend/pyproject.toml` dependency. See "Chosen integration shape" below.
- `en_core_web_sm` (spacy) is documented as a setup step but was not required to reach a
  successful claim-extraction call in this smoke test; not fully verified for the checker
  (entailment) stage or for `--claim_format triplet` end-to-end, and is called out here as an open
  item rather than a confirmed non-issue.
- Native output: `RAGChecker.evaluate(results, metrics="all_metrics")` reports retriever metrics
  (`claim_recall`, `context_precision`, `context_utilization`), generator metrics
  (`noise_sensitivity_in_relevant`, `noise_sensitivity_in_irrelevant`, `hallucination`,
  `self_knowledge`, `faithfulness`), and `overall_metrics` (`precision`, `recall`, `f1`) — all
  continuous scores, no explicit missing/unavailable/failed state in the metric objects
  themselves. Run-level failures (e.g. an extractor timeout) must be captured from the process
  boundary, not the metric payload.

### RAGVue (PyPI: `ragvue` 0.6.1)

- **Input contract**: `{question, answer, contexts: list[str]}` — reference-free, no ground-truth
  answer required. This is a much closer structural match to RAG Forensics' own
  `/analyze/custom` contract than RAGChecker's.
- **Four local, no-API-key metrics ran successfully offline** in this smoke test
  (`answer_length`, `token_overlap`, `readability`, `context_similarity` — TF-IDF/sklearn-based),
  confirming those are reproducible without any live model call or secret.
- **LLM-judged metrics (e.g. `retrieval_relevance`) work with Claude** by setting
  `RAGVUE_JUDGE_PROVIDER=anthropic`, and RAGVue's shipped default judge model naming
  (`claude-haiku-4-5-20251001`) already matches this project's model id — but only once the
  Anthropic SDK is upgraded past the version RAGChecker's dependency chain installs. With
  `anthropic==0.29.2` (the version `refchecker` demands) present in the same environment, the
  identical call failed with `Client.__init__() got an unexpected keyword argument 'proxies'` — an
  `anthropic`/`httpx` version-compatibility break, not a RAGVue defect. This is the concrete
  symptom of the incompatibility above: **RAGChecker and RAGVue cannot share one Python
  environment with a modern `anthropic` SDK.**
- **Provenance-transparency finding, directly relevant to this issue's schema requirement**: the
  same successful, Anthropic-backed `retrieval_relevance` call returned
  `raw.model: "gpt-4o-mini"` in its own metadata — RAGVue's default-model constant, not the model
  actually invoked — even with `RAGVUE_JUDGE_PROVIDER=anthropic` set, no `OPENAI_API_KEY` present
  anywhere in the environment, and a real Claude-shaped judgment returned. **RAGVue's own output
  cannot be trusted to self-report which model produced a given judgment.** Any comparison must
  record the model actually configured for a run out-of-band (from our own run configuration) and
  must not take a system's self-reported provenance field at face value. This is exactly the kind
  of thing the shared schema's `NativeSystemOutput` is for: it preserves the raw output including
  this mislabeled field, and the mapped `method` field is populated from what we configured, not
  from what RAGVue's own metadata claims.

### RAGAS baseline

Already integrated in this repository (`services/ragas_scorer.py`); no separate feasibility check
is needed. It is included as `ragas_baseline` in the schema below for comparability, using this
project's existing wrapper rather than a new integration.

### Chosen integration shape

Given the anthropic-SDK conflict, RAGChecker and RAGVue must run as **separate, isolated
processes** (their own virtual environments), invoked as subprocesses or via recorded JSON
input/output files — never added to `backend/pyproject.toml`. This keeps the backend's own
dependency graph untouched (per CLAUDE.md) and matches how a real user of RAGChecker or RAGVue
would run them: as independent tools, not library imports sharing our process. The comparative
runner (not yet built) will shell out to each tool's own CLI or a small per-tool driver script
running in its own venv, and read back JSON to populate `NativeSystemOutput.raw_output` verbatim.

## 2. Frozen provenance-preserving comparison schema

The shared schema lives in code, not only in this document, so it can be validated:
[`backend/benchmark/comparative_diagnostics.py`](../../../benchmark/comparative_diagnostics.py),
tested in
[`backend/tests/test_comparative_diagnostics_schema.py`](../../../tests/test_comparative_diagnostics_schema.py).

Key design choices:

- `NativeSystemOutput.raw_output` is always preserved as returned by the system — nothing is
  reshaped into it — and pairs with an `availability` field restricted to exactly four states
  (`healthy`, `missing`, `unavailable`, `failed`), matching the issue's requirement to distinguish
  those explicitly rather than letting an evaluator failure read as a healthy zero (see
  `models.py`'s existing `status: Literal["ok", "unavailable"]` / `"error"` conventions this
  mirrors).
- `SystemDiagnosticRecord` maps each system's output into shared fields (suspected component,
  supporting observation, evidence attribution, method, reliability, causal-strength language,
  proposed intervention) but every mapped field is optional, and `no_equivalent_fields` names a
  system's own constructs that have no cross-system analogue instead of forcing them into a
  borrowed field (e.g., RAGVue's calibration/stability metrics have no RAG Forensics equivalent
  today).
- `CaseJudgments` keeps `dataset_label`, `model_judgments` (per system), `reviewer_judgment`, and
  `intervention_evidence` as four distinct, independently settable fields — the issue's explicit
  requirement — rather than one collapsed "verdict" field.
- `ComparativeCase.stratum` is restricted to the nine strata named in the issue (agreement,
  disagreement, single-system failure exposure, discriminating/non-discriminating intervention,
  qualifier/negation/numerical/tabular/multi-source/granularity cases, and counterexamples), so a
  case cannot be added without declaring which stratum motivated its inclusion.
- `ComparativeCaseSet.status` follows the same draft/frozen pattern as
  `backend/benchmark/decomposition_evidence.py`'s review artifacts: a frozen set requires a
  `reviewer_identity` and at least one case, and `make_population_sha256` fingerprints the
  ordered case-id sequence so a later silent reordering or addition is detectable.

## 3. What is not done yet

No cases are selected in this pass. `ComparativeCaseSet` here is empty and `status="draft"` by
construction; the frozen case manifest, the actual RAGChecker/RAGVue subprocess drivers, and the
12–20 selected cases across the declared strata are follow-up work once this schema has been
reviewed. See [`CASE-SELECTION-PROTOCOL.md`](CASE-SELECTION-PROTOCOL.md) for the bounded selection
protocol that will run once the schema above is accepted.
