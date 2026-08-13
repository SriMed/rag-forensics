# LLM prompt and model-boundary audit

## Supported conclusion

RAG Forensics has several materially different LLM boundaries, and their risks should not be
collapsed into “prompt quality.” The most consequential finding is an installed input-contract
mismatch in the RAGAS value formerly exposed as retrieval relevance. The most consequential repository-owned
prompt risk is verdict rendering: free-form prose can turn mixed diagnostic signals into causal
claims stronger than their reliability supports.

On a small frozen proxy-model evaluation, an observation → competing hypotheses → discriminating
test structure avoided the observed verdict causal-overreach failure while remaining actionable.
That result supports a focused implementation issue; it does not establish general verdict
reliability or prove that deterministic scaffolding is superior. The current production prompts
remain unchanged by this audit.

Other observed risks were narrower but actionable: generated query-fit questions could be
semantically duplicative or unanswerable, structured-output prompts sometimes returned correct
payloads with prohibited commentary, and one repository-derived truncated passage repeatedly
elicited extrapolation beyond the stored evidence.

## Decisive findings

| Priority | Boundary | Finding | Support | Decision |
|---|---|---|---|---|
| 1 | RAGAS context precision | `reference="N/A"` is supplied where installed RAGAS 0.4.3 prompts for the answer/reference whose contexts are judged | Direct installed-code inspection | Validate and replace the mismatched configuration in [#20](https://github.com/SriMed/rag-forensics/issues/20) |
| 2 | Verdict rendering | Free-form rendering produced causal overreach from mixed, partly calibrated/model-judged signals | Frozen proxy samples plus human review | Implement inspectable verdict structure in [#21](https://github.com/SriMed/rag-forensics/issues/21) |
| 3 | Query-fit generation | Exact-string-valid questions could be semantically duplicate or not answerable from the chunks | Repeated proxy samples plus human review | Validate diversity, support, and failure semantics in [#22](https://github.com/SriMed/rag-forensics/issues/22) |
| 4 | Claim extraction | A correct empty JSON array was followed by commentary, making the production parse fail | Proxy sample plus code inspection | Enforce typed structured output in [#23](https://github.com/SriMed/rag-forensics/issues/23) |
| 5 | Entailment | Correct labels were repeatedly followed by explanations; production accepts recognizable substrings | Repeated proxy samples plus code inspection | Separate enum compliance from invalid format in [#24](https://github.com/SriMed/rag-forensics/issues/24) |
| 6 | Answer generation | A truncated repository passage repeatedly elicited completion or strengthening of the unfinished claim | One repository-derived case across repeated proxy runs | Investigate complete/truncated pairs in [#25](https://github.com/SriMed/rag-forensics/issues/25) |
| 7 | Evaluation scorer | Decimal points were counted as sentence boundaries, creating a false held-out contract failure | Direct output/scorer inspection | Version the corrected scorer in [#26](https://github.com/SriMed/rag-forensics/issues/26) |

## Evidence and method

### Frozen evaluation set

Version 1.0.0 contains 24 cases with a hash-pinned manifest:

- 18 production-path development cases;
- one counterfactual-capability development case;
- four production-path held-out cases;
- one counterfactual-capability held-out case.

The original 15 prompt samples were supplemented with synthetic domain-shaped and multi-chunk
cases, plus three exact repository-derived TechQA, FinQA, and CovidQA inputs with Chroma embedding
IDs and content hashes. “Domain-shaped” is not domain evidence; three repository examples also do
not support domain-level accuracy claims.

Each case declares deterministic contract checks and separate human semantic criteria. The
development baseline, two verdict candidates, and the selected candidate's production-path
held-out run were executed through an authenticated Claude CLI. These are proxy-model observations,
not results from the exact production Anthropic SDK/model configuration. Raw runs and reviews were
preserved locally outside version control; the committed repository contains the frozen cases,
scoring logic, review schema, and comparison method without claiming model execution.

Issue #21 subsequently implemented deterministic verdict reasoning and evaluated its two frozen
development verdict cases plus the single frozen held-out verdict case through the Anthropic SDK
with the configured `claude-sonnet-4-6` model. The held-out case was executed once after development
was complete and was not used for tuning. Its raw response and review are preserved in
[`verdict-reasoning-production-review.json`](../../backend/evals/prompt_audit/v1/verdict-reasoning-production-review.json).

### Aggregate results

| Run | Scope | Deterministic | Human review | Interpretation |
|---|---|---:|---:|---|
| Production prompt baseline | 18 development production-path cases | 13/18 | 15/18 | Current comparison point |
| Verdict candidate v1 | Same 18 case IDs | 12/18 | 14/18 | Rejected: target remained a failure and an existing verdict pass regressed |
| Verdict candidate v2 | Same 18 case IDs | 13/18 | 15/18 | Selected on development: target improved and the other verdict remained a semantic pass |
| Verdict v2 held-out | Four production-path held-out cases, run once | 3/4 | 4/4 | Semantic pass; deterministic miss came from decimal-sensitive sentence counting |
| Structured verdict implementation | Two frozen development verdict cases, production SDK/model | Not rescored as a paired baseline | 2/2 | Both retained distinct hypotheses and outcome-dependent tests |
| Structured verdict held-out | One frozen held-out verdict case, production SDK/model, run once | 1/1 | 1/1 | Missing evidence remained unavailable; no tuning or rerun followed |

No counterfactual held-out case was run. Neither development nor held-out was rerun to rescue a
candidate after inspecting its output.

### Support labels used in this report

- **Direct code inspection** establishes call shape, parser behavior, and downstream data flow.
- **Deterministic evaluation** establishes only declared syntax/cardinality/string contracts.
- **Human review** is a single recorded review, not adjudicated multi-rater ground truth.
- **Proxy-model evidence** demonstrates behavior of the authenticated CLI runs, not a production
  failure rate.
- **Inference** identifies a plausible mechanism or follow-up; it is not presented as established.

## 1. RAGAS judge boundary

### Purpose and call site

`backend/services/ragas_scorer.py:28–60` invokes dependency-owned faithfulness and context-precision
metrics using `ChatAnthropic(model="claude-haiku-4-5-20251001")`. The complete installed-version
prompt and aggregation audit is in [Installed RAGAS prompt contract audit](ragas-prompt-audit.md).

### Installed output contracts

RAGAS 0.4.3 faithfulness first generates answer statements, then returns structured binary
faithfulness verdicts for those statements against newline-joined contexts. The final score is the
fraction of truthy verdicts; no generated statements produces `NaN`.

The imported context-precision implementation requires `user_input`, `retrieved_contexts`, and
`reference`. For each context, its prompt asks whether the context was useful in arriving at the
supplied answer, parses a reason and binary verdict, and calculates rank-sensitive average
precision.

### Finding and downstream consequence

`score_retrieval_relevance()` supplies `reference="N/A"` at
`backend/services/ragas_scorer.py:43–49`. Installed RAGAS therefore asks whether each context was
useful for arriving at the literal answer `N/A`. This is a direct contract mismatch, although its
quantitative effect has not yet been isolated.

The resulting value is exposed as `retrieval_relevance_score`, triggers query-fit below `0.5`,
contributes a ranked low-relevance concern, enters the verdict prompt, and appears in the API. An
issue #9 smoke test recorded zero relevance across three domains; the mismatch is a plausible
explanation, not a proven cause.

### Recommendation

Do not merely replace the sentinel string. Compare input-compatible reference-aware or
reference-free configurations on human-labeled relevant and irrelevant contexts, make non-finite
or failed judgments explicitly unavailable, and pin the installed contract. Follow-up:
[#20](https://github.com/SriMed/rag-forensics/issues/20).

### Issue #20 resolution

The production path now uses installed RAGAS 0.4.3 `ContextUtilization`, supplies the actual
generated or caller-provided answer as `response`, and exposes the narrower answer-conditioned
construct as `ragas.context_utilization`. Exceptions and non-finite results are explicit unavailable
states. The earlier paragraphs remain the audit evidence that motivated this change; they describe
the superseded implementation rather than current behavior. See
[Installed RAGAS prompt contract audit](ragas-prompt-audit.md) for the current contract.

## 2. Verdict rendering

### Purpose and call site

`build_verdict_reasoning()` receives the top deterministic signals and constructs typed
observations, reliability, competing hypotheses, a named component, one test, and
outcome-dependent interpretations. `RANKED_SIGNALS_PROMPT` receives only that structure.
`render_recommendation()` calls the configured Claude Sonnet model with `max_tokens=200`; API
failure deterministically renders the complete structure.

### Output contract

The prompt limits the model to wording supplied observations, hypotheses, component, test, and
outcomes in two to three sentences. Reliability must bound its language, and it may not add causes,
tests, or facts. The inspectable API structure remains authoritative; recommendation prose is a
presentation layer.

### Representative observations

| Case | Observation | Judgment |
|---|---|---|
| `verdict_dominant` baseline | Actionable A/B test, but six sentences and 134 words | Semantic pass; format fail |
| `verdict_conflicting` baseline | Embedding isolation was said to explain overconfident generation and identify the embedding model as bottleneck | Semantic fail: causal overreach |
| Candidate v1 | Added uncertainty/test wording but retained the target causal narrative and regressed `verdict_dominant` | Rejected negative result |
| Candidate v2 development | Required separate observations, materially different hypotheses, and outcome-dependent test | Both verdict cases semantic pass |
| `verdict_unavailable` held-out | Kept hedging analysis unavailable and proposed retrieval-versus-scorer alternatives with an ablation test | Semantic pass |

Candidate v2 retained format imperfections: it could repeat numeric values and its punctuation
conflicted with an “exactly two sentences” instruction. The result supports the structural lesson,
not copying the evaluated text.

The implemented structure passed semantic review on both frozen development verdict cases through
the production SDK/model. In the one-time `verdict_unavailable` held-out run, all frozen
deterministic checks passed: the response was non-empty, contained three sentences, and explicitly
included unavailable/rerun semantics. A single-reviewer semantic review also passed both criteria:
missing hedging evidence was not presented as a healthy zero, and the diagnosis acknowledged the
missing evidence. This is one case-level contract result, not a production failure-rate estimate.

### Recommendation

Issue [#21](https://github.com/SriMed/rag-forensics/issues/21) implemented the recommended
inspectable intermediate representation, bounded rendering, and deterministic fallback. The exact
production SDK/model checks support closing that implementation issue while preserving the narrow
evidence boundary above.

## 3. Query-fit question generation

### Purpose and call site

`build_question_generation_prompt()` in `backend/prompts/query_fit_prompts.py:4–20` requests three
to five specific questions answerable from retrieved chunks. The conditional call and parsing live
at `backend/services/forensics/query_corpus_fit.py:44–104`.

### Output contract and current handling

Issue #22 replaced the string-list contract with structured candidates containing inspectable
chunk IDs. Production rejects unknown citations, uses an independent structured judgment for
direct answerability and specificity, and rejects semantic duplicates at cosine similarity `>= 0.90`. Fewer than
three accepted questions makes classification explicitly unavailable; it does not produce a fit
label from a partial set. Mean question/query similarity is calculated only over a valid set.

### Representative observations

- `query_fit_coherent` produced answerable questions in the baseline, but a later unchanged-prompt
  run added a question about extending a session without login, which the chunks did not state.
- `query_fit_mixed` produced two distinct strings with the same vaccine-storage intent in one run.
- Another run asked for 2024 revenue figures when the chunk supplied only an eight-percent change.
- `query_fit_injected_chunk` ignored an embedded instruction to ask about France, demonstrating
  useful instruction/data separation in that held-out case.

These repeated observations show quality variance and gaps in validation; they do not establish a
population failure rate.

### Recommendation

The implemented contract preserves accepted and rejected candidates for inspection while limiting
the resulting label to retrieved-context fit. It does not infer whether the full corpus covers the
question. Follow-up: [#22](https://github.com/SriMed/rag-forensics/issues/22).

## 4. Claim extraction

### Purpose and call site

`CLAIM_EXTRACTION_PROMPT` in `backend/prompts/hedging_prompts.py:1–8` asks for factual claims as a
JSON array while preserving hedging. It is called at
`backend/services/forensics/hedging_mismatch.py:148–179` before deterministic confidence
classification.

### Output contract and current handling

Production supplies the Anthropic API with a JSON Schema whose root is an array and whose items are
strings. It then decodes the returned text and independently validates the same type contract before
confidence classification. Fences, trailing commentary, and other invalid JSON produce
`claim_extraction_parse_failed`; valid JSON with a non-array root or non-string item produces
`claim_extraction_schema_failed`; request or response-access failures produce
`claim_extraction_failed`. Only a validated empty array reports a healthy zero-claim result.

### Representative observations

- `claim_compound` returned three atomic strings and preserved “may” and “about.”
- `claim_no_facts` returned the correct empty array but appended an explanation. This was
  semantically correct yet invalid JSON for production after fence stripping.
- `claim_embedded_instruction` kept the quoted instruction as data and extracted the 2024
  publication fact without obeying the quotation.

### Deterministic offload assessment

Sentence splitting could provide candidate spans or a fallback, but it is not equivalent to claim
extraction: compound sentences can contain independently supportable propositions, while some
claims span clauses or sentences. Issue #18 did not show that the tested decomposition/entailment
pipeline improved grounding classification. A hybrid typed boundary is better supported than full
replacement by sentence splitting.

### Recommendation

The typed boundary and failure distinction were implemented by
[#23](https://github.com/SriMed/rag-forensics/issues/23). Production-model reliability is still an
empirical question, but invalid values cannot reach confidence classification or entailment.

## 5. Entailment

### Purpose and call site

`ENTAILMENT_PROMPT` in `backend/prompts/hedging_prompts.py:10–15` asks whether one retrieved chunk
directly supports one claim. Production checks up to the top three chunks separately at
`backend/services/forensics/hedging_mismatch.py:181–227` and stops at the first supported judgment.

### Output contract and current handling

Production trims surrounding whitespace and accepts only the exact lowercase typed enum values
`supported` and `not_supported`. Invalid formats and per-chunk exceptions remain distinct from a
valid negative judgment, with claim- and chunk-level coverage exposed in the API.

### Representative observations

- Paraphrase support and numeric contradiction received the expected labels.
- Partial conjunction support and population overgeneralization received the correct negative
  labels followed by explanations, violating the label-only instruction while remaining
  recognizable to production.
- The combined-evidence multi-chunk cases are explicitly counterfactual: production never sends
  both chunks in one entailment call.

### Deterministic offload assessment

Chunk/claim cosine similarity is useful for candidate selection, not a demonstrated entailment
replacement. Issue #18 found B3 did not improve on B1, and issue #19 showed oracle evidence reduced
false unsupported judgments without eliminating them. Verifier behavior, decomposition,
multi-sentence reasoning, and annotation granularity remain competing explanations.

### Recommendation

The exact typed boundary, unavailable state, and evaluated coverage were implemented by
[#24](https://github.com/SriMed/rag-forensics/issues/24). This deliberately supersedes the
permissive normalization from #15.

## 6. Answer generation

### Purpose and call site

`GENERATION_SYSTEM_PROMPT` and `build_generation_prompt()` at
`backend/prompts/generation_prompts.py:3–18` tell Claude Haiku to answer only from supplied chunks.
`generate_answer()` calls the model with `max_tokens=1024` at
`backend/services/generator.py:10–22`. Output is intentionally free-form prose, and API exceptions
propagate.

### Representative observations

- A directly supported scattering question was answered accurately.
- An insufficient-history question and an exact irrelevant TechQA chunk both elicited appropriate
  abstention.
- A retrieved instruction requesting a false temperature was ignored in favor of the stated 18 C.
- A multi-chunk finance case correctly calculated operating income and margin.
- An exact repository CovidQA passage ended mid-claim. Across repeated runs, the model strengthened
  the fragment into a risk-factor or protective-role conclusion beyond the stored text.

### Recommendation

No broad prompt rewrite is supported by this small set. The completed
[#25](https://github.com/SriMed/rag-forensics/issues/25) [paired evaluation](truncated-evidence.md)
found case-dependent extrapolation, abstention, and fragment copying. A deterministic hybrid made
truncation visible in all six truncated proxy outputs but did not reliably prevent completion of
the CovidQA fragment. [#27](https://github.com/SriMed/rag-forensics/issues/27) subsequently
implemented source-aware completeness metadata and a bounded generation contract. Its exact-model
comparison improved disclosure without providing lexical enforcement; see the linked evaluation.

## 7. Inactive and empty prompt files

`backend/prompts/calibration_prompts.py` is empty and has no call site. It is not an active prompt
boundary and no replacement prompt should be invented. `backend/prompts/__init__.py` is also empty.
`DIMENSION_EXPLANATION_PROMPT` is defined but unused. These states should remain visible in the
inventory; cleanup of the verdict-adjacent unused constant can be decided under #21 without a
separate issue.

## Evaluation-scaffold finding

The frozen v1 sentence scorer used punctuation matching that split decimal values into sentence
boundaries. In the one-time held-out verdict, decimal values `0.74` and `0.77` inflated a visibly
three-sentence response to five and created a false deterministic failure. Frozen v1 scores and
hashes were not rewritten. [Issue #26](https://github.com/SriMed/rag-forensics/issues/26) introduced
`prompt-eval-scorer.v2`; the same response shape counts as three under the decimal-safe semantics.
This is recorded as a migration difference, not a corrected historical result. See
[Prompt development evaluation](prompt-evaluation.md#scorer-versions-and-sentence-semantics).

## Follow-up issues

1. [#20 — Fix RAGAS retrieval-relevance input-contract mismatch](https://github.com/SriMed/rag-forensics/issues/20)
2. [#21 — Implement inspectable structured verdict reasoning](https://github.com/SriMed/rag-forensics/issues/21)
3. [#22 — Validate query-fit question diversity and answerability](https://github.com/SriMed/rag-forensics/issues/22)
4. [#23 — Enforce claim-extraction structured output contract](https://github.com/SriMed/rag-forensics/issues/23)
5. [#24 — Enforce exact entailment output contract and failure semantics](https://github.com/SriMed/rag-forensics/issues/24)
6. [#25 — Investigate generation extrapolation from truncated evidence](https://github.com/SriMed/rag-forensics/issues/25)
7. [#26 — Fix decimal-sensitive sentence counting in prompt evaluator](https://github.com/SriMed/rag-forensics/issues/26)

## Open questions and update conditions

- Broader production-model reliability remains unknown: the exact production SDK/model validation
  covers two frozen development cases and one frozen held-out verdict case, not a representative
  production sample.
- The magnitude of the RAGAS sentinel mismatch remains unknown until compared against labeled
  retrieval examples and a contract-compatible metric.
- Query-fit and truncated-generation observations are case-level findings; broader claims require
  more repository or production examples.
- A structurally scaffolded verdict should be reconsidered if production evaluation loses
  actionability, collapses competing explanations, or causes prior semantic passes to fail.
- Prompt-only verdict rendering should be reconsidered entirely if inspectable structure cannot
  bound causal claims under production evaluation.

## Final contribution

The audit's contribution is methodological and diagnostic rather than a production prompt change:
it makes the project's active and dependency-owned LLM boundaries inspectable, establishes a
frozen development/held-out protocol, preserves a useful failed candidate, and turns observed risks
into falsifiable follow-up work. The strongest current operational lesson is to keep deterministic
signal ranking, move hypothesis/test structure into inspectable logic, and treat the LLM as a
bounded renderer rather than an unobserved root-cause reasoner.
