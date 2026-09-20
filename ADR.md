# Architectural Decision Records

Decisions made in the RAG Forensics project, in the order they were made.

---

## ADR-001: FastAPI as the backend framework

**Status:** Accepted
**Issue:** #2

Chose FastAPI over Flask or Django. FastAPI provides native async support, automatic OpenAPI docs, and Pydantic v2 integration for request/response validation — all with minimal boilerplate. The project is I/O-bound (LLM calls, vector DB queries), so async support matters.

---

## ADR-002: Pydantic models live in `models.py`, not inside `services/`

**Status:** Accepted
**Issue:** #2

All shared data models (`StoredExample`, `RetrievedChunk`, `AnalyzeResponse`, etc.) live in a single top-level `backend/models.py` rather than scattered inside service modules. This prevents circular imports and makes the data contract easy to find and extend.

---

## ADR-003: Embedded ChromaDB with persistent local storage

**Status:** Accepted
**Issue:** #3

Chose ChromaDB in embedded mode (`PersistentClient`) over a hosted vector database (Pinecone, Weaviate, etc.). This eliminates infrastructure dependencies for local development and evaluation. The index is stored at `backend/data/chroma/` and excluded from git. The tradeoff is that each developer runs `bootstrap_data.py` once to seed the index.

---

## ADR-004: Three separate ChromaDB collections, one per RAGBench domain

**Status:** Accepted
**Issue:** #3

The RAGBench benchmark spans three domains: `techqa`, `finqa`, `covidqa`. Each domain gets its own ChromaDB collection rather than a single unified collection with a domain metadata filter. This simplifies per-domain queries and avoids cross-domain contamination in retrieval.

`retrieve_for_example()` searches all three collections in sequence and returns the first match.

---

## ADR-005: L2 distance converted to similarity via `1 - distance`

**Status:** Accepted; interpretation narrowed by ADR-042
**Issue:** #3

ChromaDB returns L2 distances (lower = closer). For the rest of the pipeline to reason about scores uniformly, distances are converted to similarity scores clamped to `[0.0, 1.0]` via `score = max(0.0, 1.0 - distance)`. All downstream code (forensics, scoring, API responses) works with similarity scores, not raw distances.

---

## ADR-006: Claude (Anthropic SDK) for answer generation, not an OpenAI-compatible wrapper

**Status:** Accepted; grounding contract extended by ADR-041
**Issue:** #4

Used the Anthropic Python SDK directly rather than routing through LangChain or an OpenAI-compatible shim. This keeps the dependency surface smaller and avoids abstraction layers that can obscure errors. Model: `claude-haiku-4-5-20251001` (fast and cheap for generation).

The generator's system prompt enforces grounding: "use ONLY information from the provided context chunks."

---

## ADR-007: RAGAS for faithfulness and retrieval relevance scoring

**Status:** Superseded by ADR-033
**Issue:** #4

Rather than writing custom faithfulness/relevance scoring from scratch, delegated to the RAGAS library (`faithfulness` + `context_precision` metrics). RAGAS is purpose-built for RAG evaluation and uses an LLM judge internally. Bound RAGAS to Claude via the `ChatAnthropic` LangChain adapter.

---

## ADR-008: Tri-state verdicts (pass / warn / fail) rather than raw scores

**Status:** Superseded by ADR-038
**Issue:** #4

RAGAS returns continuous scores in `[0.0, 1.0]`. These are bucketed into `pass` (≥ 0.75), `warn` (≥ 0.5), and `fail` (< 0.5) before being returned to the caller. Continuous scores are ambiguous for users — tri-state verdicts force a clear interpretation. The thresholds are explicit constants, easy to tune.

---

## ADR-009: Each forensics module is independent and callable in isolation

**Status:** Superseded by ADR-038
**Issues:** #5–#8

The four forensics modules (`retrieval_distribution`, `hedging_mismatch`, `chunk_attribution`, `confidence_calibration`) share no internal state and have no dependencies on each other. Each accepts only what it needs (chunks, answer, question) and returns a `DimensionResult`. This makes them independently testable and allows partial implementation without breaking the endpoint.

Unimplemented modules return a `_STUB_DIMENSION` placeholder so the API remains callable throughout development.

---

## ADR-010: System prompts live in `prompts/`, separate from service logic

**Status:** Accepted
**Issues:** #4, #5–#8

LLM prompts (system instructions and context builders) live in `backend/prompts/` rather than inlined in service files (e.g., `prompts/generation_prompts.py`). This separates prompt engineering from orchestration logic and makes prompts easy to iterate on without touching service code.

---

## ADR-011: Retrieval distribution analyzed as a probabilistic signal

**Status:** Superseded by ADR-038
**Issue:** #5

Rather than treating retrieval as a binary pass/fail (did we get good chunks?), the `retrieval_distribution` module analyzes the *shape* of the score distribution using five metrics:

- **score_gap**: cliff between top and second score — a large gap suggests the retriever is confident but narrow
- **score_entropy**: Shannon entropy of the normalized distribution — high entropy = flat / uncertain retrieval
- **decay_rate**: exponential decay parameter fit to score-vs-rank — high decay = sharp drop-off
- **tail_mass**: score mass beyond the top two chunks — high tail = diffuse retrieval
- **top_score**: raw value of the best match

Uses `numpy` for entropy and `scipy.optimize.curve_fit` for the exponential fit. Gracefully degrades to zero-valued metrics for edge cases (< 3 chunks, failed curve fit).

---

## ADR-012: TDD as a non-negotiable workflow contract

**Status:** Accepted
**Applies to:** all issues

Tests are written before implementation. Each test file must fail before any implementation exists, then pass after. No commits unless tests are green. No live API calls in tests — all external services (Anthropic, RAGAS, ChromaDB) are mocked at defined seams. This is enforced in CLAUDE.md and applies to both human and AI contributors.

The 3-attempt rule: if a test cannot be made to pass after 3 focused attempts, stop and report rather than continuing to loop.

---

## ADR-013: Poetry for all dependency management

**Status:** Accepted
**Applies to:** all issues

All Python dependencies are managed via Poetry. `pyproject.toml` is the single source of truth. `pip install` is not used; neither is manual edits to `pyproject.toml`. Dev-only dependencies use `poetry add --group dev`. This ensures reproducible environments across machines.

---

## ADR-014: CORS enabled only for localhost:3000

**Status:** Accepted; deployment assumption superseded by ADR-043
**Issue:** #2

The FastAPI backend allows cross-origin requests only from `http://localhost:3000`, which is the Next.js dev server. No wildcard origins. ADR-043 retains localhost operation as the intended delivery boundary rather than anticipating a hosted frontend origin.

---

## ADR-015: Noisy third-party loggers suppressed to WARNING

**Status:** Accepted
**Issue:** #4

Libraries like `httpx`, `httpcore`, `langchain`, `chromadb`, `ragas`, and `sentence_transformers` are verbose at DEBUG level. Their loggers are explicitly set to WARNING in `main.py` so that application-level debug logs (retrieval scores, chunk counts, model selection) remain readable without noise.

---

## ADR-016: `bootstrap_data.py` is never run automatically

**Status:** Accepted
**Issue:** #3

Seeding the ChromaDB index from RAGBench is slow and expensive (network + embedding compute). The script is never triggered automatically by tests, server startup, or CI. It runs only when explicitly instructed. The `backend/data/` directory is gitignored so the index is never committed.

---

## ADR-017: `retrieve_for_example` returns `RetrievalResult`, not a bare tuple

**Status:** Accepted
**Issue:** #14

`retrieve_for_example` previously returned `tuple[str, list[RetrievedChunk]]`. It now returns `tuple[str, RetrievalResult]`, where `RetrievalResult` holds `chunks`, `query_embedding`, and `chunk_embeddings`. The query embedding is obtained by calling `collection._embedding_function([question])` after the ChromaDB query; chunk embeddings are extracted from `query_result["embeddings"][0]` via `include=["embeddings"]`.

The bare tuple was sufficient when chunks were the only output. Adding embedding space analysis required passing pre-computed vectors through the pipeline without re-embedding. Bundling them in a named model makes the contract explicit and avoids positional unpacking errors as the return value grows.

---

## ADR-018: Embedding space analysis accepts pre-computed embeddings, makes no model calls

**Status:** Accepted
**Issue:** #14

`analyze_embedding_space(query_embedding, chunk_embeddings, chunk_ids)` is a pure function — it takes numpy arrays and returns `EmbeddingSpaceMetrics`. It never instantiates or calls an embedding model. Embedding happens once in `_retrieve_with_embeddings` and is threaded through `RetrievalResult`; the analysis function only does geometry (cosine distances, PCA via scikit-learn).

This mirrors the design of `analyze_retrieval_distribution`: forensics functions are pure math, not I/O. It keeps them fast, fully testable with synthetic data, and free of API key requirements.

---

## ADR-019: Never use `or []` on values from `query_result["embeddings"]`

**Status:** Accepted
**Issue:** #14

ChromaDB's `query_result["embeddings"][0]` is a `numpy.ndarray` of shape `(n_results, embedding_dim)`, not a Python list. Using `ndarray or []` raises `ValueError: The truth value of an array with more than one element is ambiguous`. This error was caught by the broad `except Exception` in `retrieve_for_example`, silently returning an empty result and producing a confusing downstream crash in `analyze_embedding_space`.

The fix: assign `raw_chunk_embeddings = query_result["embeddings"][0]` directly and use explicit `is None` checks for any guard logic. Never use Python's boolean short-circuit operators on numpy arrays.

---

## ADR-020: Deterministic lexicon for confidence classification, not an LLM

**Status:** Accepted
**Issue:** #6

Confidence classification (definitive / hedged / uncertain) in `hedging_mismatch.py` uses a hand-coded lexicon and regex word-boundary matching rather than a Claude API call. The alternatives were: (a) another LLM round-trip per claim, or (b) a fine-tuned NLI model. The lexicon approach is deterministic, zero-latency, fully testable with no mocks, and covers the epistemic-marker vocabulary that matters for RAG answer hedging (modal verbs, approximators, attribution shields, first-person softeners). The tradeoff is that novel hedging constructions outside the lexicon will be misclassified as definitive — acceptable given the explicit priority ordering (uncertain > hedged > definitive) and the continuous-fraction output that smooths individual errors.

---

## ADR-021: Entailment response parsed with substring containment, not exact equality

**Status:** Superseded by ADR-035
**Issue:** #15

The entailment step in `hedging_mismatch.py` checks whether Claude's response indicates `supported` or `not_supported`. The original implementation used exact equality after `.strip().lower()`. This silently misclassified any response with trailing punctuation (`"supported."`) or a prefix (`"yes, supported"`) as `not_supported`, biasing `overconfident_fraction` downward. The fix uses an order-of-operations substring check: first reject if `"not_supported"` or `"not supported"` appears, then accept if `"supported"` appears, otherwise log a warning and default to `not_supported`. The warning makes silent misparsing observable without raising an exception that would abort the per-claim loop.

---

## ADR-022: Frontend API functions injected as optional props with library defaults

**Status:** Accepted
**Issue:** #1

`ExampleBrowser` accepts `loadExample` and `analyzeExample` as optional props, defaulting to the real stub implementations from `lib/api.ts`. Tests inject mock functions via props; `page.tsx` renders `<ExampleBrowser />` with no props and gets the defaults. The alternative — importing stubs directly inside the component — would require Jest module mocking (`jest.mock('@/lib/api')`) to test, which is harder to reason about and ties tests to module structure. The alternative of marking `page.tsx` as `"use client"` to allow function-prop passing was rejected: keeping `page.tsx` as a Server Component preserves the option to do server-side data fetching there in future issues (#10, #11) when real backend wiring lands.

---

## ADR-023: Deferred promises for in-flight state assertions in frontend tests

**Status:** Accepted
**Issue:** #1

Frontend tests that assert loading spinner visibility use a `deferred()` helper that returns a `{ promise, resolve }` pair, giving tests explicit control over when async mocks settle. The alternative — `jest.fn(() => new Promise(resolve => setTimeout(resolve, 0)))` — is unreliable because `userEvent.setup()` under React 18's `act()` boundary drains the macro-task queue before returning from `await user.click()`, making the in-flight state unobservable. Deferred promises sidestep this by holding the promise open until the test explicitly calls `resolve()` inside `act()`, making spinner-present and spinner-absent assertions deterministic.

---

## ADR-024: `get_embedding_model()` singleton placed in `retriever.py`, not a separate module

**Status:** Superseded by ADR-039
**Issue:** #7

Chunk attribution needs to embed answer sentences using the same model that ChromaDB uses to embed chunks (`sentence-transformers/all-MiniLM-L6-v2`). Rather than creating a separate `services/embedding.py` module, `get_embedding_model()` was added to `retriever.py` as a module-level cached singleton. The alternative (a dedicated embedding module) would be cleaner if more than one service needed the model, but currently only `chunk_attribution.py` calls it. Colocating it in the retriever keeps the model name in one place and avoids a one-function module with no other responsibility. If a second consumer appears, extract to `services/embedding.py`.

---

## ADR-025: Chunk attribution uses pre-computed chunk embeddings; only sentences are embedded at call time

**Status:** Accepted
**Issue:** #7

`analyze_chunk_attribution(answer, chunks, chunk_embeddings)` accepts chunk embeddings as a pre-computed `list[list[float]]` sourced from `RetrievalResult.chunk_embeddings`. It calls `get_embedding_model().encode(sentences)` only for the answer sentences, which are new content not available at retrieval time. The alternative — re-embedding chunks inside the function — would double the embedding work and couple the forensics module to retrieval internals. This follows the same design as `analyze_embedding_space` (ADR-018): forensics functions are pure math over pre-computed vectors, with embedding happening exactly once in the retrieval layer.

---

## ADR-026: Query-corpus fit module is conditional — triggered by upstream forensics signals

**Status:** Superseded by ADR-040
**Issue:** #8

`analyze_query_corpus_fit` is the only forensics module that can short-circuit to a no-op. It checks three trigger conditions (`query_isolation > 1.2`, `retrieval_relevance_score < 0.5`, or `score_entropy > 1.5 AND faithfulness_score < 0.5`) before making any LLM calls; if none are met it returns a sentinel `_UNTRIGGERED` object immediately. The alternatives were: always run question generation (expensive, noisy for queries that retrieved well), or gate it at the router (spreads conditional logic across layers). Putting the gate inside the module keeps the router unconditional and makes the module self-contained and independently testable — a test can assert the Anthropic client is never called when signals are below threshold.

---

## ADR-027: Mismatch type classified by mean cosine similarity between suggested questions and original query

**Status:** Superseded by ADR-040
**Issue:** #8

After generating suggested questions, each is embedded and its cosine similarity to the original query embedding is computed. The mean of these scores determines `mismatch_type`: `> 0.6` → `query_mismatch` (user was in the right neighborhood, just phrased it differently), `< 0.3` → `coverage_gap` (corpus doesn't cover the topic), otherwise `ambiguous`. The alternative was an additional LLM call to classify the mismatch. Using cosine similarity is zero-cost (model already loaded), deterministic, and directly measures the geometric relationship that defines the two failure modes — it is the most natural signal for this classification.

---

## ADR-028: Prompt uses f-string concatenation, not `str.format()`, for chunk text interpolation

**Status:** Accepted
**Issue:** #8

`build_question_generation_prompt()` in `prompts/query_fit_prompts.py` uses f-string concatenation rather than `str.format()` or a `.format()`-style template. Chunk texts retrieved from a knowledge base frequently contain curly braces (JSON snippets, code examples, template literals). Passing such text through `str.format()` raises `KeyError` or silently corrupts the prompt. The f-string approach interpolates `chunk_texts` and `original_question` at definition time, so brace characters in the content are never interpreted as format placeholders.

---

## ADR-029: Verdict generation is a two-stage pipeline — deterministic rule match, then Claude render

**Status:** Superseded by ADR-036
**Issue:** #9

## ADR-030: `/analyze/custom` computes embeddings inline using the cached singleton from `retriever.py`

**Status:** Accepted
**Issue:** #13

The `/analyze/custom` endpoint accepts pre-scored BYO chunks (no ChromaDB) but still needs embeddings for `chunk_attribution` and `embedding_space`. It computes them inline by calling `get_embedding_model()` from `retriever.py` — the same `SentenceTransformer` singleton used by the retrieval path. The alternative was requiring callers to supply embeddings, which would have made the API harder to use. The singleton is already warm by the time a custom request arrives (server startup loads it via `/example`), so the overhead is just the encode call, not model loading.

---

## ADR-031: Frontend API URL injected via `NEXT_PUBLIC_API_URL` env var, defaulting to localhost

**Status:** Accepted; deployment assumption superseded by ADR-043
**Issue:** #11

`frontend/lib/api.ts` reads `process.env.NEXT_PUBLIC_API_URL` with a fallback of `http://localhost:8000`. This supports configurable local ports and integration environments without code changes; the default serves the local workflow adopted in ADR-043. The snake_case → camelCase mapping from backend response fields is done inside `lib/api.ts` so components work with idiomatic TypeScript field names and are decoupled from the backend's naming conventions.

---

## ADR-032: Oracle evidence is a label-derived diagnostic, not a B4 method

**Status:** Accepted
**Issue:** #19

The oracle-evidence experiment runs only on fully supported RAGBench response sentences with
concrete annotated document-sentence keys. It compares B3's similarity-selected evidence with all
annotated evidence while holding claim decomposition, verifier, and threshold fixed. Maximum
entailment across annotated sentences produces the oracle sentence decision, but every raw
claim/evidence pair is retained. Missing annotations, support sentinels, and verifier errors remain
explicit exclusions or unevaluated states.

This condition is not named B4 and is not included in deployable grounding methods because it uses
benchmark labels unavailable at inference time. Unsupported sentences are excluded from its main
failure-localization claim because RAGBench does not provide a well-defined oracle negative-evidence
sentence. The paired difference can localize supported-sentence false negatives; it cannot
establish production classifier performance or causal responsibility for all RAG failures.

---

## ADR-033: RAGAS context utilization replaces ambiguous retrieval relevance

**Status:** Accepted
**Issue:** #20

RAGAS 0.4.3's exported `context_precision` requires a reference answer, but the project supplied the
sentinel `"N/A"`. The project now uses `ContextUtilization`, whose required inputs are
`user_input`, `response`, and `retrieved_contexts`, and supplies the actual generated or
caller-provided answer. The public construct is renamed from retrieval relevance to
answer-conditioned context utilization because the metric judges whether ranked contexts were
useful for producing that answer; it does not isolate question–context relevance or retriever
quality. This supersedes the metric choice in ADR-007 and renames the score-dependent trigger in
ADR-026.

RAGAS scores are represented as `{score, status, error}`. Evaluation exceptions and non-finite
values produce an explicit unavailable result rather than zero or a request failure. Numeric
query-fit triggers skip unavailable inputs, and verdict ranking exposes unavailable evaluations as
separate diagnostic signals. The same failure model applies to faithfulness because installed
RAGAS can also produce `NaN` there.

---

## ADR-034: Claim extraction is a schema-constrained, locally validated boundary

**Status:** Accepted
**Issue:** #23

Claim extraction supplies the Anthropic Messages API with an exact root-array-of-strings JSON
Schema and repeats strict validation locally before any string operation. Markdown fences,
trailing prose, and malformed JSON are not repaired because accepting a recoverable prefix would
make commentary outside the payload indistinguishable from a valid response. A valid empty array
remains a successful zero-claim observation.

Failures are separated into transport/response access (`claim_extraction_failed`), JSON decoding
(`claim_extraction_parse_failed`), and decoded-schema validation
(`claim_extraction_schema_failed`). All return an unavailable hedging analysis, and downstream
ranking must use the status rather than interpreting the numeric placeholders as healthy zeros.

---

## ADR-035: Exact typed entailment boundary with unavailable judgments

**Status:** Accepted; supersedes ADR-021
**Issue:** #24

Entailment output is a typed string enum with exactly two values: `supported` and
`not_supported`. Production trims surrounding whitespace but does not normalize case,
punctuation, spaces, prefixes, explanations, or other prose. Every attempted chunk records either
an evaluated enum verdict, invalid format with raw output, or request/response error. A claim is
unavailable when none of its chunk attempts yields a valid verdict; unavailable claims do not
enter mismatch-fraction denominators and instead contribute explicit coverage counts and an
unavailable-judgment signal. Evaluation still continues past negative, invalid, and failed checks
and short-circuits on the first valid `supported` verdict. This rejects the permissive substring
strategy in ADR-021 because semantic recognizability is not contract compliance, and because
coercing invalid output to `not_supported` confounds model-format failure with evidence absence.

---

## ADR-036: Verdict reasoning is deterministic and inspectable before bounded rendering

**Status:** Accepted
**Issue:** #21

Ranked signals feed a typed deterministic structure containing observations and reliability,
competing hypotheses, a named component, one discriminating test, and interpretations for each
outcome. Retrieval and generation hypotheses remain separate when both kinds of signal are present.
Unavailable analyses are missing evidence and produce a restore-and-rerun test rather than a clean
interpretation. The API exposes this structure as `verdict_reasoning` while retaining
`verdict_signals` and recommendation prose.

Claude may only render the supplied structure; it cannot select a cause, component, test, or
outcome. A failed rendering returns a deterministic serialization of the complete structure. This
implements the structural lesson from issue #16's small frozen proxy-model evaluation without
claiming that deterministic scaffolding is generally superior or that production reliability has
been established. Any production-improvement claim still requires validation through the exact
production Anthropic SDK and configured model, and held-out cases must not be used for iterative
prompt tuning.

---

## ADR-037: Retrieved-context fit requires three validated, semantically distinct questions

**Status:** Accepted
**Issue:** #22

Generated question candidates cite the retrieved chunk IDs needed to answer them. Production
rejects citations outside the retrieved set, asks a separate structured model pass to validate
direct answerability and specificity from the cited text, and greedily rejects later candidates whose embedding
has cosine similarity `>= 0.90` to an accepted question. At least three accepted questions are
required before their mean similarity to the original query can produce a retrieved-context fit
label. A smaller set is preserved for inspection but returns
`error="insufficient_valid_questions"`, with no label or mean similarity. Proceeding from one or
two questions would let a narrow or unstable sample drive a confident downstream signal;
discarding all evidence would make generation failures harder to audit. This contract concerns
only retrieved passages and does not establish full-corpus coverage.

---

## ADR-038: Diagnostic modules return typed observations, not tri-state verdicts

**Status:** Accepted; supersedes ADR-008, ADR-009, and ADR-011

The interactive analysis path exposes five complementary forensics modules:
`retrieval_distribution`, `embedding_space`, `chunk_attribution`, `hedging_mismatch`, and
`query_corpus_fit`. Each returns its own typed metrics and method-specific availability semantics;
modules do not return a shared `DimensionResult`, development stubs, or `pass`/`warn`/`fail`
labels. RAGAS metrics similarly return `{score, status, error}` rather than coerced verdicts.

Numeric observations remain inspectable at their native granularity. Retrieval-distribution shape
is descriptive rather than probabilistic evidence of quality, and a failed decay fit is represented
as `null`, not zero. The verdict layer may rank observations for investigation, but its priorities
are heuristic ordering indices rather than calibrated severities or replacements for the underlying
typed results. This structure preserves method differences and prevents unavailable measurements
from appearing healthy.

---

## ADR-039: One retriever-owned embedding model defines the local analysis space

**Status:** Accepted; supersedes ADR-024

The cached `sentence-transformers/all-MiniLM-L6-v2` instance remains owned by `retriever.py`, but it
is a shared project service used by retrieval, sentence attribution, retrieved-context-fit analysis,
and `/analyze/custom`. Keeping one canonical instance prevents the embedded retrieval path from
silently comparing vectors produced by different models and avoids duplicate model loading.

This choice deliberately couples the local analyses to the project's embedding space. Caller
scores and text from an external RAG system may have been produced in a different space;
`/analyze/custom` therefore re-embeds caller text locally and must disclose that its geometric
observations do not reproduce the caller's production retriever geometry. If the project later
supports multiple embedding models or caller-supplied embeddings, model identity and revision must
become explicit request and provenance fields before cross-space comparisons are allowed.

---

## ADR-040: Retrieved-context fit is conditional and cannot establish corpus coverage

**Status:** Accepted; supersedes ADR-026 and ADR-027; complements ADR-037

Retrieved-context-fit analysis runs only when query isolation exceeds `1.2`, answer-conditioned
context utilization is available and below `0.5`, or normalized retrieval entropy exceeds `0.9`
while available faithfulness is below `0.5`. Unavailable upstream scores do not satisfy numeric
triggers. Triggering initiates question generation and validation; it does not itself produce a fit
label.

After the validation and diversity requirements in ADR-037 are satisfied, mean cosine similarity
to the original question yields `retrieved_context_near_miss` above `0.6`,
`retrieved_context_topic_gap` below `0.3`, and `ambiguous` otherwise. These labels describe only the
retrieved passages. They do not show that query wording caused a failure or that the full corpus
contains or lacks an answer. The rename from `query_mismatch` and `coverage_gap` makes that evidence
boundary part of the public contract.

---

## ADR-041: Chunk completeness is a provenance-bearing source-boundary state

**Status:** Accepted

Every retrieved chunk carries `completeness` (`complete`, `truncated`, or `unknown`) separately from
`completeness_source` (`source`, `caller`, or `unavailable`). Known states require source or caller
provenance; unknown requires unavailable provenance. Missing or malformed stored metadata fails
closed to `unknown`/`unavailable`, and custom callers may assert a known state only with caller
provenance. Terminal punctuation is never promoted to source-boundary evidence.

Generation receives these states explicitly. It must not guess a known-truncated continuation and
must disclose truncation when it prevents a complete answer; it must not describe an unknown chunk
as truncated. The API returns structured chunk details so consumers can inspect the state without
parsing prompt text. This extends the grounding instruction recorded in ADR-006: source-boundary
metadata constrains generation but is not evidence that a chunk is relevant or sufficient.

---

## ADR-042: Retrieval scores are bounded observations, not cross-retriever calibrated quantities

**Status:** Accepted; narrows ADR-005

The embedded Chroma path continues to convert its configured distance with
`max(0, 1 - distance)`, while `/analyze/custom` accepts caller-declared
`normalized_similarity` values in `[0, 1]`. The common numeric range is an API bound, not evidence
that scores from different retrievers, embedding models, rerankers, or corpora have equivalent
meaning or calibration.

Distribution shape and absolute thresholds may be interpreted only within a documented score
semantics and compatible retrieval configuration. Comparative studies must preserve native scores
and provenance rather than treating the range as a shared measurement scale. Supporting BM25,
distances, logits, or additional retrievers requires an explicit score-semantics contract rather
than automatic conversion.

---

## ADR-043: RAG Forensics is distributed for local operation, not hosted as a service

**Status:** Accepted

The intended delivery model is a reproducible local tool that a researcher or developer can run
beside an existing RAG system and call through `/analyze/custom`. A centrally hosted, public, or
multi-tenant deployment is not a project end goal. Local operation keeps caller-provided questions,
answers, and retrieved context under the operator's control and avoids turning research software
into an externally operated data-processing service.

Packaging should minimize setup friction while preserving inspectability: dependencies and model
revisions remain pinned, data bootstrap is explicit and idempotent, local persistence is
documented, configuration failures are visible, and an end-to-end example verifies integration.
Container or release artifacts may support reproducible local installation, but they must not
introduce a hosted-service requirement. The localhost CORS and API URL defaults in ADR-014 and
ADR-031 remain appropriate for development; their assumptions about an eventual Vercel/Railway
deployment are superseded by this decision.

---

## ADR-044: Comparative diagnostic tools run in isolated environments, never as backend dependencies

**Status:** Accepted
**Issue:** #30

A feasibility check installing RAGChecker and RAGVue together confirmed a real dependency
conflict: RAGChecker's `refchecker` dependency pins `anthropic<0.30,>=0.29`, while this project
pins `anthropic>=0.86,<0.87` for its own generation and forensics LLM calls. The two cannot coexist
in one resolved environment without downgrading the SDK the backend itself depends on — and the
older SDK version is independently broken against the project's pinned `httpx`
(`Client.__init__() got an unexpected keyword argument 'proxies'`), which surfaced as a live
failure in RAGVue's Anthropic-backed judge when both tools shared a venv.

RAGChecker and RAGVue are therefore never added to `backend/pyproject.toml`. Each runs in its own
isolated virtual environment, invoked as a subprocess or driver script whose JSON output is read
back verbatim into `NativeSystemOutput.raw_output`
(`backend/benchmark/comparative_diagnostics.py`). This preserves each tool's own dependency
resolution and intended semantics, keeps the backend's dependency graph untouched, and matches how
an external user would actually run these tools — as independent processes, not library imports.

A related finding shaped the schema itself: RAGVue's own metadata mislabeled which model produced
a judgment (`raw.model: "gpt-4o-mini"` even when configured for and actually using Claude via
`RAGVUE_JUDGE_PROVIDER=anthropic`). A comparative record must not trust a system's self-reported
provenance field; the run configuration we actually set is the source of truth for `method`, while
the raw self-reported output is preserved unmodified for inspection.

---

---

## ADR-045: One LLM boundary, explicit failure classes, and readiness that matches the local workflow

**Status:** Accepted
**Issue:** #12

Four services each constructed their own Anthropic client, indexed `response.content[0].text`
without checking block type, and caught `Exception` to degrade gracefully. That made request
timeouts impossible to set in one place and let programming errors masquerade as "the model
failed", which is the opposite of the visible-failure requirement in ADR-043.

All backend LLM calls now go through `services/llm.py`. It builds the client with a bounded timeout
and retry count, returns the first text block, and converts SDK failures into `LLMError`. Modules
degrade only on `LLMError` and on malformed model output (`ValueError`, including JSON decode
errors); the query-fit module also degrades when the local embedding model cannot load or run
(`RuntimeError`, `OSError`), because its explicit `fit_computation_failed` status predates this
change. Any other exception propagates to the endpoint, which logs it and returns a generic 500.
Missing `ANTHROPIC_API_KEY` raises at client construction and is deliberately not caught, so a
missing key fails visibly rather than as a degraded diagnostic.

HTTP 500 responses no longer echo exception text, which can contain paths or upstream messages;
detail stays in server logs. `GET /health` is liveness only. `GET /ready` is 503 when
`ANTHROPIC_API_KEY` is unset and otherwise 200, and it reports bundled-corpus availability without
gating on it: `/analyze/custom` is the primary supported path and must work without bootstrapping
the RAGBench corpus (ADR-043). Readiness performs no model calls.

`ruff` (lint only, no repo-wide reformat, so frozen evaluation code is untouched) and `mypy` are
configured in `backend/pyproject.toml`, and CI runs ruff, pytest, eslint, and jest. mypy runs
informationally until the remaining type errors are cleared. Evaluation and benchmark scripts keep
their own pinned model IDs and broad handlers that record evaluator failures; they are excluded
from the blind-except rule because reproducibility of versioned results takes precedence there.

## ADR-046: Application setup, separate RAGAS boundary, and staged corpus replacement

**Status:** Accepted
**Issue:** #12

The backend is a local application managed with Poetry's non-package mode. Installation uses the
lock file without attempting to build a root package. The supported native setup uses Python 3.11,
Poetry 2.2.1, and Node.js 22; custom analysis is available without bootstrapping the demo corpus.
The canonical setup and verification commands live in
[Local setup and verification](docs/reference/local-setup.md).

This corrects ADR-045's scope statement: the shared `services/llm.py` helper covers direct project
calls, while RAGAS retains its `ChatAnthropic` adapter and dependency-owned prompts and parsing.
Both paths take a 60-second request timeout and two SDK retries from `config.py`. RAGAS evaluation
uses a 60-second task timeout and one outer attempt, avoiding multiplication by its default retry
policy. Its broad exception boundary still records an unavailable metric. Frozen experiments keep
their own configuration. None of these settings establishes a total analysis deadline.

Corpus replacement builds a staging collection before renaming the existing collection to a
backup and promoting the replacement. Promotion failure attempts rollback; the backup is deleted
only after promotion succeeds. Empty inputs are rejected. Fallback IDs use domain-prefixed SHA-256
question digests instead of Python's process-randomized hash. Bootstrap runs with the backend
stopped and is not a concurrent or crash-atomic operation; interrupted renames may require manual
recovery from the retained backup. This preserves the embedded Chroma architecture without
claiming transactional guarantees across collection renames or domains.

The maintained smoke client checks liveness/readiness by default and requires `--analyze` to send
the bundled public request to the model-backed analysis path. Historical notebooks are explicitly
marked unsupported. CI checks frontend TypeScript compilation alongside lint and Jest. Full live
clean-environment verification and model/dataset revision pinning remain work under issue #12.

## ADR-047: Python 3.13 as the primary runtime with minimum-version CI

**Status:** Accepted
**Issue:** #12

Python 3.13 replaces Python 3.11 as the primary local setup version specified in ADR-046. Python
3.11 remains the declared minimum and a separate CI matrix job; dependency constraints and lint
and type-check targets retain that minimum. The primary version was verified by installing the
committed lock into a fresh Python 3.13.11 virtual environment on macOS and running the offline
suite. This is interpreter/dependency verification, not a live clean-machine acceptance run.

The locked LangChain/Pydantic stack emits a Pydantic V1 compatibility warning on Python 3.14.
Python 3.14 also deferred evaluation of an invalid Chroma client annotation that failed during
imports on both Python 3.11 CI and the fresh Python 3.13 environment. The annotation now names
Chroma's `ClientAPI` rather than its `PersistentClient` factory. Keeping both primary and minimum
versions in CI makes such runtime differences visible. Python 3.14 remains allowed by the package
metadata but is not the recommended local setup.

[CONTRIBUTING.md](CONTRIBUTING.md) is the tracked source of shared contributor instructions,
superseding ADR-012's reliance on the local-only `CLAUDE.md`. Model and download boundaries remain
mocked in automated tests; database tests may use isolated temporary storage, as the bootstrap
tests do, without modifying a developer's corpus.

## ADR-048: Logs record operations, never caller content

**Status:** Accepted
**Issue:** #12

ADR-043 requires that caller content leave the local process only for explicitly documented model
providers. Logs are a second, easily overlooked exit: the backend defaulted to DEBUG, and the
Anthropic SDK logs request payloads at DEBUG, so question, answer, and chunk text appeared in logs
by default. This was confirmed with synthetic data and a mocked transport. Application log calls
also included question text, claim text, and raw model output, and failure logging attached
exception messages, which routinely echo their input (validation and provider errors do).

The backend now defaults to INFO. The `anthropic` logger is pinned to WARNING alongside the other
noisy third-party loggers, so SDK payloads stay out of logs even when the host configures root
logging at DEBUG. Application logs carry identifiers, counts, lengths, statuses, and exception
types, not content. Failures are logged through `services/failure_detail.py`, which emits the
exception type and stack frames but neither the message nor any chained cause. The cost is that
server logs no longer show why a failure occurred, only where; that is accepted for a tool that
processes caller data, and a developer who needs messages can reproduce the failure locally.

Regression tests enforce the policy: a subprocess test checks that a mocked SDK call does not reach
the log in either root-logging mode, and per-module tests assert that marker strings from
questions, claims, model output, and exception messages never appear in captured logs, even at
DEBUG. New log calls that include caller-derived values should be treated as violations of this
decision.
