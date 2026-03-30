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

**Status:** Accepted
**Issue:** #3

ChromaDB returns L2 distances (lower = closer). For the rest of the pipeline to reason about scores uniformly, distances are converted to similarity scores clamped to `[0.0, 1.0]` via `score = max(0.0, 1.0 - distance)`. All downstream code (forensics, scoring, API responses) works with similarity scores, not raw distances.

---

## ADR-006: Claude (Anthropic SDK) for answer generation, not an OpenAI-compatible wrapper

**Status:** Accepted
**Issue:** #4

Used the Anthropic Python SDK directly rather than routing through LangChain or an OpenAI-compatible shim. This keeps the dependency surface smaller and avoids abstraction layers that can obscure errors. Model: `claude-haiku-4-5-20251001` (fast and cheap for generation).

The generator's system prompt enforces grounding: "use ONLY information from the provided context chunks."

---

## ADR-007: RAGAS for faithfulness and retrieval relevance scoring

**Status:** Accepted
**Issue:** #4

Rather than writing custom faithfulness/relevance scoring from scratch, delegated to the RAGAS library (`faithfulness` + `context_precision` metrics). RAGAS is purpose-built for RAG evaluation and uses an LLM judge internally. Bound RAGAS to Claude via the `ChatAnthropic` LangChain adapter.

---

## ADR-008: Tri-state verdicts (pass / warn / fail) rather than raw scores

**Status:** Accepted
**Issue:** #4

RAGAS returns continuous scores in `[0.0, 1.0]`. These are bucketed into `pass` (≥ 0.75), `warn` (≥ 0.5), and `fail` (< 0.5) before being returned to the caller. Continuous scores are ambiguous for users — tri-state verdicts force a clear interpretation. The thresholds are explicit constants, easy to tune.

---

## ADR-009: Each forensics module is independent and callable in isolation

**Status:** Accepted
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

**Status:** Accepted
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

**Status:** Accepted
**Issue:** #2

The FastAPI backend allows cross-origin requests only from `http://localhost:3000`, which is the Next.js dev server. No wildcard origins. This will need revisiting when the project is deployed (Vercel URL will need to be added).

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
