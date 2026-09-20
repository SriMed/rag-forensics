# RAG Forensics

> An inspectable diagnostic record for investigating why a RAG answer may have gone wrong.

RAG evaluations commonly collapse a response into scores such as faithfulness or relevance.
Those scores can identify concern, but they rarely identify a unique cause. A weak answer may
reflect retrieval, missing evidence, contradiction, changed qualifiers, generation, or simply a
failed evaluator. Treating one observed score as proof of one cause hides those alternatives.

RAG Forensics keeps them visible. Given a question, answer, and retrieved context, it records
observable signals, evidence candidates, method assumptions, reliability labels, ranked
hypotheses, and follow-up tests. It is a hypothesis-generation layer—not a root-cause oracle.

The project follows one chain: **ambiguous failure → inspectable observations → competing
explanations → an intervention that could discriminate between them → controlled evaluation of the
individual measurement components.** It runs locally beside an existing RAG system and is
distributed as inspectable research software, not a hosted service. It was motivated by recurring
diagnostic ambiguity in a production RAG system; no proprietary incidents, outputs, or user
research are included, and every empirical claim here comes from public datasets and committed
evaluation artifacts.

## Quick start

Use Python 3.13 and Poetry 2.2.1 for the backend. From the repository root:

```bash
cd backend
poetry env use python3.13
poetry install
cp .env.example .env
# Replace your_key_here in .env with your Anthropic API key.
poetry run uvicorn main:app --host 127.0.0.1 --port 8000
```

Custom analysis needs no corpus bootstrap. In another terminal, from `backend/`:

```bash
poetry run python -m scripts.smoke_local
# Optional live check: sends the bundled public example to Anthropic and incurs API usage.
poetry run python -m scripts.smoke_local --analyze
# Offline tests: external model calls are mocked; no API key is required.
poetry run pytest
```

See [Local setup and verification](docs/reference/local-setup.md) for frontend startup, optional
demo bootstrap, storage and caches, failure recovery, and the limits of these checks. For a worked
example of the output, read
[How RAG Forensics investigates an answer](docs/explainers/how-rag-forensics-works.md).

## Where to look in the code

| If you want to see… | Start at |
|---|---|
| The request path and how analyses are combined | [`backend/routers/analyze.py`](backend/routers/analyze.py) |
| The five forensics modules | [`backend/services/forensics/`](backend/services/forensics/) |
| How signals become ranked hypotheses and a follow-up test | [`backend/services/verdict_generator.py`](backend/services/verdict_generator.py) |
| Response shapes and availability semantics | [`backend/models.py`](backend/models.py) |
| The offline grounding evaluators and their held-out comparison | [`backend/benchmark/grounding.py`](backend/benchmark/grounding.py) |
| The oracle-evidence diagnostic | [`backend/benchmark/oracle_evidence.py`](backend/benchmark/oracle_evidence.py) |

## The argument

A diagnostic system should distinguish:

1. **what it observed**;
2. **how that observation was produced**;
3. **what the observation can and cannot establish**;
4. **which competing explanations remain**; and
5. **what intervention could discriminate between them**.

RAG Forensics implements that contract at the level of individual answers.

| Observation | Supports investigating | Does not establish |
|---|---|---|
| Flat or sharply decaying retrieval scores | retriever uncertainty or concentration | whether any retrieved passage is correct |
| Query isolated from retrieved embeddings | retrieved-context mismatch | that the full corpus lacks an answer |
| Low answer-to-context similarity | a weak source candidate | contradiction, hallucination, or lack of entailment |
| Definitive language with weak evidence | possible epistemic overstatement | the correct confidence level |
| Retrieved content answers a different question | query/retrieval intervention | whether query wording caused the failure |

The verdict layer ranks these signals as heuristic priorities. Its values are not probabilities,
calibrated severities, or causal attributions.

## What the system produces

- retrieval-distribution and embedding-space observations;
- sentence- or claim-level source candidates with raw scores;
- hedging/evidence mismatch observations;
- conditional retrieved-context-fit hypotheses;
- explicit evaluator failures instead of healthy-looking zeroes;
- ranked investigation signals and falsifiable follow-up tests;
- label-preserving RAGBench reports and explicit span-to-sentence RAGTruth reports.

The API exposes two entry points:

- `POST /example` runs the demonstration pipeline over a stored RAGBench example.
- `POST /analyze/custom` accepts a caller-provided question, answer, and retrieved chunks.

Two operational endpoints make no model calls: `GET /health` (liveness) and `GET /ready`, which
returns 503 until `ANTHROPIC_API_KEY` is configured and reports whether the bundled RAGBench corpus
has been bootstrapped. Server errors return a generic message; details are in the server logs.

See [Methods and architecture](docs/reference/methods.md) and
[integration documentation](docs/reference/api-integration.md) for the full contracts.

## What the evidence currently says

Grounding methods are evaluated on RAGBench without regenerating answers or changing their source
documents. Thresholds are selected on validation data and evaluated on untouched test data with
clustered confidence intervals. The principal comparison covers prevalence checks, a
**whole-sentence similarity evaluator**, a **claim-similarity evaluator** using deterministic
decomposition, and a **claim-entailment evaluator** that adds a pinned third-party NLI
cross-encoder to the same claims and evidence candidates. The evaluators are offline experimental
methods, not the product; RAG Forensics tests the NLI verifier because the evaluator relies on
its output, not because it created it.

**Null result.** On a seeded sample of up to 100 validation and 100 test examples from each
RAGBench domain:

| Held-out macro metric | Whole-sentence similarity | Claim-entailment |
|---|---:|---:|
| F1 | 0.301 | 0.278 |
| AUPRC | 0.215 | 0.247 |

The paired claim-entailment minus whole-sentence-similarity macro-F1 difference was `-0.022` with
a 95% interval of `[-0.066, 0.025]`. Macro AUPRC increased, but its interval also included no
improvement, and the direction varied by domain. A small RAGTruth external-validation run was
similarly mixed. The evidence does **not** support either evaluator as a reliable standalone
grounding detector. Similarity remains useful for navigating to candidate evidence.

**Oracle-evidence diagnostic.** Replacing the claim-entailment evaluator's selected evidence with
RAGBench's human-annotated supporting evidence tests the verification step separately. On 188
eligible supported sentences, the false-unsupported rate fell from `0.452` to `0.287`; the paired
difference was `-0.165` with a 95% example-clustered interval of `[-0.230, -0.101]`. Evidence
selection is therefore a meaningful—but not exclusive—bottleneck: substantial errors persist even
with annotated evidence. This is label-derived analysis, not a deployable classifier, and it does
not explain failures on unsupported sentences.

**TechQA pilot.** Crossing deterministic versus human-reviewed claims with selected versus
annotated evidence on 39 eligible sentences, all paired-effect 95% intervals included zero.
Residual review raised a hypothesis that some rejections require evidence sentences to be supplied
jointly; it did not establish a decomposition-quality effect or a general verifier limitation. See
the [pilot results](docs/reference/benchmarks.md#techqa-pilot-result) and
[interpretation](docs/explainers/decomposition-by-evidence.md#a-preliminary-finding-from-the-techqa-pilot).

The best-supported contribution is the transparent, label-preserving framework that makes these
results—and their remaining uncertainty—inspectable. Protocols, commands, revisions, and
limitations are in [Benchmarking and current evidence](docs/reference/benchmarks.md).

### Other evaluations

These studies are small and purposive; each states its own evidence boundary.

- [Truncated-evidence generation](docs/reference/truncated-evidence.md): how generation behaves
  when supplied evidence is visibly cut off.
- [Prompt and model-boundary audit](docs/reference/prompt-audit.md) and the
  [installed RAGAS prompt audit](docs/reference/ragas-prompt-audit.md): what each LLM boundary
  asks, how it fails, and what the audit did not test.
- [Prompt development evaluation](docs/reference/prompt-evaluation.md): versioned cases,
  deterministic scorers, and a held-out split.

## Limits and next steps

The project has not established that this diagnostic record improves decisions for real users, and
the ranking and follow-up-test layer has not been evaluated for whether it discriminates between
hypotheses. There are no external-consumer or production-incident data; the present evaluation
targets diagnostic validity, provenance, failure semantics, and controlled interventions on public
data. The evidence supports reading the system as narrowing an investigation, not identifying the
cause of a bad answer.

Next, evaluation should separate claim-decomposition errors, evidence-selection errors,
multi-sentence or numerical reasoning failures, verifier errors, and annotation-granularity
mismatches. A planned comparison with RAGChecker and RAGVue on public cases has a frozen schema and
protocol but no selected cases; see
[`backend/evals/comparative_diagnostics/v1/`](backend/evals/comparative_diagnostics/v1/README.md).

## Documentation

- [Documentation guide](docs/README.md) — the full map, organized by reader intent
- [Local setup and verification](docs/reference/local-setup.md)
- [Worked example of the investigation workflow](docs/explainers/how-rag-forensics-works.md)
- [Methods, outputs, architecture, and limitations](docs/reference/methods.md)
- [Benchmark protocol, results, and reproducible commands](docs/reference/benchmarks.md)
- [Architectural decisions](ADR.md) and [contributing](CONTRIBUTING.md)
