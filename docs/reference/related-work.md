# Related work in RAG evaluation and debugging

This review situates RAG Forensics among systems that evaluate, diagnose, debug, or optimize
retrieval-augmented generation pipelines. It is a focused review of primary publications rather
than a systematic literature review. The sources were reviewed on 2026-09-05; descriptions of
software and unpublished preprints may change.

## Evaluation and diagnosis

RAGAS established a reference-free evaluation framework that separates context relevance,
faithfulness, and answer relevance. ARES evaluates similar dimensions with task-adapted lightweight
judges and prediction-powered inference. Both primarily characterize system quality through metric
estimates rather than reconstructing the investigation of an individual failure.

RAGChecker is a closer diagnostic baseline. It uses claim-level entailment to provide retriever and
generator metrics, including claim recall, context precision, context utilization, noise
sensitivity, hallucination, self-knowledge, and faithfulness. Its meta-evaluation compares metric
outputs with human judgments, providing empirical support for the reliability of its diagnostic
measurements.

RAGVUE is the closest published comparison in purpose and presentation. It separates retrieval,
answer quality, grounding, and judge stability; produces structured explanations; and supports
inspection of individual question-answer-context records. Its published evaluation compares
RAGVUE with RAGAS on 100 synthetic StrategyQA-derived cases. This establishes substantial overlap
with RAG Forensics in fine-grained, explanation-bearing evaluation, although the two systems
organize their outputs differently.

## Interactive debugging and optimization

RAGGY addresses debugging as an interactive workflow. Developers can alter pipeline stages and
propagate changes through a composable RAG pipeline, enabling direct what-if analysis. Its design is
informed by a qualitative study with 12 engineers. RAGSmith instead treats RAG design as
dataset-level architecture search across nine technique families and 46,080 feasible
configurations. These systems represent two complementary intervention scales: interactive
component experimentation and global pipeline optimization.

Doctor-RAG extends diagnosis into automated repair for agentic RAG. It identifies the earliest
failure in a retrieval-reasoning trajectory and applies a targeted repair while reusing the valid
prefix. Because it studies multi-step agentic trajectories rather than completed static RAG
records, it is adjacent rather than directly comparable.

## Comparison

| Work | Primary purpose | Unit and evidence | Relation to RAG Forensics |
|---|---|---|---|
| [RAGAS](https://aclanthology.org/2024.eacl-demo.16/) (2024) | Reference-free evaluation | Response- and dataset-level LLM-judged metrics | Supplies two model-judged observations used by RAG Forensics; does not represent the full diagnostic record |
| [ARES](https://aclanthology.org/2024.naacl-long.20/) (2024) | Automated system evaluation | Population estimates supported by synthetic training data and a small human-labeled set | Provides component scores and statistical uncertainty; RAG Forensics emphasizes case-level hypotheses whose priorities are not calibrated probabilities |
| [RAGChecker](https://arxiv.org/abs/2408.08067) (2024) | Fine-grained retriever and generator diagnosis | Claim-level metrics, a benchmark, and meta-evaluation against human judgments | Direct diagnostic baseline with substantial overlap in claim decomposition, grounding, and context-utilization analysis |
| [RAGGY / RAG Without the Lag](https://doi.org/10.1145/3772318.3790874) (2026) | Interactive what-if debugging | Pipeline stages and parameter changes; qualitative study with 12 engineers | Executes pipeline interventions directly, whereas RAG Forensics records a proposed intervention for a completed case |
| [RAGVUE](https://aclanthology.org/2026.eacl-demo.35/) (2026) | Explainable reference-free diagnosis | Individual records and aggregate reports with structured explanations and cross-judge calibration | Closest direct peer in fine-grained, explanation-bearing evaluation |
| [RAGSmith](https://arxiv.org/abs/2511.01386) (2025) | End-to-end architecture search | Dataset-level optimization across RAG configurations | Selects a pipeline for a dataset rather than investigating one output |
| [Doctor-RAG](https://arxiv.org/abs/2604.00865) (2026 preprint) | Diagnosis and local repair of agentic RAG | Retrieval-reasoning trajectories, error taxonomy, localization, and repair operators | Extends diagnosis into automated repair for a different, agentic setting |

## Position of RAG Forensics

RAG Forensics studies how heterogeneous diagnostic observations can be assembled without treating
them as proof of a unique cause. Its primary artifact is an **inspectable investigation record**
that preserves the distinction between observations and hypotheses. Each observation retains its
method and reliability; evaluator failures remain unavailable rather than becoming numeric scores;
competing explanations remain visible; and a proposed intervention includes interpretations for
possible outcomes.

This emphasis differs from metric suites that primarily estimate quality, interactive tools that
execute pipeline changes, and optimizers that select configurations. It also overlaps materially
with RAGChecker's component diagnostics and RAGVUE's structured explanations. The contribution is
therefore best understood as a particular organization of diagnostic evidence and experimental
follow-up, accompanied by label-preserving evaluations that report mixed and null findings.

The present evidence demonstrates the structure of the record and evaluates selected grounding
methods. It does not establish higher diagnostic accuracy, faster diagnosis, or improved developer
decisions relative to related systems.

## Comparative evaluation priorities

A direct comparison with RAGChecker and RAGVUE could present developers with the same labeled
failures and measure:

1. correctness and calibration of the suspected failing component;
2. whether unsupported causes remain visibly uncertain;
3. the quality and discriminating value of the proposed next test;
4. time and number of interventions needed to reach a correct repair; and
5. evaluator failures or missing inputs mistakenly presented as healthy results.

The comparison above consequently describes differences in documented design and published
evidence, not performance rankings.

A feasibility check (issue #30) confirmed both frameworks run on the same public inputs, but only
from isolated environments, not as dependencies of this backend. It also found that RAGVue's
self-reported provenance can be wrong, so the frozen comparison schema treats it as untrusted. The
findings, schema, and the case-selection protocol (frozen before any case is chosen) are in
[`backend/evals/comparative_diagnostics/v1/README.md`](../../backend/evals/comparative_diagnostics/v1/README.md)
and [`CASE-SELECTION-PROTOCOL.md`](../../backend/evals/comparative_diagnostics/v1/CASE-SELECTION-PROTOCOL.md).

## Scope and exclusions

The review prioritizes primary publications for systems that explicitly address RAG evaluation,
diagnosis, debugging, or optimization. It is not an exhaustive comparison of general LLM tracing
and observability products. Domain-specific work such as
[RAG-X](https://arxiv.org/abs/2603.03541), which diagnoses retrieval and generation for medical
question answering, reinforces the broader conclusion that component-level diagnosis is an active
and non-unique research direction, but it is not a direct general-purpose baseline for this
repository.
