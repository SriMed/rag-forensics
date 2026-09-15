# RAG Forensics documentation

The documentation is organized by reader intent. Start with an explainer for an approachable
introduction to a concept, or use the reference material for implementation details, experimental
protocols, and reproducible commands.

## Explainers

- [How RAG Forensics investigates an answer](explainers/how-rag-forensics-works.md) follows one
  example from retrieved passages through evidence candidates, competing hypotheses, follow-up
  tests, the offline grounding evaluators, and an oracle condition that substitutes annotated
  evidence to isolate one step of the method.
- [Understanding the oracle-evidence experiment](explainers/oracle-evidence.md) explains how the
  completed experiment separates evidence-selection errors from downstream verification errors,
  with its interpretation and explicit limits.
- [Understanding the decomposition-by-evidence experiment](explainers/decomposition-by-evidence.md)
  explains what makes a claim well-formed enough to verify, why the blinded review can't reason
  from evidence, why claim count affects the measured outcome on its own, and how the four-condition
  design compares claim representations and evidence conditions without isolating decomposition quality.

## Reference

- [Methods and architecture](reference/methods.md) documents the diagnostic methods, output
  semantics, architecture, and limitations.
- [Related work in RAG evaluation and debugging](reference/related-work.md) compares the project's
  contribution and evidence with diagnostic evaluators, interactive debuggers, and pipeline
  optimizers while recording the limits of that comparison.
- [Benchmarking and current evidence](reference/benchmarks.md) records protocols, commands,
  empirical results, uncertainty, and open research questions.
- [Custom API integration](reference/api-integration.md) describes the request and response
  contracts for analyzing caller-provided RAG outputs.
- [Prompt development evaluation](reference/prompt-evaluation.md) documents the versioned prompt
  cases, deterministic scorers, held-out split, review schema, and evidence limits.
- [LLM prompt and model-boundary audit](reference/prompt-audit.md) records the decisive findings,
  proxy evaluation, implemented mitigations, and remaining limitations.
- [Installed RAGAS prompt audit](reference/ragas-prompt-audit.md) records the dependency-owned
  prompt contracts, model configuration, parsing and failure behavior, and downstream influence.
- [Truncated-evidence generation evaluation](reference/truncated-evidence.md) records the paired
  protocol, observed extrapolation and abstention behavior, source-metadata contract, exact-model
  comparison, and evidence limits from issues #25 and #27.

## Project records

- [Architectural decisions](../ADR.md) preserves the project's append-only decision history.
- [Domain language](../CONTEXT.md) defines the canonical names used for grounding evaluators and
  oracle-evidence conditions.

The repository's main [README](../README.md) gives the shortest overview and quick-start path.
