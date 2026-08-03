# RAG Forensics documentation

The documentation is organized by reader intent. Start with an explainer for an approachable
introduction to a concept, or use the reference material for implementation details, experimental
protocols, and reproducible commands.

## Explainers

- [Understanding the oracle-evidence experiment](explainers/oracle-evidence.md) explains how the
  proposed experiment separates evidence-selection errors from downstream verification errors,
  with examples and explicit limits.

## Reference

- [Methods and architecture](reference/methods.md) documents the diagnostic methods, output
  semantics, architecture, and limitations.
- [Benchmarking and current evidence](reference/benchmarks.md) records protocols, commands,
  empirical results, uncertainty, and the current research boundary.
- [Custom API integration](reference/api-integration.md) describes the request and response
  contracts for analyzing caller-provided RAG outputs.

## Project records

- [Architectural decisions](../ADR.md) preserves the project's append-only decision history.

The repository's main [README](../README.md) gives the shortest overview and quick-start path.
