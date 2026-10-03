# Stratum rules for issue #30 (comparative-diagnostics.v1)

This file freezes [`CASE-SELECTION-PROTOCOL.md`](CASE-SELECTION-PROTOCOL.md) step 2 (stratum eligibility) and step 3 (selection) before anyone looks at the candidate-pool outputs. The code is [`backend/benchmark/comparative_strata.py`](../../../benchmark/comparative_strata.py), tested in [`backend/tests/test_comparative_strata.py`](../../../tests/test_comparative_strata.py). If this file and the code disagree, the code is what ran; fix the disagreement with a new protocol version, not an edit. The owner settled the four open decisions on 2026-10-03; ADR-056 records them.

## Inputs

- The pool run written by [`run_pool.py`](run_pool.py): every system's native output and mapped record for each of the 98 candidate-pool cases.
- The RAGBench record for the case (dataset label, domain, response).
- The oracle-evidence diagnostic's per-sentence predictions for the case's eligible sentences, run locally at the pinned NLI verifier revision (no LLM calls).

## Exclusions

A case is excluded, and the reason recorded in the manifest, when any system's batch crashed and was not rerun, or when RAG Forensics produced no output at all (which also removes the RAGAS baseline, since RAGAS runs inside it). A system that *reports* its own failure is not excluded: that is what the `single_system_exposes_failure` stratum is for.

## Per-system "flags unsupported content"

A flag means at least one claim is unsupported, matching RAGBench's label, where one unsupported sentence makes a response `contains_unsupported`. A system with no usable score gets no flag and is left out of agreement checks.

| System | Flags unsupported when |
|---|---|
| RAG Forensics | its top-ranked signal is `low_faithfulness`, `unattributed_content` or `overconfidence` |
| RAGAS baseline | `ragas.faithfulness` has status `ok` and score < 1.0 |
| RAGChecker | the case is healthy and `faithfulness` < 1.0 |
| RAGVue | `strict_faithfulness` ran without an embedded error and scored < 1.0 |

## Strata

| Stratum | Eligible when | Target |
|---|---|---:|
| `single_system_exposes_failure` | exactly one system exposes a failure: RAGChecker or RAGVue not `healthy`; RAG Forensics not `healthy` or any `*_unavailable` signal other than the two it passes through from RAGAS; the RAGAS baseline when either RAGAS metric is not `ok` | 2 |
| `counterexample_to_preferred_interpretation` | RAG Forensics' flag contradicts the dataset label while every other flagged system agrees with the label | 2 |
| `intervention_discriminates_hypotheses` | for at least one eligible sentence, the grounding evaluator called it unsupported with selected evidence and supported with oracle evidence | 2 |
| `intervention_fails_to_localize` | for at least one eligible sentence, it stayed unsupported with oracle evidence | 2 |
| `component_diagnoses_disagree` | RAG Forensics' proposed test targets `retriever` or `answer generator`, and the RAGVue-derived direction differs. RAGVue's direction is generation when `strict_faithfulness` < 1.0, otherwise retrieval when `retrieval_relevance` < 0.5. RAGChecker is left out because its retriever metrics need a reference answer. | 3 |
| `evidence_attributions_disagree` | never: no system other than RAG Forensics attributes the answer to specific chunks, so there is nothing to compare across systems | 0 |
| `qualifier_negation_numerical_tabular_multisource_or_granularity` | the domain is `finqa`, or the response contains a negation or qualifier word (`not`, `no`, `never`, `without`, `except`, `only`, `unless`, `neither`, `nor`, `none`, or an `n't` contraction). This is a lexical proxy, not a judgment that the qualifier caused a failure. | 3 |
| `systems_agree_labels_contradict` | at least two systems have flags, they all agree, and they disagree with the dataset label | 3 |
| `systems_agree_labels_support` | as above, but they agree with the dataset label | 3 |

The two intervention strata test an intervention on the grounding *evaluator* (supplying annotated evidence), not on the RAG system that wrote the answer. See [the oracle-evidence explainer](../../../../docs/explainers/oracle-evidence.md).

## Selection

Strata are filled in the order of the table above, narrowest first, so broad strata can't use up the rare cases. Within a stratum, cases not already selected are sorted by case ID and drawn with `random.Random(30)`; a stratum with fewer eligible cases than its target takes all of them. Each case is selected into one stratum only. Targets sum to 20. Hand-picking is not used for any stratum, including the two the protocol allows it for.

The manifest records, for every stratum, the number of eligible cases, the number selected, and every exclusion with its reason, so purposive coverage can't be read as prevalence.
