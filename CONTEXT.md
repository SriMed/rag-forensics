# RAG Forensics

Canonical language for discussing the project's diagnostic records and grounding evaluations.

## Language

**Claim-entailment evaluator**:
An offline grounding evaluator that decomposes a response sentence into claims, selects evidence
for each claim, scores entailment with a pinned verifier, and aggregates the claim judgments.
_Avoid_: B3, B3 method, claim-plus-NLI pipeline

**Claim-similarity evaluator**:
An offline grounding evaluator that decomposes a response sentence into claims and scores each
claim against similarity-selected evidence.
_Avoid_: B2, B2 method

**Whole-sentence similarity evaluator**:
An offline grounding evaluator that scores a complete response sentence against candidate evidence.
_Avoid_: B1, B1 method

**Oracle-evidence condition**:
A label-derived experimental condition that supplies annotated supporting evidence to an evaluator;
it is not a deployable grounding method.
_Avoid_: B4, oracle method
