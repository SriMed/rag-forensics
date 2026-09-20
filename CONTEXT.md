# RAG Forensics

Canonical language for discussing the project's diagnostic records and grounding evaluations.

## Language

**Response sentence**:
An un-decomposed unit of the model's answer, before claim splitting. Each response sentence in a
factorial or oracle-evidence report is scored supported only when every claim it decomposes into
passes verification.

**Claim**:
The smallest independently verifiable unit a response sentence is decomposed into. A claim must be
checkable on its own — handed to a verifier with no other sentence, no surrounding paragraph, and
one candidate evidence sentence, it must state something concrete enough to judge. A dangling
pronoun, a sentence fragment missing its subject, or a bare list entry that only means something
next to its list header all fail this test.
_Avoid_: "clause" as a synonym once decomposition review has run (a clause is what the
deterministic splitter proposes; a claim is what survives review)

**Evidence**:
A candidate source sentence a claim is checked against, drawn from the retrieved context (or,
under an oracle-evidence condition, from RAGBench's annotated supporting sentences). Evidence never
comes from the response itself — it is always the retrieved or annotated material the response was
supposed to be grounded in.

**Aggregation exposure**:
How many independently-scored claims a response sentence's supported/unsupported verdict depends
on. Because a sentence is marked supported only when every one of its claims passes, raising claim
count lowers the odds of an all-pass outcome for reasons unrelated to whether the sentence is
actually well grounded — real partial-coverage gaps and ordinary verifier/retrieval noise both
compound with claim count. A sentence's aggregation exposure changing between two decompositions is
not, by itself, evidence that one decomposition is more correct than the other.
_Avoid_: reading a lower supported rate as proof of worse grounding without checking claim count

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

**Retrieved-context fit**:
A conditional observation of what questions the retrieved passages appear able to answer, labeled a
near miss, topic gap, or ambiguous. It describes only the retrieved passages and never establishes
whether the full corpus can answer the query. The code and API name is `query_corpus_fit`.
_Avoid_: "corpus fit" or "corpus coverage" as a description of what the result establishes
