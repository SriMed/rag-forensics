# Truncated evidence in answer generation

This reference records issue #25's paired evaluation of generation from complete versus visibly
truncated evidence. Its evidence boundary is narrow: three purposively selected pairs, four prompt
conditions, two repetitions, and a Codex CLI proxy model. It is not a production-model evaluation
or a failure-rate estimate.

## Decisive finding

The current evidence-only generation prompt did not handle all forms of truncation alike. Across
two repetitions per case, it extrapolated the repository-derived CovidQA fragment, abstained when
the FinQA value needed for arithmetic was absent, and repeated the incomplete TechQA fragment.
Therefore “truncated evidence” is distinct from ordinary insufficient context: a model can see
enough semantic direction to complete a thought even while missing the exact ending.

The hybrid alternative—terminal detection, explicit metadata, and a qualification instruction—made
truncation explicit in all six truncated outputs. It did not strictly prevent completion: both
CovidQA outputs still introduced `risk factor`, though they also stated that the supporting
sentence was truncated and limited what could be concluded. All 24 complete-passage outputs across
the four conditions remained useful on human review.

## Protocol and exact results

The versioned evaluation assets preserve the complete inputs, exact truncated inputs,
transformations, source URLs or local embedding IDs, raw responses, deterministic lexical labels,
and reviewer interpretation:

- [`cases.json`](../../backend/evals/truncated_evidence/v1/cases.json) — frozen inputs and provenance
- [`proxy-results.json`](../../backend/evals/truncated_evidence/v1/proxy-results.json) — 48 raw outputs
- [evaluation README](../../backend/evals/truncated_evidence/v1/README.md) — conditions, counts,
  review semantics, limitations, and reproduction command

The CovidQA complete counterpart comes from the original open-access paper; the stored Chroma
passage ends exactly after `may be a risk`. FinQA and TechQA truncations are deterministic prefixes
of exact repository-derived evidence already preserved by the prompt-audit dataset.

## What the alternatives establish

| Condition | Supported observation | Important limit |
| --- | --- | --- |
| Current prompt | Behavior varied across extrapolation, abstention, and fragment copying. | Six truncated proxy outputs do not measure reliability. |
| Explicit metadata | Metadata alone did not stop CovidQA extrapolation and was usually not surfaced. | The comparison supplied accurate metadata; the production data model currently does not. |
| Qualification instruction | Disclosure improved, but one CovidQA output still supplied the missing phrase. | Prompt compliance is not an enforcement boundary. |
| Deterministic hybrid | The simple detector separated all three constructed pairs and hybrid outputs disclosed truncation 6/6 times. | Terminal punctuation is a heuristic and can flag complete punctuation-free prose or tables. |

## Recommendation and update conditions

Do not broadly rewrite the generation prompt based on these three pairs. A focused follow-up should
preserve source-aware chunk-completeness metadata at ingestion, carry it through `RetrievedChunk`,
and give generation a bounded contract for incomplete evidence. The terminal detector is suitable
as a warning or fallback, not authoritative provenance. The follow-up should test the exact
production model and include detector-specificity cases before changing default behavior.
[Issue #27](https://github.com/SriMed/rag-forensics/issues/27) tracks that implementation work.

Revisit this conclusion if production-model comparison does not reproduce the CovidQA behavior,
if source-aware completeness cannot be recovered, or if a broader representative sample shows that
the disclosure tradeoff materially reduces usefulness on complete evidence.

## Implemented source-metadata contract

Issue [#27](https://github.com/SriMed/rag-forensics/issues/27) introduced explicit chunk metadata:

- `completeness`: `complete`, `truncated`, or `unknown`;
- `completeness_source`: `source`, `caller`, or `unavailable`.

`unknown` must pair with `unavailable`; known states must have source or caller provenance. Existing
Chroma records and custom requests without the new fields remain compatible and resolve to
`unknown`/`unavailable`. Malformed stored metadata also fails closed to that unavailable state.
Custom API clients may assert known completeness only with `caller` provenance.

RAGBench provides already-formed document strings without original source-boundary metadata.
Consequently, bootstrap records their completeness as unknown; terminal punctuation is never
promoted to provenance. The heuristic used in the investigation flags headings, punctuation-free
complete prose, scalar values, and serialized tables as possible truncation, demonstrating why it
is suitable only as a warning or fallback.

For source-known truncated chunks, the production prompt prohibits guessing the missing
continuation and requires disclosure when the missing text prevents a complete answer. Unknown
chunks are not described as truncated. The API returns `retrieved_chunk_details` so the state and
provenance remain inspectable after analysis.

### Exact production-model comparison

The [v2 reviewed run](../../backend/evals/truncated_evidence/v2/README.md) used the exact
`claude-haiku-4-5-20251001` model for 24 calls. Complete-evidence usefulness remained 6/6 in each
condition. Truncation disclosure improved from 4/6 with the pre-#27 prompt to 6/6 with the contract.
Strict avoidance of the held-back phrase remained 4/6: both CovidQA contract responses supplied
`risk factor`, but neither supplied the missing object and both disclosed that it was unavailable.

No response-level lexical rejection is implemented. The hidden source continuation is unavailable
at runtime, so a substring rule cannot reliably distinguish completion, paraphrase, negation, or a
supported phrase elsewhere in the evidence. Strict enforcement would require a source-aware
comparison boundary or a separately validated verifier.
