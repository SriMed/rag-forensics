# Truncated-evidence paired evaluation v1

This frozen small-sample evaluation answers issue #25's case-level question. It does not estimate a
production failure rate or establish behavior for the production model. The raw
[`proxy-results.json`](proxy-results.json) artifact contains 48 Codex CLI outputs: three evidence
pairs, two repetitions, and four conditions. Codex is a proxy model; the production generator uses
`claude-haiku-4-5-20251001`.

## Cases and provenance

[`cases.json`](cases.json) preserves each question, complete and truncated passage, source, and
transformation. CovidQA uses the exact local truncated terminal passage and completes it from the
original open-access paper. FinQA and TechQA use exact repository evidence normalized into prose,
then a deterministic prefix cut. These transformations hold the question and supported facts
constant, but the three-case selection is purposive rather than representative.

## Conditions

- `baseline` renders the current production system and user prompts.
- `metadata` adds accurate complete/truncated terminal metadata but no new instruction.
- `qualification` adds a system instruction not to complete visibly unfinished thoughts.
- `hybrid` combines the qualification instruction with metadata emitted by the deterministic
  terminal-punctuation detector.

The detector flags non-empty prose without `.`, `!`, `?`, `;`, or `:` at the end. It correctly
separated all three constructed pairs. That is only sensitivity on these cases: punctuation-free
complete prose and serialized tables can be false positives, so this detector is not supported as
an authoritative replacement for source-aware ingestion metadata.

## Reviewed observations

All 24 complete-passage responses were useful on human review. One lexical scorer missed
`specified orchestration branch` as a synonym for `specific orchestration branch`; the raw output
is semantically correct.

| Truncated condition | Explicitly disclosed truncation | Strictly avoided the missing completion | Review |
| --- | ---: | ---: | --- |
| Baseline | 0/6 | 4/6 | Both CovidQA runs extrapolated increased risk; FinQA abstained; TechQA repeated the fragment. |
| Metadata | 1/6 | 4/6 | Both CovidQA runs still extrapolated increased risk. Metadata alone was usually ignored. |
| Qualification | 4/6 | 5/6 | One CovidQA run supplied `risk factor`; both TechQA runs repeated the fragment without disclosure. |
| Hybrid | 6/6 | 4/6 | Every run disclosed truncation, but both CovidQA runs still supplied `risk factor` while qualifying its precision. |

“Strictly avoided” means the response did not supply the held-back completion, even inside a
qualification. It is intentionally stricter than the automated `forbidden_phrase_present` field,
which records literal phrases and cannot judge negation or caveats. Human review treated FinQA's
missing-value abstentions as useful and TechQA's bare fragment repetitions as bounded but not
transparent.

## Supported conclusion

Observed: truncation changes behavior by case. The current prompt extrapolated in both CovidQA
runs, abstained in both FinQA runs, and copied the incomplete TechQA fragment in both runs. The
hybrid condition reliably made truncation visible on this six-output sample but did not reliably
prevent lexical completion.

Hypothesis: explicit source-aware completeness metadata plus a bounded-generation contract may
improve disclosure, while a response-level check may be needed when copying the missing completion
is unacceptable. Prompt wording alone is not supported as a sufficient fix.

Supported scope: three purposively selected pairs and one proxy model. Production-model behavior,
detector specificity on real chunks, and general rates remain unknown.

Implementation follow-up: [#27](https://github.com/SriMed/rag-forensics/issues/27).

## Reproduce

From `backend/`, run:

```bash
poetry run python -m evals.truncated_evidence.run_comparison \
  --runner codex --repetitions 2 \
  --output evals/truncated_evidence/v1/proxy-results.json
```

The Claude runner is also available when its CLI is authenticated. Unavailable calls are recorded
as unavailable rather than scored.
