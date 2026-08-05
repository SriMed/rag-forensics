# Installed RAGAS prompt contract audit

This record describes the dependency-owned prompts actually loaded by the repository's installed
RAGAS version. It is intentionally separate from the custom prompt runner: these prompts and their
parsers are implemented inside RAGAS, and their behavior can change when the dependency changes.

## Installed boundary

| Property | Installed value |
|---|---|
| RAGAS version | `0.4.3` |
| Project constraint | `ragas >=0.4.3,<0.5.0` |
| Model wrapper | `langchain_anthropic.ChatAnthropic` |
| Model identifier | `claude-haiku-4-5-20251001` |
| Project call site | `backend/services/ragas_scorer.py` |
| Dependency modules | `ragas.metrics._faithfulness`, `ragas.metrics._context_precision` |

The observations below come from direct inspection of the installed Python objects and source on
2026-08-03. They are installed-version evidence, not a guarantee about every version allowed by
the project's dependency range.

## Faithfulness

### Required inputs and processing

The installed `Faithfulness` metric requires `user_input`, `response`, and `retrieved_contexts`.
It uses two LLM stages:

1. `StatementGeneratorPrompt` receives the question and answer. Its instruction asks the model to
   break each answer sentence into fully understandable, pronoun-free statements and return JSON.
   The parsed output is `statements: list[str]`.
2. `NLIStatementPrompt` joins all retrieved contexts with newlines and asks for a binary verdict for
   every generated statement: `1` when directly inferable from the combined context and `0`
   otherwise. Each parsed item contains the statement, a reason, and an integer verdict.

The score is the fraction of generated statements whose verdict is truthy. No generated statements
produces `NaN` with a warning. The metric declares `max_retries = 1`; structured parsing and retry
behavior are dependency-owned.

### Interpretation and downstream influence

The project exposes this value as `ragas.faithfulness.score`. It is a model-judged continuous value, not
ground truth or a calibrated probability. It affects:

- the `entropy_faithfulness` trigger in `query_corpus_fit.py`;
- the deterministic `low_faithfulness` priority in `verdict_generator.py`;
- the numeric context supplied to the verdict-rendering prompt;
- the API's `ragas.faithfulness.score` field.

Because statement generation changes the units being judged, variation or decomposition errors in
the first LLM stage can change the final score even when the answer and evidence remain fixed.
Issue #18's mixed/null decomposition result and issue #19's residual oracle-evidence error are
direct reasons not to interpret this score as an isolated measurement of evidence selection.

## Context utilization

### Required inputs and processing

The project now imports the installed `context_utilization` object, a `ContextUtilization`
implementation built on `LLMContextPrecisionWithoutReference`. Its declared required inputs are
`user_input`, `retrieved_contexts`, and `response`. For each retrieved context independently,
`ContextPrecisionPrompt` asks whether that context was useful in arriving at the supplied answer.
It parses a `reason: str` and binary `verdict: int`, then computes rank-sensitive average precision
over the context verdicts.

The prompt's operative instruction is:

> Given question, answer and context verify if the context was useful in arriving at the given
> answer. Give verdict as 1 if useful and 0 if not with JSON output.

The metric declares `max_retries = 1`. Prompt formatting, structured-output parsing, multiple
generation, and verdict ensembling are dependency-owned.

### Project input contract

`score_context_utilization()` supplies the actual generated or caller-provided answer as `response`.
No reference sentinel is used. The supported interpretation is therefore answer-conditioned context
utilization: whether each retrieved context was useful for producing that answer, with rank-sensitive
average-precision aggregation. It does not independently establish question–context relevance or
retriever quality.

### Downstream influence

The project exposes the result as `ragas.context_utilization`. Its finite score affects:

- the unconditional `< 0.5` query-fit trigger;
- the deterministic `low_context_utilization` priority;
- the numeric context supplied to the verdict-rendering prompt;
- the API's `ragas.context_utilization.score` field.

Consequently, a dependency-owned prompt or input-contract mismatch can trigger an additional LLM
analysis and elevate a diagnostic concern. The score should remain labeled `model_judged` and must
not be described as a direct retriever measurement.

### Labeled comparison

On 2026-08-05, the issue #20 comparison runner evaluated four synthetic cases containing six
human-labeled relevant or irrelevant contexts through the Claude CLI `haiku` alias. The superseded
`reference="N/A"` configuration scored every case `0.0` and matched 3 of 6 context labels. The
selected context-utilization configuration matched all 6 labels and produced aggregate scores of
`1.0` for relevant-only, `0.0` for irrelevant-only, `1.0` for relevant-then-irrelevant, and `0.5`
for irrelevant-then-relevant. The mixed ordering confirms rank-sensitive aggregation on this run.

The cases, reviewed results, and runner are under
`backend/evals/context_utilization/v1/`. This small model-judged synthetic check validates the input
contract and expected contrasts; it is not a calibrated accuracy estimate or a general benchmark
claim.

## Failure and upgrade behavior

`backend/services/ragas_scorer.py` catches evaluation/conversion exceptions and rejects every
non-finite result. The API reports `status="unavailable"`, `score=null`, and either
`evaluation_failed` or `non_finite_score`; it never substitutes zero. Query-fit skips thresholds
whose score is unavailable, while verdict ranking emits an explicit unavailable signal. These
semantics apply to both context utilization and faithfulness.

The project permits any RAGAS release from 0.4.3 up to, but excluding, 0.5.0. Prompt text, examples,
Pydantic output models, retry behavior, aggregation, or which implementation the exported metric
name resolves to could therefore change after a dependency update without changes under
`backend/prompts/`.

For any RAGAS upgrade:

1. record the resolved package version;
2. inspect the concrete metric objects imported by `ragas_scorer.py`;
3. diff instructions, examples, input/output models, retries, and aggregation;
4. verify the project's supplied fields against the installed required columns;
5. rerun a frozen baseline before accepting score comparability.

## Supported conclusion

RAGAS contributes three dependency-owned LLM judgments to the two exposed scores: answer statement
generation, statement-level faithfulness judgment, and per-context usefulness judgment. These
scores materially influence downstream diagnostic ranking. The current retrieval-relevance call
also supplies `N/A` where the installed metric expects an answer/reference, so its interpretation is
not presently secure. No prompt-quality or accuracy rate is claimed without model runs and reviewed
outputs.
