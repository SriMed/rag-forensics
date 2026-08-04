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

The project exposes this value as `faithfulness_score`. It is a model-judged continuous value, not
ground truth or a calibrated probability. It affects:

- the `entropy_faithfulness` trigger in `query_corpus_fit.py`;
- the deterministic `low_faithfulness` priority in `verdict_generator.py`;
- the numeric context supplied to the verdict-rendering prompt;
- the API's `ragas.faithfulness_score` field.

Because statement generation changes the units being judged, variation or decomposition errors in
the first LLM stage can change the final score even when the answer and evidence remain fixed.
Issue #18's mixed/null decomposition result and issue #19's residual oracle-evidence error are
direct reasons not to interpret this score as an isolated measurement of evidence selection.

## Context precision used as retrieval relevance

### Required inputs and processing

The imported `context_precision` object is the installed `ContextPrecision` implementation built on
`LLMContextPrecisionWithReference`. Its declared required inputs are `user_input`,
`retrieved_contexts`, and `reference`. For each retrieved context independently,
`ContextPrecisionPrompt` asks whether that context was useful in arriving at the supplied answer.
It parses a `reason: str` and binary `verdict: int`, then computes rank-sensitive average precision
over the context verdicts.

The prompt's operative instruction is:

> Given question, answer and context verify if the context was useful in arriving at the given
> answer. Give verdict as 1 if useful and 0 if not with JSON output.

The metric declares `max_retries = 1`. Prompt formatting, structured-output parsing, multiple
generation, and verdict ensembling are dependency-owned.

### Project input mismatch

`score_retrieval_relevance()` currently supplies `reference="N/A"`. In the installed metric,
`reference` becomes the prompt's `answer`. Therefore the model is asked whether each context was
useful for arriving at the literal answer `N/A`, rather than the generated answer or a benchmark
reference answer.

This is a direct installed-code finding, not a sampled-model result. It makes the construct measured
by the current call unclear and is a plausible explanation for the zero retrieval-relevance scores
recorded in issue #9's smoke test. A follow-up should compare an appropriate reference-aware metric
configuration or a reference-free retrieval metric before interpreting this field as retrieval
relevance. This audit does not change production behavior.

### Downstream influence

The project exposes the result as `retrieval_relevance_score`. It affects:

- the unconditional `< 0.5` query-fit trigger;
- the deterministic `low_retrieval_relevance` priority;
- the numeric context supplied to the verdict-rendering prompt;
- the API's `ragas.retrieval_relevance_score` field.

Consequently, a dependency-owned prompt or input-contract mismatch can trigger an additional LLM
analysis and elevate a diagnostic concern. The score should remain labeled `model_judged` and must
not be described as a direct retriever measurement.

## Failure and upgrade behavior

`backend/services/ragas_scorer.py` calls `ragas.evaluate()` and directly converts the returned first
metric value with `float(...)`. It does not define a local parse fallback or explicit unavailable
state. Exceptions propagate to the request-level caller; `NaN` can also originate from the installed
faithfulness metric when no statements are generated.

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
