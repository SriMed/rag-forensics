# Context-utilization labeled cases v1

These four small, hand-labeled cases test the interpretation selected in issue #20: whether each
retrieved context was useful for producing the supplied answer. `human_context_labels` are review
labels for context usefulness (`1`) or non-usefulness (`0`); they are not labels for the model's
numeric aggregate.

The cases cover relevant-only, irrelevant-only, and both orderings of a mixed pair because installed
RAGAS 0.4.3 computes rank-sensitive average precision. A conforming reviewed run should distinguish
the relevant-only case from the irrelevant-only case and score `relevant_then_irrelevant` above
`irrelevant_then_relevant`. Model-judged output remains variable and is not ground truth or a
calibrated probability.

The repository does not record invented live scores. The comparison runner uses the authenticated
Claude CLI and records the model alias, raw per-case scores, per-context verdicts, reasons, and
review placeholders.

Run the comparison from `backend/` with:

```bash
poetry run python evals/context_utilization/run_comparison.py
```

The runner prints both the superseded sentinel configuration and the selected configuration for
each case. Fill each `review` field when preserving an output artifact; do not infer review labels
from the scores themselves.

## Reviewed run — 2026-08-05

[`reviewed-results.json`](reviewed-results.json) records the Claude CLI `haiku` run. The sentinel
configuration scored all four cases `0.0` and matched 3 of 6 context labels because useful contexts
were rejected as not useful for the literal answer `N/A`. Context utilization matched all 6 labels:
relevant-only `1.0`, irrelevant-only `0.0`, relevant-first mixed `1.0`, and irrelevant-first mixed
`0.5`. The mixed-case difference confirms the installed rank-sensitive aggregation on this sample.

This is a reviewed four-case synthetic check, not a calibrated accuracy estimate or evidence that
the same agreement rate generalizes to production data.
