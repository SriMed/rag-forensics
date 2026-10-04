# Understanding the comparative disagreement set

Several tools claim to diagnose what went wrong in a RAG answer. Issue #30 runs four of them on the same public cases and asks:

> Where do RAG Forensics, RAGChecker, RAGVUE, and a RAGAS baseline agree or disagree, and what does each disagreement look like up close?

The answer is a small, deliberately varied set of cases to inspect, not a scoreboard. The exact counts are in [Benchmarking and current evidence](../reference/benchmarks.md#comparative-disagreement-set), the rules in [`STRATUM-RULES.md`](../../backend/evals/comparative_diagnostics/v1/STRATUM-RULES.md), and the cases in [`selected-cases.json`](../../backend/evals/comparative_diagnostics/v1/selected-cases.json). This page explains the design calls behind them and how to read a case.

## Four systems, one shared question

The systems don't answer the same question by default. RAGChecker scores claims against a reference answer, RAGVUE scores retrieval and answer quality without one, RAGAS produces quality scores, and RAG Forensics ranks hypotheses and proposes a follow-up test. Comparing them first required finding what each could actually run on the same inputs without changing what its output means. The [feasibility check](../../backend/evals/comparative_diagnostics/v1/README.md) found:

- **RAGChecker** needs a reference answer for every metric except `faithfulness`. RAGBench has no reference answer separate from the response under evaluation, so only `faithfulness` is used. The other metrics are recorded as unavailable rather than computed against an invented answer.
- **RAGVUE** runs without a reference answer. Its output, however, misreports which judge model produced it, so the manifest records the model the run actually configured.
- **RAGAS** already runs inside RAG Forensics, so the baseline is read from that output rather than run twice.
- RAGChecker and RAGVUE pin conflicting versions of the Anthropic SDK, so each runs unmodified in its own environment ([ADR-044](../../ADR.md)).

The one question all four can answer is whether a response contains unsupported content. A system **flags** a case when at least one claim is unsupported, which matches how RAGBench labels a response: one unsupported sentence makes the whole response "contains unsupported".

RAG Forensics is built to keep several hypotheses visible rather than collapse them into one score, so its flag counts every hypothesis, not just the top-ranked one. It flags a case when its hedging detector finds a claim stated definitively but unsupported, or its attribution detector finds a sentence with no close source, at any rank. Its `low_faithfulness` signal is the RAGAS score re-ranked, so that one counts under the RAGAS baseline instead.

## A pool that only says "supported"

The cases come from a fixed candidate pool: the 98 RAGBench examples behind the supported sentences studied in the [oracle-evidence experiment](oracle-evidence.md). Reusing that pool keeps the intervention results available for every case. It also means **every case is labeled fully supported**.

That shapes what the set can show. When all systems agree with the label, they all said "supported". When they all agree against it, they all flagged a response the annotators accepted. The set contains no case where a system catches a response the dataset itself calls unsupported. Any flag here is a disagreement with the label, but not necessarily an error: annotations can be wrong, and the systems judge different things.

RAGBench also has no retrieval scores. Its records carry a placeholder score of `1.0` on every chunk, and RAG Forensics receives them with scores declared unavailable. Its score-distribution analysis is therefore switched off for this pool, and the comparison says nothing about it.

## How cases were chosen

#30 lists kinds of cases worth inspecting: agreement and disagreement, a failure only one system reports, interventions that do or don't settle a question, tricky wording, and counterexamples to RAG Forensics' own reading. Each kind became a **stratum** with a mechanical eligibility rule, written down before the outputs it was applied to were inspected.

Within each stratum, cases were drawn with a fixed random seed, starting from the narrowest strata so that broad ones couldn't use up the rare cases. Nobody hand-picked an illustrative example. The manifest records how many cases were eligible for each stratum next to how many were selected. Seeing 3 selected of 32 eligible stops anyone from reading the set's proportions as how often each situation occurs.

Two strata needed rules the issue didn't anticipate:

- **Component disagreement.** Only RAG Forensics names a failing component, through the target of its proposed test. RAGVUE's scores were turned into a direction: low faithfulness points to generation, and low retrieval relevance with a faithful answer points to retrieval. This is a derived reading of RAGVUE's scores, not something RAGVUE claims itself.
- **Evidence attribution.** No system except RAG Forensics ties the answer to specific chunks, so there is nothing to compare across systems. The stratum is recorded as infeasible instead of approximated.

The intervention strata use the oracle-evidence diagnostic: supplying annotated evidence to the grounding evaluator either turns its "unsupported" judgment into "supported" or doesn't. That intervenes on the evaluator, not on the RAG system that wrote the answer.

## Reading a case

Each case in the manifest keeps four kinds of information apart:

- **Native output**: what each system returned, unmodified, with its run state. `healthy`, `unavailable`, and `failed` are distinct, so an evaluator error can't pass for a low score.
- **Mapped fields**: the same output projected onto shared fields such as supporting observation, method, and reliability. A system that has no equivalent leaves the field empty and says why in `no_equivalent_fields`.
- **Judgments**: the dataset label, each system's flag, the oracle-evidence result, and the reviewer's interpretation, each in its own field. A dataset label is not a model judgment, and neither is a reviewer's conclusion.
- **Selection**: the stratum the case was drawn for, how many cases were eligible, and the other strata it also qualified for.

## What the pool showed

The pool-level counts are descriptive. RAG Forensics flagged 42 cases, RAGAS 40, RAGVUE 39, and RAGChecker 30. 32 cases were flagged by no system and 11 by all four. Agreement between any two systems falls between 63 and 70 of 98, so no pair stands apart.

The structural patterns say more than the counts:

- **RAG Forensics mostly proposes testing a boundary, not blaming a component.** Its proposed test targeted the retrieval-to-generation boundary in 76 cases and named a single component in only 14. RAGVUE's scores, read as a direction, point one way in 60 cases. That is a difference in what the tools claim: RAG Forensics keeps the competing explanations open where a score commits to one. It's also why component disagreement has only 2 eligible cases.
- **Only RAG Forensics reports its own failures.** In 11 cases its retrieved-context fit check couldn't generate enough valid questions and said so. The other systems returned no failure states on any case, which can mean they never failed or that a failure would look like an ordinary score.
- **Eleven responses were flagged by every system** although the annotators labeled them supported. Those are the cases most worth reading by hand, because either the annotation or all four judges are wrong.

## What the design got wrong

The rules went through three versions, and every superseded draw is recorded in the rules file.

- **The pool's labels weren't checked before the first rules were frozen.** The v1 counterexample rule needed RAG Forensics to contradict the label while every other system agreed with it, and no case qualified. It was relaxed after the aggregate counts were seen ([ADR-057](../../ADR.md)). This repeats a lesson from the [joint-evidence experiment](joint-evidence.md#what-the-design-got-wrong): check what a population can produce before freezing rules over it.
- **The first run fed RAG Forensics fake retrieval scores.** RAGBench's placeholders reached it as real similarity scores, and their perfectly flat distribution made "ambiguous retrieval" its top hypothesis in 75 of 98 cases. The API now accepts declared-unavailable scores (issue #35), and RAG Forensics was rerun.
- **The first flag rule measured ranking, not detection.** Counting only RAG Forensics' top-ranked signal suited a scoring tool, not one built around several hypotheses. Combined with the fake scores, it made RAG Forensics look as if it rarely flagged anything.
- **The first run also exposed a defect.** The hedging check had discarded about a quarter of its "not supported" verdicts because the model attached explanations to them. That was fixed before the rerun (issue #34), and none of the rerun's 631 checks was malformed.
- **The set leans heavily on one domain.** The pool is 87 CovidQA examples, so tabular and numerical wording is barely represented; the draw contains one FinQA case.

Re-selecting after fixing defects is a judgment call. The fixes were defined by the contracts the defects broke rather than by how the comparison read, and the [decision records](../../ADR.md) (ADR-056 to ADR-060) keep each version's reasoning.

## What this collection does not establish

It doesn't rank the systems, estimate how often any failure occurs, or show that any system diagnoses more accurately or helps developers more. Those claims need the separate evaluation in issue #31, with cases where the true cause is known. The collection supports something narrower: inspecting, one case at a time, how differently built diagnostic tools describe the same answer, and where their constructs fail to line up.
