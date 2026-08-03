# Benchmarking and current evidence

RAG Forensics uses label-preserving evaluation. The benchmark runner analyzes the dataset’s
original question, documents, response, sentence boundaries, and labels. It does not retrieve a
new context or generate a replacement answer.

## RAGBench baseline

The baseline runner evaluates semantic attribution as an unsupported-sentence detector:

```bash
cd backend
poetry run python -m benchmark.cli \
  --domain techqa \
  --split test \
  --limit 100 \
  --seed 42 \
  --output output/ragbench-techqa.json
```

Reports preserve raw similarity, source candidates, sentence-support mappings, dataset
configuration, skipped rows, timestamps, and aggregate confusion metrics.

The first 100-example TechQA test run covered 946 sentences:

| Metric | Result |
|---|---:|
| Precision | 0.381 |
| Recall | 0.403 |
| F1 | 0.392 |
| AUROC | 0.563 |
| Coverage | 1.000 |

An in-sample threshold sweep mostly improved recall by predicting nearly every sentence as
unsupported. This established that the original `0.4` threshold was not a defensible standalone
hallucination boundary.

## B0–B3 experiment

The scientific runner compares:

- `b0_always_supported`;
- `b0_always_unsupported`;
- `b1_sentence_similarity`;
- `b2_claim_similarity`;
- `b3_claim_entailment`.

B2 and B3 use identical claims and evidence candidates. B3 preserves entailment, neutral, and
contradiction probabilities rather than collapsing all non-entailment outcomes.

RAGBench’s official validation split selects each threshold. The test split is evaluated with
those frozen thresholds. Calibration and test example IDs are checked for overlap.

```bash
cd backend
poetry run python -m benchmark.experiment_cli \
  --domains techqa finqa covidqa \
  --calibration-split validation \
  --evaluation-split test \
  --seed 42 \
  --bootstrap-iterations 2000 \
  --output output/ragbench-grounding.json
```

The report includes:

- per-domain and pooled confusion counts, precision, recall, F1, AUROC, AUPRC, prevalence, and
  coverage;
- example-clustered 95% confidence intervals;
- paired macro-F1 and macro-AUPRC intervals for B3 versus B1;
- claim-to-parent-sentence and evidence provenance;
- raw verifier scores, failures, configuration, immutable data/model revisions, and code state;
- false-positive and false-negative categories for numbers, negation, qualifiers, partial
  support, multi-source labels, and contradiction.

### Preliminary RAGBench result

A seeded run sampled up to 100 validation and 100 test examples per domain. It retained 299
records in each partition after one explicit skip and used 500 bootstrap iterations.

| Held-out macro metric | B1 | B2 | B3 |
|---|---:|---:|---:|
| F1 | 0.301 | 0.291 | 0.278 |
| AUPRC | 0.215 | 0.198 | 0.247 |

Paired B3−B1 results:

| Metric | Difference | 95% interval |
|---|---:|---:|
| Macro F1 | -0.022 | [-0.066, 0.025] |
| Macro AUPRC | 0.032 | [-0.016, 0.123] |

The predeclared success rule required B3 to improve both macro F1 and AUPRC, exclude no
improvement on the primary paired interval, and improve consistently across domains. This run
does not meet that rule. It is a mixed/null result, not evidence of superiority or inferiority.
It remains a sampled experiment rather than a final full-corpus estimate.

## RAGTruth external validation

RAGTruth uses character-level hallucination spans and heterogeneous source content. It does not
provide RAGBench-style gold source-sentence attribution. The adapter therefore:

- deterministically serializes source objects;
- labels a response sentence unsupported when it overlaps an annotated span;
- retains `implicit_true` spans as context-unsupported;
- excludes non-`good` responses;
- records malformed rows as explicit skips.

This evaluates sentence-level outcomes, not evidence-source selection.

```bash
poetry run python -m benchmark.ragtruth_cli \
  --source-info /path/to/source_info.jsonl \
  --responses /path/to/response.jsonl \
  --dataset-revision <official-ragtruth-commit> \
  --seed 42 \
  --bootstrap-iterations 2000 \
  --output output/ragtruth-grounding.json
```

A 25-train/25-test adapter-validation run at RAGTruth commit
`c103204b9ce28d6bbad859304bf30de72b8ed8fe` produced:

| Held-out macro metric | B1 | B3 |
|---|---:|---:|
| F1 | 0.093 | 0.256 |
| AUPRC | 0.295 | 0.211 |

B3 improved F1 but reduced AUPRC and was inconsistent across QA, summarization, and data-to-text.
Under the same decision rule this is mixed. The small sample and 100-bootstrap interval are
runtime validation, not final estimates.

## What these benchmarks establish

They establish that the evaluation pipeline preserves labels and provenance, prevents direct
calibration/test overlap, reports uncertainty, and can expose a null result.

They do not establish:

- that similarity or NLI proves factual grounding;
- that an unsupported response was caused by retrieval depth, chunking, generation, or corpus
  coverage;
- that the deterministic clause splitter produces correct atomic claims;
- that the sampled results generalize to the full datasets or production traffic.

## Oracle-evidence failure localization

The implemented diagnostic compares selected evidence with RAGBench’s annotated supporting
sentences. It is an analysis intervention, not a fourth deployable grounding method: labels supply
the oracle evidence, so its output is not test-time classifier performance.

The primary population is response sentences marked fully supported with at least one concrete
document-sentence key. Missing annotations and non-document support sentinels are excluded with
explicit reason counts. Unsupported sentences are outside the primary comparison because the
dataset does not identify a corresponding "correct negative evidence" sentence.

The paired conditions hold claim decomposition, NLI verifier and revision, and entailment
threshold fixed:

- **selected evidence:** B3's top cosine-similarity evidence for each claim;
- **oracle evidence:** every annotated sentence is scored for every claim, with maximum entailment
  used for the decision while all raw claim/evidence pairs remain in the report.

The primary outcome is the paired change in false-unsupported rate, with an example-clustered
confidence interval. The report also preserves selected-evidence hit-at-1, claim scores,
multi-source provenance, eligibility coverage, and verifier failures.

```bash
cd backend
poetry run python -m benchmark.oracle_evidence_cli \
  --domains techqa finqa covidqa \
  --evaluation-split test \
  --entailment-threshold <frozen-b3-threshold> \
  --seed 42 \
  --bootstrap-iterations 2000 \
  --output output/ragbench-oracle-evidence.json
```

A material reduction supports evidence selection as a bottleneck for eligible supported
sentences. Little or no reduction leaves verifier, decomposition, multi-sentence reasoning, and
annotation-granularity failures unresolved. Heterogeneous results argue against a global
bottleneck claim. The diagnostic cannot establish why unsupported sentences fail, causal
responsibility for the full RAG pipeline, or production performance. See
[Understanding the oracle-evidence experiment](../explainers/oracle-evidence.md) for a
plain-language explanation.
