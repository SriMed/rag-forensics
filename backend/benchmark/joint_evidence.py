"""Joint versus separate evidence verification (issue #32).

Implements the frozen protocol in ``evals/joint_evidence/v1-techqa/protocol.md``.  A joint
premise reaches the unchanged verifier as one evidence candidate whose key joins the evidence
sentence keys with ``+``.  Inputs over the token budget are flagged, never truncated.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from benchmark.decomposition_evidence import validate_claim_review
from benchmark.grounding import ClaimDecomposer, EntailmentVerifier
from models import (
    AtomicClaim,
    ConfidenceInterval,
    EvidenceCandidate,
    NLIVerifierScores,
    RAGBenchEvaluationRecord,
)

TOKEN_BUDGET = 512
WINDOW = 1
TOP_K = 3
DISTRACTOR_EXCLUSION_RADIUS = 2
LENIENCY_MARGIN = 0.05
SIGNIFICANCE = 0.05
MAX_SIGN_FLIP_CLUSTERS = 20

TokenCounter = Callable[[str, str], int]
UnavailableReason = Literal[
    "over_budget", "verifier_error", "control_unavailable", "no_claims", "no_evidence"
]
Label = Literal["confounded_by_leniency", "supported", "inconclusive"]
Outcome = tuple[bool | None, str | None]


class JointPremise(BaseModel):
    keys: list[str]
    text: str
    document_ids: list[str]

    @property
    def evidence_key(self) -> str:
        return "+".join(self.keys)


class JointClaimVerification(BaseModel):
    claim: AtomicClaim
    evidence_keys: list[str]
    premise: str
    token_count: int
    support_score: float | None
    predicted_supported: bool | None
    status: Literal["ok", "over_budget", "verifier_error"]
    nli_scores: NLIVerifierScores | None = None
    error: str | None = None


class SentenceOutcome(BaseModel):
    example_id: str
    domain: str
    sentence_key: str
    sentence: str
    gold_unsupported: bool
    predicted_unsupported: bool | None
    unavailable_reason: UnavailableReason | None
    claims: list[JointClaimVerification]
    separate_pairs: list[JointClaimVerification] = Field(default_factory=list)


class TransitionCounts(BaseModel):
    corrected: int
    introduced: int
    unchanged: int
    unavailable: dict[str, int]


class ContrastResult(BaseModel):
    population: Literal["supported", "unsupported"]
    condition: str
    reference: str
    transitions: TransitionCounts
    interval: ConfidenceInterval | None
    sign_flip_p: float | None = None
    per_example_difference: dict[str, int]


class PopulationSummary(BaseModel):
    supported_sentences: int
    supported_examples: int
    unsupported_sentences: int
    unsupported_examples: int


class JointEvidenceMetadata(BaseModel):
    protocol: str
    protocol_sha256: str
    dataset: str
    dataset_revision: str
    evaluation_split: str
    sample_limit_per_domain: int | None
    domains: list[str]
    seed: int
    bootstrap_iterations: int
    embedding_model: str
    embedding_model_revision: str
    verifier_model: str
    verifier_revision: str
    entailment_threshold: float
    aggregation: Literal["all_claims_supported"] = "all_claims_supported"
    window: int = WINDOW
    top_k: int = TOP_K
    token_budget: int = TOKEN_BUDGET
    deterministic_decomposer: str
    deterministic_decomposer_version: str
    residual_review_sha256: str | None
    claim_review_revision: int
    reviewer_identity: str


class JointEvidenceReport(BaseModel):
    schema_version: Literal["joint-evidence-report.v1"] = "joint-evidence-report.v1"
    metadata: JointEvidenceMetadata | None = None
    claim_review_sha256: str
    population: PopulationSummary
    supported: dict[str, list[SentenceOutcome]]
    unsupported: dict[str, list[SentenceOutcome]]
    contrasts: dict[str, ContrastResult]
    subgroup_keys: list[str]
    subgroup: dict[str, TransitionCounts]
    labels: dict[str, Label]


# --- evidence construction -------------------------------------------------


def _positions(record: RAGBenchEvaluationRecord) -> dict[str, int]:
    return {sentence.key: index for index, sentence in enumerate(record.document_sentences)}


def window_keys(
    record: RAGBenchEvaluationRecord, anchor_keys: Sequence[str], width: int = WINDOW,
) -> list[str]:
    """Anchors plus ``width`` neighbors each side, merged, within the anchor's document."""
    sentences = record.document_sentences
    positions = _positions(record)
    selected = set()
    for key in anchor_keys:
        anchor = positions[key]
        for index in range(anchor - width, anchor + width + 1):
            if 0 <= index < len(sentences) and sentences[index].document_id == sentences[anchor].document_id:
                selected.add(index)
    return [sentences[index].key for index in sorted(selected)]


def _distractor_seed(seed: int, record: RAGBenchEvaluationRecord, sentence_key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{record.domain}:{record.example_id}:{sentence_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def distractor_keys(
    record: RAGBenchEvaluationRecord, sentence_key: str, n: int, seed: int,
) -> list[str] | None:
    """Seeded same-document distractors, or None when fewer than ``n`` are eligible."""
    sentences = record.document_sentences
    positions = _positions(record)
    annotated = record.sentence_support[sentence_key].supporting_sentence_keys
    any_annotated = {
        key for support in record.sentence_support.values() for key in support.supporting_sentence_keys
    }
    documents = {sentences[positions[key]].document_id for key in annotated}
    nearby = {
        index
        for key in annotated
        for index in range(
            positions[key] - DISTRACTOR_EXCLUSION_RADIUS, positions[key] + DISTRACTOR_EXCLUSION_RADIUS + 1
        )
    }
    eligible = [
        sentence.key
        for index, sentence in enumerate(sentences)
        if sentence.document_id in documents
        and sentence.key not in any_annotated
        and index not in nearby
        and sentence.text.strip()
    ]
    if len(eligible) < n:
        return None
    if n == 0:
        return []
    rng = np.random.default_rng(_distractor_seed(seed, record, sentence_key))
    drawn = rng.choice(eligible, size=n, replace=False).tolist()
    return sorted(drawn, key=eligible.index)


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest scores; ties go to the earlier document position."""
    return [int(index) for index in np.argsort(-scores, kind="stable")[:k]]


def build_premise(record: RAGBenchEvaluationRecord, keys: Sequence[str]) -> JointPremise:
    positions = _positions(record)
    ordered = sorted(set(keys), key=positions.__getitem__)
    sentences = [record.document_sentences[positions[key]] for key in ordered]
    document_ids = list(dict.fromkeys(sentence.document_id for sentence in sentences))
    return JointPremise(
        keys=ordered, text=" ".join(sentence.text for sentence in sentences), document_ids=document_ids,
    )


# --- verification and aggregation -------------------------------------------


def verify_joint(
    verifier: EntailmentVerifier, count_tokens: TokenCounter, claim: AtomicClaim,
    premise: JointPremise, threshold: float, budget: int = TOKEN_BUDGET,
) -> JointClaimVerification:
    """Score one (premise, claim) pair; over-budget pairs never reach the verifier."""
    tokens = count_tokens(premise.text, claim.text)

    def result(
        status: Literal["ok", "over_budget", "verifier_error"],
        scores: NLIVerifierScores | None = None, error: str | None = None,
    ) -> JointClaimVerification:
        return JointClaimVerification(
            claim=claim, evidence_keys=premise.keys, premise=premise.text, token_count=tokens,
            support_score=scores.entailment if scores else None,
            predicted_supported=scores.entailment >= threshold if scores else None,
            status=status, nli_scores=scores, error=error,
        )

    if tokens > budget:
        return result("over_budget")
    candidate = EvidenceCandidate(
        sentence_key=premise.evidence_key, document_id="+".join(premise.document_ids),
        text=premise.text, selection_score=1.0,
    )
    try:
        return result("ok", verifier.score(claim, candidate))
    except Exception as exc:
        return result("verifier_error", error=str(exc))


def verify_separate(
    verifier: EntailmentVerifier, count_tokens: TokenCounter, claim: AtomicClaim,
    record: RAGBenchEvaluationRecord, keys: Sequence[str], threshold: float, budget: int = TOKEN_BUDGET,
) -> tuple[JointClaimVerification, list[JointClaimVerification]]:
    """Condition A: score each sentence alone and keep the best usable pair (the #29 oracle)."""
    pairs = [
        verify_joint(verifier, count_tokens, claim, build_premise(record, [key]), threshold, budget)
        for key in keys
    ]
    over = [pair for pair in pairs if pair.status == "over_budget"]
    if over:
        return over[0], pairs
    usable = [pair for pair in pairs if pair.support_score is not None]
    if not usable:
        return pairs[0], pairs
    return max(usable, key=lambda pair: cast(float, pair.support_score)), pairs


def aggregate_sentence(verifications: Sequence[JointClaimVerification]) -> Outcome:
    """Return (predicted_unsupported, unavailable_reason) under all-claims-supported."""
    if not verifications:
        return None, "no_claims"
    if any(item.status == "over_budget" for item in verifications):
        return None, "over_budget"
    if any(item.status == "verifier_error" for item in verifications):
        return None, "verifier_error"
    return not all(item.predicted_supported for item in verifications), None


# --- analysis ----------------------------------------------------------------


def _is_error(predicted_unsupported: bool, error_when_unsupported: bool) -> bool:
    return predicted_unsupported if error_when_unsupported else not predicted_unsupported


def transition_counts(
    reference: Sequence[Outcome], condition: Sequence[Outcome], error_when_unsupported: bool,
) -> TransitionCounts:
    corrected = introduced = unchanged = 0
    unavailable: dict[str, int] = {}
    for (ref, ref_reason), (cond, cond_reason) in zip(reference, condition, strict=True):
        if ref is None or cond is None:
            reason = (cond_reason if cond is None else ref_reason) or "unavailable"
            unavailable[reason] = unavailable.get(reason, 0) + 1
            continue
        before, after = _is_error(ref, error_when_unsupported), _is_error(cond, error_when_unsupported)
        if before and not after:
            corrected += 1
        elif after and not before:
            introduced += 1
        else:
            unchanged += 1
    return TransitionCounts(corrected=corrected, introduced=introduced, unchanged=unchanged, unavailable=unavailable)


def paired_differences(
    clusters: Sequence[str], reference: Sequence[Outcome], condition: Sequence[Outcome],
    error_when_unsupported: bool,
) -> dict[str, list[int]]:
    """Per-example lists of (condition error - reference error) over paired available sentences."""
    differences: dict[str, list[int]] = {}
    for cluster, (ref, _), (cond, _) in zip(clusters, reference, condition, strict=True):
        if ref is None or cond is None:
            continue
        differences.setdefault(cluster, []).append(
            int(_is_error(cond, error_when_unsupported)) - int(_is_error(ref, error_when_unsupported))
        )
    return differences


def sign_flip_p_value(cluster_differences: Mapping[str, Sequence[int]]) -> float | None:
    """Exact two-sided cluster sign-flip test of the mean paired difference."""
    totals = [sum(values) for values in cluster_differences.values()]
    count = sum(len(values) for values in cluster_differences.values())
    if not totals or count == 0:
        return None
    if len(totals) > MAX_SIGN_FLIP_CLUSTERS:
        raise ValueError("exact sign-flip enumeration is limited to 20 clusters")
    observed = abs(sum(totals)) / count
    extreme = sum(
        abs(sum(sign * total for sign, total in zip(signs, totals))) / count >= observed - 1e-12
        for signs in itertools.product((1, -1), repeat=len(totals))
    )
    return extreme / 2 ** len(totals)


def clustered_difference_interval(
    cluster_differences: Mapping[str, Sequence[int]], iterations: int, seed: int,
) -> ConfidenceInterval | None:
    """Percentile bootstrap of the mean paired difference, resampling examples."""
    clusters = sorted(name for name, values in cluster_differences.items() if values)
    if not clusters:
        return None

    def statistic(sampled: Sequence[str]) -> float:
        return float(np.mean([value for name in sampled for value in cluster_differences[name]]))

    rng = np.random.default_rng(seed)
    samples = [
        statistic([clusters[int(i)] for i in rng.integers(0, len(clusters), len(clusters))])
        for _ in range(iterations)
    ]
    return ConfidenceInterval(
        point_estimate=statistic(clusters), lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)), iterations=iterations, seed=seed,
    )


def label_result(
    primary: TransitionCounts, primary_p: float | None, leniency: ConfidenceInterval | None,
    size_control: TransitionCounts | None,
) -> Label:
    """Apply the frozen interpretation rules in order."""
    if leniency is None:
        return "inconclusive"
    if leniency.point_estimate > LENIENCY_MARGIN or leniency.lower > 0:
        return "confounded_by_leniency"
    primary_holds = (
        primary.corrected > primary.introduced and primary_p is not None and primary_p < SIGNIFICANCE
    )
    control_holds = size_control is None or size_control.corrected > size_control.introduced
    return "supported" if primary_holds and control_holds else "inconclusive"


# --- runner ------------------------------------------------------------------

SUPPORTED_CONDITIONS = ("A", "B", "B+", "S", "T1", "C")
UNSUPPORTED_CONDITIONS = ("T1", "C", "W")
SUPPORTED_CONTRASTS = {
    "bplus_vs_a": ("B+", "A", True),
    "bplus_vs_s": ("B+", "S", True),
    "c_vs_t1": ("C", "T1", True),
    "b_vs_a": ("B", "A", False),
}
UNSUPPORTED_CONTRASTS = {
    "w_vs_t1_unsupported": ("W", "T1"),
    "c_vs_t1_unsupported": ("C", "T1"),
}


class _Context:
    def __init__(self, verifier, count_tokens, threshold, budget):
        self.verifier = verifier
        self.count_tokens = count_tokens
        self.threshold = threshold
        self.budget = budget

    def joint(self, claim, record, keys):
        return verify_joint(
            self.verifier, self.count_tokens, claim, build_premise(record, keys), self.threshold, self.budget,
        )


def _outcome(record, sentence, gold, claims, reason=None, separate_pairs=()):
    predicted, aggregated_reason = (None, reason) if reason else aggregate_sentence(claims)
    return SentenceOutcome(
        example_id=record.example_id, domain=record.domain, sentence_key=sentence.key,
        sentence=sentence.text, gold_unsupported=gold, predicted_unsupported=predicted,
        unavailable_reason=cast(UnavailableReason | None, aggregated_reason), claims=list(claims),
        separate_pairs=list(separate_pairs),
    )


def _claim_scores(embedding_model, record, claims: Sequence[AtomicClaim]) -> np.ndarray:
    documents = np.asarray(embedding_model.encode([s.text for s in record.document_sentences]), dtype=float)
    vectors = np.asarray(embedding_model.encode([claim.text for claim in claims]), dtype=float)
    return cosine_similarity(vectors, documents)


def _selected_keys(record, scores: np.ndarray, k: int) -> list[str]:
    return [record.document_sentences[index].key for index in top_k_indices(scores, k)]


def _supported_outcomes(record, sentence, claims, scores, context, seed):
    annotated = record.sentence_support[sentence.key].supporting_sentence_keys
    rows: dict[str, SentenceOutcome] = {}
    best_pairs = [
        verify_separate(context.verifier, context.count_tokens, claim, record, annotated,
                        context.threshold, context.budget)
        for claim in claims
    ]
    rows["A"] = _outcome(record, sentence, False, [best for best, _ in best_pairs],
                         separate_pairs=[pair for _, pairs in best_pairs for pair in pairs])
    rows["B"] = _outcome(record, sentence, False, [context.joint(c, record, annotated) for c in claims])
    window = window_keys(record, annotated)
    rows["B+"] = _outcome(record, sentence, False, [context.joint(c, record, window) for c in claims])
    distractors = distractor_keys(record, sentence.key, len(window) - len(set(annotated)), seed)
    rows["S"] = (
        _outcome(record, sentence, False, [], reason="control_unavailable")
        if distractors is None
        else _outcome(record, sentence, False,
                      [context.joint(c, record, [*annotated, *distractors]) for c in claims])
    )
    rows["T1"] = _outcome(record, sentence, False, [
        context.joint(c, record, _selected_keys(record, scores[i], 1)) for i, c in enumerate(claims)
    ])
    rows["C"] = _outcome(record, sentence, False, [
        context.joint(c, record, _selected_keys(record, scores[i], TOP_K)) for i, c in enumerate(claims)
    ])
    return rows


def _unsupported_outcomes(record, sentence, claims, scores, context):
    if not record.document_sentences:
        return {name: _outcome(record, sentence, True, [], reason="no_evidence") for name in UNSUPPORTED_CONDITIONS}
    top1 = [_selected_keys(record, scores[i], 1) for i in range(len(claims))]
    return {
        "T1": _outcome(record, sentence, True, [context.joint(c, record, top1[i]) for i, c in enumerate(claims)]),
        "C": _outcome(record, sentence, True, [
            context.joint(c, record, _selected_keys(record, scores[i], TOP_K)) for i, c in enumerate(claims)
        ]),
        "W": _outcome(record, sentence, True, [
            context.joint(c, record, window_keys(record, top1[i])) for i, c in enumerate(claims)
        ]),
    }


def _pairs(rows: Sequence[SentenceOutcome]) -> list[Outcome]:
    return [(row.predicted_unsupported, row.unavailable_reason) for row in rows]


def _contrast(population, condition, reference, rows, error_when_unsupported, iterations, seed, with_test):
    clusters = [f"{row.domain}:{row.example_id}" for row in rows[reference]]
    ref, cond = _pairs(rows[reference]), _pairs(rows[condition])
    differences = paired_differences(clusters, ref, cond, error_when_unsupported)
    return ContrastResult(
        population=population, condition=condition, reference=reference,
        transitions=transition_counts(ref, cond, error_when_unsupported),
        interval=clustered_difference_interval(differences, iterations, seed),
        sign_flip_p=sign_flip_p_value(differences) if with_test else None,
        per_example_difference={name: sum(values) for name, values in differences.items()},
    )


def run_joint_evidence_experiment(
    records: Sequence[RAGBenchEvaluationRecord], embedding_model,
    deterministic_decomposer: ClaimDecomposer, verifier: EntailmentVerifier,
    count_tokens: TokenCounter, threshold: float, review, review_sha256: str,
    subgroup_keys: Collection[str] = (), bootstrap_iterations: int = 2000, seed: int = 42,
    expected_population: tuple[int, int] | None = None,
    metadata: JointEvidenceMetadata | None = None, token_budget: int = TOKEN_BUDGET,
) -> JointEvidenceReport:
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1")
    eligible, reviewed = validate_claim_review(records, review, deterministic_decomposer)
    unsupported_sentences = [
        (record, sentence)
        for record in records
        for sentence in record.response_sentences
        if sentence.key in record.unsupported_response_sentence_keys
    ]
    supported_count = sum(len(record.response_sentences) for record in eligible)
    if expected_population is not None and expected_population != (supported_count, len(unsupported_sentences)):
        raise ValueError(
            f"population counts {(supported_count, len(unsupported_sentences))} do not match "
            f"the frozen protocol {expected_population}"
        )
    context = _Context(verifier, count_tokens, threshold, token_budget)

    supported: dict[str, list[SentenceOutcome]] = {name: [] for name in SUPPORTED_CONDITIONS}
    for record in eligible:
        claims_by_sentence = []
        for sentence in record.response_sentences:
            item = reviewed[(record.domain, record.example_id, sentence.key)]
            texts = item.deterministic_claims if item.accept_deterministic else item.reviewed_claims
            claims_by_sentence.append([
                AtomicClaim(claim_id=f"{sentence.key}.claim-{i}", parent_sentence_key=sentence.key, text=text)
                for i, text in enumerate(texts)
            ])
        scores = _claim_scores(embedding_model, record, [c for claims in claims_by_sentence for c in claims])
        offset = 0
        for sentence, claims in zip(record.response_sentences, claims_by_sentence):
            rows = _supported_outcomes(record, sentence, claims, scores[offset:offset + len(claims)], context, seed)
            offset += len(claims)
            for name, row in rows.items():
                supported[name].append(row)

    unsupported: dict[str, list[SentenceOutcome]] = {name: [] for name in UNSUPPORTED_CONDITIONS}
    by_record: dict[tuple[str, str], list] = {}
    for record, sentence in unsupported_sentences:
        by_record.setdefault((record.domain, record.example_id), []).append((record, sentence))
    for pairs in by_record.values():
        record = pairs[0][0]
        claims_by_sentence = [deterministic_decomposer.decompose(s.key, s.text) for _, s in pairs]
        flat = [c for claims in claims_by_sentence for c in claims]
        scores = _claim_scores(embedding_model, record, flat) if flat and record.document_sentences else np.zeros((0, 0))
        offset = 0
        for (_, sentence), claims in zip(pairs, claims_by_sentence):
            rows = _unsupported_outcomes(record, sentence, claims, scores[offset:offset + len(claims)], context)
            offset += len(claims)
            for name, row in rows.items():
                unsupported[name].append(row)

    contrasts = {
        name: _contrast("supported", cond, ref, supported, True, bootstrap_iterations, seed, with_test)
        for name, (cond, ref, with_test) in SUPPORTED_CONTRASTS.items()
    }
    contrasts.update({
        name: _contrast("unsupported", cond, ref, unsupported, False, bootstrap_iterations, seed, False)
        for name, (cond, ref) in UNSUPPORTED_CONTRASTS.items()
    })

    subgroup_set = set(subgroup_keys)
    in_subgroup = [f"{r.domain}:{r.example_id}:{r.sentence_key}" in subgroup_set for r in supported["A"]]
    subgroup = {
        name: transition_counts(
            [p for p, keep in zip(_pairs(supported[ref]), in_subgroup) if keep],
            [p for p, keep in zip(_pairs(supported[cond]), in_subgroup) if keep],
            error_when_unsupported=True,
        )
        for name, (cond, ref, _) in SUPPORTED_CONTRASTS.items()
    }
    labels: dict[str, Label] = {
        "B+": label_result(
            contrasts["bplus_vs_a"].transitions, contrasts["bplus_vs_a"].sign_flip_p,
            contrasts["w_vs_t1_unsupported"].interval, contrasts["bplus_vs_s"].transitions,
        ),
        "C": label_result(
            contrasts["c_vs_t1"].transitions, contrasts["c_vs_t1"].sign_flip_p,
            contrasts["c_vs_t1_unsupported"].interval, None,
        ),
    }
    return JointEvidenceReport(
        metadata=metadata, claim_review_sha256=review_sha256,
        population=PopulationSummary(
            supported_sentences=supported_count, supported_examples=len(eligible),
            unsupported_sentences=len(unsupported_sentences), unsupported_examples=len(by_record),
        ),
        supported=supported, unsupported=unsupported, contrasts=contrasts,
        subgroup_keys=sorted(subgroup_set), subgroup=subgroup, labels=labels,
    )
