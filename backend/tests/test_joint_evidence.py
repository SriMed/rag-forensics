import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from benchmark import joint_evidence_cli
from benchmark.decomposition_evidence import (
    ClaimReviewArtifact,
    ResidualReviewArtifact,
    ResidualReviewItem,
    make_claim_review_template,
)
from benchmark.grounding import DeterministicClaimDecomposer
from benchmark.joint_evidence import (
    TransitionCounts,
    aggregate_sentence,
    build_premise,
    clustered_difference_interval,
    distractor_keys,
    label_result,
    run_joint_evidence_experiment,
    sign_flip_p_value,
    top_k_indices,
    transition_counts,
    verify_joint,
    verify_separate,
    window_keys,
)
from benchmark.ragbench import adapt_ragbench_row
from models import AtomicClaim, ConfidenceInterval, NLIVerifierScores

CRASH = "APAR PH1 fixes the crash."
INTRO = "Intro text applies."
VERSION = "Version 9 is required."

DOCUMENT_0 = [
    ("0a", "Intro text."),
    ("0b", "Context before."),
    ("0c", "The patch is APAR PH1."),
    ("0d", "It fixes the crash."),
    ("0e", "Filler one."),
    ("0f", "Release notes follow."),
    ("0g", "Version 9 notes."),
    ("0h", "It is mandatory."),
    ("0i", "Filler two."),
]
DOCUMENT_1 = [("1a", "Other doc one."), ("1b", "Other doc two.")]

# Dimensions: crash topic, intro topic, version topic, filler.
VECTORS = {
    CRASH: [1, 0, 0, 0],
    INTRO: [0, 1, 0, 0],
    VERSION: [0, 0, 1, 0],
    "Intro text.": [0, 1, 0, 0],
    "The patch is APAR PH1.": [1, 0, 0, 0],
    "It fixes the crash.": [0.6, 0, 0, 0.8],
    "Version 9 notes.": [0, 0, 1, 0],
    "It is mandatory.": [0, 0, 0.6, 0.8],
}


class TextEmbedding:
    def encode(self, texts):
        return np.asarray([VECTORS.get(text, [0, 0, 0, 1]) for text in texts], dtype=float)


class KeySetVerifier:
    """Entails a claim only when the premise contains every required evidence key."""

    name = "key-set"
    revision = "1"

    def __init__(self, required, failing=()):
        self.required = required
        self.failing = set(failing)
        self.premises = []

    def score(self, claim, evidence):
        self.premises.append(evidence.text)
        if claim.text in self.failing:
            raise RuntimeError("verifier unavailable")
        keys = set(evidence.sentence_key.split("+"))
        entailment = 0.9 if self.required[claim.text] <= keys else 0.1
        return NLIVerifierScores(
            entailment=entailment, neutral=1 - entailment, contradiction=0.0,
            label="entailment" if entailment > 0.5 else "neutral",
        )


REQUIRED = {CRASH: {"0c", "0d"}, INTRO: {"0a"}, VERSION: {"0g", "0h"}}


def short_tokens(premise, claim):
    return len(premise.split()) + len(claim.split())


def _record():
    return adapt_ragbench_row(
        {
            "id": "ex-1",
            "question": "Does the patch fix the crash?",
            "documents": [" ".join(t for _, t in DOCUMENT_0), " ".join(t for _, t in DOCUMENT_1)],
            "documents_sentences": [
                [list(pair) for pair in DOCUMENT_0], [list(pair) for pair in DOCUMENT_1],
            ],
            "response": f"{CRASH} {INTRO} {VERSION}",
            "response_sentences": [["a", CRASH], ["b", INTRO], ["u", VERSION]],
            "sentence_support_information": [
                {"response_sentence_key": "a", "fully_supported": True, "supporting_sentence_keys": ["0c"]},
                {"response_sentence_key": "b", "fully_supported": True, "supporting_sentence_keys": ["0a"]},
                {"response_sentence_key": "u", "fully_supported": False, "supporting_sentence_keys": []},
            ],
            "unsupported_response_sentence_keys": ["u"],
        },
        domain="techqa",
    )


def _review(status="frozen"):
    template = make_claim_review_template([_record()], DeterministicClaimDecomposer())
    data = template.model_dump()
    data["status"] = status
    data["reviewer_identity"] = "reviewer@example.com" if status == "frozen" else None
    for item in data["items"]:
        item["accept_deterministic"] = True if status == "frozen" else None
    return ClaimReviewArtifact.model_validate(data)


def _claim(text, key="a"):
    return AtomicClaim(claim_id=f"{key}.claim-0", parent_sentence_key=key, text=text)


def _run(count_tokens=short_tokens, verifier=None, **kwargs):
    return run_joint_evidence_experiment(
        [_record()], TextEmbedding(), DeterministicClaimDecomposer(),
        verifier or KeySetVerifier(REQUIRED), count_tokens, 0.5, _review(), "c" * 64,
        subgroup_keys={"techqa:ex-1:a"}, bootstrap_iterations=50, seed=42, **kwargs,
    )


def _keys(report, population, condition, sentence_key):
    rows = getattr(report, population)[condition]
    row = next(x for x in rows if x.sentence_key == sentence_key)
    return [claim.evidence_keys for claim in row.claims]


# --- evidence construction -------------------------------------------------


def test_window_adds_one_neighbor_each_side_in_document_order():
    record = _record()
    assert window_keys(record, ["0c"]) == ["0b", "0c", "0d"]
    assert window_keys(record, ["0a"]) == ["0a", "0b"]


def test_window_merges_overlaps_and_never_crosses_documents():
    record = _record()
    assert window_keys(record, ["0c", "0b"]) == ["0a", "0b", "0c", "0d"]
    assert window_keys(record, ["0i"]) == ["0h", "0i"]
    assert window_keys(record, ["1a"]) == ["1a", "1b"]


def test_distractors_exclude_annotated_and_nearby_sentences_and_use_frozen_seed():
    record = _record()
    # Sentence a is supported by 0c; sentence b's annotation (0a) is excluded too.
    eligible = ["0f", "0g", "0h", "0i"]
    digest = hashlib.sha256(b"42:techqa:ex-1:a").digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    expected = sorted(rng.choice(eligible, size=2, replace=False).tolist(), key=eligible.index)

    assert distractor_keys(record, "a", 2, seed=42) == expected
    assert distractor_keys(record, "a", 2, seed=42) == expected


def test_distractor_shortfall_returns_none_instead_of_padding():
    assert distractor_keys(_record(), "a", 5, seed=42) is None


def test_top_k_breaks_ties_by_earlier_document_position():
    assert top_k_indices(np.asarray([0.5, 0.9, 0.5, 0.1]), 2) == [1, 0]
    assert top_k_indices(np.asarray([0.0, 0.0, 0.0]), 2) == [0, 1]


def test_premise_is_deduplicated_document_ordered_and_space_joined():
    premise = build_premise(_record(), ["0d", "0c", "0c"])
    assert premise.keys == ["0c", "0d"]
    assert premise.text == "The patch is APAR PH1. It fixes the crash."
    assert premise.evidence_key == "0c+0d"


# --- verification and unavailable states ----------------------------------


def test_joint_verification_records_inputs_scores_and_provenance():
    verifier = KeySetVerifier(REQUIRED)
    premise = build_premise(_record(), ["0c", "0d"])
    result = verify_joint(verifier, short_tokens, _claim(CRASH), premise, 0.5)

    assert result.status == "ok"
    assert result.predicted_supported is True
    assert result.evidence_keys == ["0c", "0d"]
    assert result.premise == premise.text
    assert result.token_count == short_tokens(premise.text, CRASH)
    assert result.nli_scores.entailment == pytest.approx(0.9)


def test_over_budget_pair_is_flagged_and_never_sent_to_the_verifier():
    verifier = KeySetVerifier(REQUIRED)
    premise = build_premise(_record(), ["0c", "0d"])
    result = verify_joint(verifier, lambda p, c: 513, _claim(CRASH), premise, 0.5)

    assert result.status == "over_budget"
    assert result.predicted_supported is None
    assert result.token_count == 513
    assert verifier.premises == []


def test_budget_boundary_is_inclusive_at_512_tokens():
    premise = build_premise(_record(), ["0c", "0d"])
    result = verify_joint(KeySetVerifier(REQUIRED), lambda p, c: 512, _claim(CRASH), premise, 0.5)
    assert result.status == "ok"


def test_verifier_exception_is_unavailable_not_unsupported():
    verifier = KeySetVerifier(REQUIRED, failing={CRASH})
    premise = build_premise(_record(), ["0c"])
    result = verify_joint(verifier, short_tokens, _claim(CRASH), premise, 0.5)

    assert result.status == "verifier_error"
    assert result.predicted_supported is None
    assert "verifier unavailable" in result.error


def test_separate_scoring_takes_the_best_annotated_sentence_per_claim():
    best, pairs = verify_separate(
        KeySetVerifier({CRASH: {"0d"}}), short_tokens, _claim(CRASH), _record(),
        ["0c", "0d"], 0.5,
    )
    assert [pair.evidence_keys for pair in pairs] == [["0c"], ["0d"]]
    assert best.evidence_keys == ["0d"]
    assert best.predicted_supported is True


def test_sentence_aggregation_requires_all_claims_and_reports_unavailable_reason():
    record = _record()
    verifier = KeySetVerifier(REQUIRED, failing={INTRO})
    ok = verify_joint(verifier, short_tokens, _claim(CRASH), build_premise(record, ["0c", "0d"]), 0.5)
    bad = verify_joint(verifier, short_tokens, _claim(CRASH), build_premise(record, ["0c"]), 0.5)
    error = verify_joint(verifier, short_tokens, _claim(INTRO), build_premise(record, ["0a"]), 0.5)
    over = verify_joint(verifier, lambda p, c: 600, _claim(CRASH), build_premise(record, ["0c"]), 0.5)

    assert aggregate_sentence([ok]) == (False, None)
    assert aggregate_sentence([ok, bad]) == (True, None)
    assert aggregate_sentence([ok, error]) == (None, "verifier_error")
    assert aggregate_sentence([error, over]) == (None, "over_budget")


# --- analysis -------------------------------------------------------------


def test_transition_counts_separate_corrections_introductions_and_unavailable():
    reference = [(True, None), (False, None), (True, None), (None, "over_budget"), (False, None)]
    condition = [(False, None), (True, None), (True, None), (False, None), (None, "verifier_error")]
    counts = transition_counts(reference, condition, error_when_unsupported=True)
    assert counts == TransitionCounts(
        corrected=1, introduced=1, unchanged=1,
        unavailable={"over_budget": 1, "verifier_error": 1},
    )


def test_transition_counts_on_unsupported_sentences_track_false_supported_errors():
    counts = transition_counts(
        [(True, None), (True, None)], [(False, None), (True, None)],
        error_when_unsupported=False,
    )
    assert (counts.corrected, counts.introduced, counts.unchanged) == (0, 1, 1)


def test_sign_flip_enumerates_every_cluster_assignment():
    assert sign_flip_p_value({"x": [-1], "y": [-1], "z": [-1]}) == pytest.approx(2 / 8)
    assert sign_flip_p_value({"x": [-1, 0], "y": [1]}) == pytest.approx(1.0)
    assert sign_flip_p_value({"x": [0], "y": [0]}) == pytest.approx(1.0)
    assert sign_flip_p_value({}) is None


def test_clustered_interval_of_a_constant_difference_has_no_width():
    interval = clustered_difference_interval({"x": [1, 1], "y": [1]}, iterations=20, seed=1)
    assert (interval.point_estimate, interval.lower, interval.upper) == (1.0, 1.0, 1.0)
    assert clustered_difference_interval({}, iterations=20, seed=1) is None


def _interval(point, lower, upper):
    return ConfidenceInterval(point_estimate=point, lower=lower, upper=upper, iterations=1, seed=1)


GOOD = TransitionCounts(corrected=5, introduced=1, unchanged=3, unavailable={})
FLAT = TransitionCounts(corrected=1, introduced=1, unchanged=7, unavailable={})


@pytest.mark.parametrize(
    ("leniency", "p_value", "size_control", "expected"),
    [
        (_interval(0.06, -0.01, 0.12), 0.01, GOOD, "confounded_by_leniency"),
        (_interval(0.02, 0.005, 0.04), 0.01, GOOD, "confounded_by_leniency"),
        (_interval(0.05, -0.01, 0.10), 0.01, GOOD, "supported"),
        (_interval(0.0, -0.03, 0.03), 0.05, GOOD, "inconclusive"),
        (_interval(0.0, -0.03, 0.03), 0.01, FLAT, "inconclusive"),
        (None, 0.01, GOOD, "inconclusive"),
    ],
)
def test_label_rules_follow_the_frozen_order(leniency, p_value, size_control, expected):
    assert label_result(GOOD, p_value, leniency, size_control) == expected


def test_label_without_size_control_uses_primary_contrast_only():
    assert label_result(GOOD, 0.01, _interval(0.0, -0.03, 0.03), None) == "supported"
    assert label_result(FLAT, 0.01, _interval(0.0, -0.03, 0.03), None) == "inconclusive"


# --- end-to-end runner ------------------------------------------------------


def test_runner_builds_each_condition_from_the_frozen_rules():
    report = _run()

    assert _keys(report, "supported", "A", "a") == [["0c"]]
    assert _keys(report, "supported", "B", "a") == [["0c"]]
    assert _keys(report, "supported", "B+", "a") == [["0b", "0c", "0d"]]
    s_keys = _keys(report, "supported", "S", "a")[0]
    assert "0c" in s_keys and len(s_keys) == 3 and set(s_keys) - {"0c"} <= {"0f", "0g", "0h", "0i"}
    assert _keys(report, "supported", "T1", "a") == [["0c"]]
    assert _keys(report, "supported", "C", "a") == [["0a", "0c", "0d"]]
    assert _keys(report, "unsupported", "T1", "u") == [["0g"]]
    assert _keys(report, "unsupported", "C", "u") == [["0a", "0g", "0h"]]
    assert _keys(report, "unsupported", "W", "u") == [["0f", "0g", "0h"]]


def test_runner_accounts_for_every_sentence_and_reports_contrasts():
    report = _run()

    assert report.population.supported_sentences == 2
    assert report.population.unsupported_sentences == 1
    for rows in report.supported.values():
        assert sorted(row.sentence_key for row in rows) == ["a", "b"]
    contrasts = report.contrasts
    assert contrasts["bplus_vs_a"].transitions == TransitionCounts(
        corrected=1, introduced=0, unchanged=1, unavailable={},
    )
    assert contrasts["bplus_vs_s"].transitions.corrected == 1
    assert contrasts["c_vs_t1"].transitions.corrected == 1
    assert contrasts["b_vs_a"].transitions.unchanged == 2
    assert contrasts["bplus_vs_a"].sign_flip_p == pytest.approx(1.0)
    assert contrasts["w_vs_t1_unsupported"].transitions.introduced == 1
    assert contrasts["c_vs_t1_unsupported"].transitions.introduced == 1
    assert contrasts["w_vs_t1_unsupported"].interval.point_estimate == pytest.approx(1.0)
    assert report.labels == {"B+": "confounded_by_leniency", "C": "confounded_by_leniency"}
    assert report.subgroup["bplus_vs_a"].corrected == 1
    assert report.subgroup["bplus_vs_a"].unchanged == 0


def test_runner_uses_deterministic_claims_for_unsupported_sentences():
    report = _run()
    row = report.unsupported["T1"][0]
    assert [claim.claim.text for claim in row.claims] == [VERSION]
    assert row.gold_unsupported is True


def test_runner_flags_over_budget_sentences_and_keeps_them_out_of_contrasts():
    def budget(premise, claim):
        return 600 if "It is mandatory." in premise else 10

    verifier = KeySetVerifier(REQUIRED)
    report = _run(count_tokens=budget, verifier=verifier)

    w_row = report.unsupported["W"][0]
    assert (w_row.predicted_unsupported, w_row.unavailable_reason) == (None, "over_budget")
    assert all("It is mandatory." not in premise for premise in verifier.premises)
    assert report.contrasts["w_vs_t1_unsupported"].transitions.unavailable == {"over_budget": 1}
    assert report.contrasts["w_vs_t1_unsupported"].interval is None
    assert report.labels == {"B+": "inconclusive", "C": "inconclusive"}


def test_runner_records_verifier_failures_as_unavailable():
    report = _run(verifier=KeySetVerifier(REQUIRED, failing={INTRO}))
    row = next(x for x in report.supported["B+"] if x.sentence_key == "b")
    assert (row.predicted_unsupported, row.unavailable_reason) == (None, "verifier_error")
    assert report.contrasts["bplus_vs_a"].transitions.unavailable == {"verifier_error": 1}


def test_runner_rejects_unfrozen_review_and_population_mismatch():
    with pytest.raises(ValueError, match="must be frozen"):
        run_joint_evidence_experiment(
            [_record()], TextEmbedding(), DeterministicClaimDecomposer(),
            KeySetVerifier(REQUIRED), short_tokens, 0.5, _review("draft"), "c" * 64,
        )
    with pytest.raises(ValueError, match="population"):
        _run(expected_population=(39, 330))


# --- CLI and public fixture -------------------------------------------------

FIXTURE = Path(__file__).resolve().parents[1] / "evals/joint_evidence/fixture/fixture.json"


def test_cli_rejects_claim_review_that_is_not_the_frozen_artifact(tmp_path, monkeypatch):
    claims = tmp_path / "claims.json"
    claims.write_text("{}")
    residual = tmp_path / "residual.json"
    residual.write_text("{}")
    monkeypatch.setattr(joint_evidence_cli, "_load_records", lambda *a, **k: pytest.fail("loaded data"))
    with pytest.raises(ValueError, match="claim review sha256"):
        joint_evidence_cli.main([
            "run", "--claims", str(claims), "--residual-review", str(residual),
            "--output", str(tmp_path / "out.json"),
        ])


def test_cli_rejects_residual_review_that_is_not_the_frozen_artifact(tmp_path, monkeypatch):
    claims = tmp_path / "claims.json"
    claims.write_text("{}")
    residual = tmp_path / "residual.json"
    residual.write_text("{}")
    monkeypatch.setattr(joint_evidence_cli, "FROZEN_CLAIM_REVIEW_SHA256", joint_evidence_cli.file_sha256(claims))
    with pytest.raises(ValueError, match="residual review sha256"):
        joint_evidence_cli.main([
            "run", "--claims", str(claims), "--residual-review", str(residual),
            "--output", str(tmp_path / "out.json"),
        ])


def test_subgroup_is_the_multi_sentence_support_residual_category():
    def item(key, category):
        return ResidualReviewItem(
            domain="techqa", example_id="ex-1", sentence_key=key, question="q", full_response="r",
            sentence="s", reviewed_claims=["c"], annotated_evidence=["e"], verifier_outcomes=[],
            category=category,
        )

    review = ResidualReviewArtifact(
        schema_version="decomposition-evidence-residuals.v1", status="frozen",
        reviewer_identity="reviewer@example.com", reviewer_instructions="x",
        experiment_report_sha256="d" * 64,
        items=[item("a", "multi_sentence_support"), item("b", "ambiguous_label")],
    )
    assert joint_evidence_cli.subgroup_keys(review) == {"techqa:ex-1:a"}


def test_public_fixture_validates_and_runs_offline():
    records, review = joint_evidence_cli.load_fixture(FIXTURE)
    report = run_joint_evidence_experiment(
        records, TextEmbedding(), DeterministicClaimDecomposer(),
        KeySetVerifier(defaultdict_required()), short_tokens, 0.5, review, "f" * 64,
        bootstrap_iterations=20,
    )
    assert report.population.supported_sentences >= 3
    assert report.population.unsupported_sentences >= 2
    assert report.population.supported_examples >= 2
    for rows in [*report.supported.values(), *report.unsupported.values()]:
        assert all(row.predicted_unsupported is not None for row in rows)


def defaultdict_required():
    return defaultdict(set)
