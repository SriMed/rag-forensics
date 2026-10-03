"""CLI for the frozen joint-evidence experiment (issue #32) and its public synthetic fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.decomposition_evidence import (
    ClaimReviewArtifact,
    ResidualReviewArtifact,
    file_sha256,
    validate_claim_review,
)
from benchmark.experiment_cli import ENTAILMENT_MODEL, ENTAILMENT_REVISION, _load_records
from benchmark.grounding import CrossEncoderNLIVerifier, DeterministicClaimDecomposer
from benchmark.joint_evidence import JointEvidenceMetadata, run_joint_evidence_experiment
from benchmark.ragbench import DATASET_NAME, adapt_ragbench_row
from pins import DATASET_REVISION, EMBEDDING_MODEL, EMBEDDING_REVISION

PROTOCOL = Path(__file__).resolve().parents[1] / "evals/joint_evidence/v1-techqa/protocol.md"
FROZEN_CLAIM_REVIEW_SHA256 = "049cb1f15687be26661195f811694ece72e7deb2866e231288bb85586b9e7c39"
FROZEN_RESIDUAL_REVIEW_SHA256 = "5d7a83de0a775e0ebb5152f340f5c0d0a9a2f4db6798b2026b4adc03b3fb6ad1"
FROZEN_THRESHOLD = 0.0017914474026707317
EXPECTED_POPULATION = (39, 330)
DOMAINS = ["techqa"]
EVALUATION_SPLIT = "test"
EVALUATION_LIMIT = 100
SEED = 42
BOOTSTRAP_ITERATIONS = 2000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen joint-evidence protocol.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the TechQA pilot from the private frozen reviews")
    run.add_argument("--claims", required=True)
    run.add_argument("--residual-review", required=True)
    run.add_argument("--output", required=True)
    fixture = sub.add_parser("run-fixture", help="run the public synthetic fixture")
    fixture.add_argument("--fixture", required=True)
    fixture.add_argument("--output", required=True)
    return parser


def subgroup_keys(review: ResidualReviewArtifact) -> set[str]:
    return {
        f"{item.domain}:{item.example_id}:{item.sentence_key}"
        for item in review.items
        if item.category == "multi_sentence_support"
    }


def load_fixture(path: Path):
    """Return (records, frozen claim review) for a synthetic fixture, validated together."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = [adapt_ragbench_row(entry["row"], entry["domain"]) for entry in data["rows"]]
    review = ClaimReviewArtifact.model_validate(data["claim_review"])
    validate_claim_review(records, review, DeterministicClaimDecomposer())
    return records, review


def _verifier_and_counter():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ENTAILMENT_MODEL, revision=ENTAILMENT_REVISION)

    def count_tokens(premise: str, claim: str) -> int:
        return len(tokenizer(premise, claim)["input_ids"])

    return CrossEncoderNLIVerifier(ENTAILMENT_MODEL, revision=ENTAILMENT_REVISION), count_tokens


def _embedding():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)


def _metadata(review: ClaimReviewArtifact, residual_sha256: str | None, limit: int | None,
              domains: list[str]) -> JointEvidenceMetadata:
    return JointEvidenceMetadata(
        protocol="joint-evidence-protocol.v1", protocol_sha256=file_sha256(PROTOCOL),
        dataset=DATASET_NAME, dataset_revision=DATASET_REVISION, evaluation_split=EVALUATION_SPLIT,
        sample_limit_per_domain=limit, domains=domains, seed=SEED,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS, embedding_model=EMBEDDING_MODEL,
        embedding_model_revision=EMBEDDING_REVISION, verifier_model=ENTAILMENT_MODEL,
        verifier_revision=ENTAILMENT_REVISION, entailment_threshold=FROZEN_THRESHOLD,
        deterministic_decomposer=DeterministicClaimDecomposer.name,
        deterministic_decomposer_version=DeterministicClaimDecomposer.version,
        residual_review_sha256=residual_sha256, claim_review_revision=review.revision,
        reviewer_identity=review.reviewer_identity or "",
    )


def _write(path: Path, report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-fixture":
        records, review = load_fixture(Path(args.fixture))
        verifier, count_tokens = _verifier_and_counter()
        report = run_joint_evidence_experiment(
            records, _embedding(), DeterministicClaimDecomposer(), verifier, count_tokens,
            FROZEN_THRESHOLD, review, file_sha256(Path(args.fixture)),
            bootstrap_iterations=BOOTSTRAP_ITERATIONS, seed=SEED,
            metadata=_metadata(review, None, None, sorted({r.domain for r in records})),
        )
        _write(Path(args.output), report)
        return 0

    claims_path, residual_path = Path(args.claims), Path(args.residual_review)
    # Identity checks come first so a wrong artifact never triggers data or model loading.
    if file_sha256(claims_path) != FROZEN_CLAIM_REVIEW_SHA256:
        raise ValueError("claim review sha256 does not match the frozen protocol")
    if file_sha256(residual_path) != FROZEN_RESIDUAL_REVIEW_SHA256:
        raise ValueError("residual review sha256 does not match the frozen protocol")
    review = ClaimReviewArtifact.model_validate_json(claims_path.read_text(encoding="utf-8"))
    residual = ResidualReviewArtifact.model_validate_json(residual_path.read_text(encoding="utf-8"))
    records, _ = _load_records(DOMAINS, EVALUATION_SPLIT, EVALUATION_LIMIT, SEED, DATASET_REVISION)
    verifier, count_tokens = _verifier_and_counter()
    report = run_joint_evidence_experiment(
        records, _embedding(), DeterministicClaimDecomposer(), verifier, count_tokens,
        FROZEN_THRESHOLD, review, FROZEN_CLAIM_REVIEW_SHA256, subgroup_keys=subgroup_keys(residual),
        bootstrap_iterations=BOOTSTRAP_ITERATIONS, seed=SEED, expected_population=EXPECTED_POPULATION,
        metadata=_metadata(review, FROZEN_RESIDUAL_REVIEW_SHA256, EVALUATION_LIMIT, DOMAINS),
    )
    _write(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
