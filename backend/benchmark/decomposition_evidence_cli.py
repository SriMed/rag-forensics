"""CLI for the decomposition-by-evidence experiment and its review artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.decomposition_evidence import (
    ClaimReviewArtifact,
    ExperimentMetadata,
    file_sha256,
    make_claim_review_template,
    make_residual_review_template,
    run_decomposition_evidence_experiment,
)
from benchmark.experiment_cli import (
    DATASET_REVISION,
    DOMAINS,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    ENTAILMENT_MODEL,
    ENTAILMENT_REVISION,
    _load_records,
)
from benchmark.grounding import CrossEncoderNLIVerifier, DeterministicClaimDecomposer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen 2x2 decomposition/evidence diagnostic.")
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("prepare-claims", help="create a human claim-review draft")
    template.add_argument("--output", required=True)
    template.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    template.add_argument("--evaluation-split", default="test")
    template.add_argument("--evaluation-limit", type=int, default=100)
    template.add_argument("--seed", type=int, default=42)
    template.add_argument("--dataset-revision", default=DATASET_REVISION)

    run = sub.add_parser("run", help="run all four conditions from a frozen claim review")
    run.add_argument("--claims", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--residual-review-output", required=True)
    run.add_argument("--entailment-threshold", type=float, required=True)
    run.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    run.add_argument("--evaluation-split", default="test")
    run.add_argument("--evaluation-limit", type=int, default=100)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--bootstrap-iterations", type=int, default=2000)
    run.add_argument("--dataset-revision", default=DATASET_REVISION)
    run.add_argument("--embedding-revision", default=EMBEDDING_REVISION)
    run.add_argument("--entailment-model", default=ENTAILMENT_MODEL)
    run.add_argument("--entailment-revision", default=ENTAILMENT_REVISION)
    return parser


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-claims":
        records, _ = _load_records(args.domains, args.evaluation_split, args.evaluation_limit,
                                   args.seed, args.dataset_revision)
        _write(
            Path(args.output),
            make_claim_review_template(records, DeterministicClaimDecomposer()),
        )
        return 0

    claims_path = Path(args.claims)
    review = ClaimReviewArtifact.model_validate_json(claims_path.read_text(encoding="utf-8"))
    records, _ = _load_records(args.domains, args.evaluation_split, args.evaluation_limit,
                               args.seed, args.dataset_revision)
    from sentence_transformers import SentenceTransformer
    embedding = SentenceTransformer(EMBEDDING_MODEL, revision=args.embedding_revision)
    verifier = CrossEncoderNLIVerifier(args.entailment_model, revision=args.entailment_revision)
    report = run_decomposition_evidence_experiment(
        records, embedding, DeterministicClaimDecomposer(), verifier,
        args.entailment_threshold, review, file_sha256(claims_path),
        args.bootstrap_iterations, args.seed,
        ExperimentMetadata(
            dataset="galileo-ai/ragbench", dataset_revision=args.dataset_revision,
            evaluation_split=args.evaluation_split,
            sample_limit_per_domain=args.evaluation_limit, seed=args.seed,
            bootstrap_iterations=args.bootstrap_iterations,
            embedding_model=EMBEDDING_MODEL, embedding_model_revision=args.embedding_revision,
            verifier_model=args.entailment_model, verifier_revision=args.entailment_revision,
            entailment_threshold=args.entailment_threshold,
            deterministic_decomposer=DeterministicClaimDecomposer.name,
            deterministic_decomposer_version=DeterministicClaimDecomposer.version,
            claim_review_revision=review.revision,
            reviewer_identity=review.reviewer_identity or "",
        ),
    )
    output = Path(args.output)
    _write(output, report)
    _write(
        Path(args.residual_review_output),
        make_residual_review_template(report, file_sha256(output), records, review),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
