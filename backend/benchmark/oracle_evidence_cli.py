"""CLI for the label-derived RAGBench oracle-evidence diagnostic."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

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
from benchmark.oracle_evidence import run_oracle_evidence_diagnostic
from models import OracleEvidenceRunMetadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Localize supported-sentence failures by replacing selected evidence "
            "with label-derived RAGBench evidence. Not a deployable classifier."
        )
    )
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--embedding-revision", default=EMBEDDING_REVISION)
    parser.add_argument("--entailment-model", default=ENTAILMENT_MODEL)
    parser.add_argument("--entailment-revision", default=ENTAILMENT_REVISION)
    parser.add_argument(
        "--entailment-threshold",
        type=float,
        required=True,
        help="Frozen B3 entailment threshold; use the calibrated run value when available.",
    )
    parser.add_argument("--output", required=True)
    return parser


def _code_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    return f"{commit}+dirty" if dirty else commit


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be at least 1")
    if args.evaluation_limit is not None and args.evaluation_limit < 1:
        raise ValueError("--evaluation-limit must be at least 1")
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBEDDING_MODEL, revision=args.embedding_revision)
    verifier = CrossEncoderNLIVerifier(
        args.entailment_model, revision=args.entailment_revision
    )
    decomposer = DeterministicClaimDecomposer()
    records, skipped = _load_records(
        args.domains,
        args.evaluation_split,
        args.evaluation_limit,
        args.seed,
        args.dataset_revision,
    )
    report = run_oracle_evidence_diagnostic(
        records,
        embedding_model,
        decomposer,
        verifier,
        args.entailment_threshold,
        args.bootstrap_iterations,
        args.seed,
    )
    report.metadata = OracleEvidenceRunMetadata(
        dataset="galileo-ai/ragbench",
        dataset_revision=args.dataset_revision,
        evaluation_split=args.evaluation_split,
        seed=args.seed,
        embedding_model=EMBEDDING_MODEL,
        embedding_model_revision=args.embedding_revision,
        entailment_model=args.entailment_model,
        entailment_model_revision=args.entailment_revision,
        claim_decomposer=decomposer.name,
        claim_decomposer_version=decomposer.version,
        entailment_threshold=args.entailment_threshold,
        code_commit=_code_commit(),
        sample_count=len(records),
        skipped_rows=skipped,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
