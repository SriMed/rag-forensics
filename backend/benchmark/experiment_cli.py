"""CLI for calibrated, multi-domain B0-B3 grounding experiments."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from benchmark.experiment import run_experiment
from benchmark.grounding import (
    CrossEncoderNLIVerifier,
    DeterministicClaimDecomposer,
    METHODS,
)
from benchmark.ragbench import DATASET_NAME, RAGBenchRowError, adapt_ragbench_row
from models import GroundingRunMetadata

DATASET_REVISION = "97808f3e5fd16ede40bbff6c2949af8139b2eb7b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENTAILMENT_MODEL = "cross-encoder/nli-deberta-v3-base"
ENTAILMENT_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
DOMAINS = ("techqa", "finqa", "covidqa")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate and compare B0-B3 grounding methods without test leakage."
    )
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--calibration-split", default="validation")
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--calibration-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--embedding-revision", default=EMBEDDING_REVISION)
    parser.add_argument("--entailment-model", default=ENTAILMENT_MODEL)
    parser.add_argument(
        "--entailment-revision",
        default=ENTAILMENT_REVISION,
        help="Immutable Hugging Face commit hash for the NLI model.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods retained in the report; all methods share the same evaluation pass.",
    )
    return parser


def _load_records(
    domains: list[str],
    split: str,
    limit: int | None,
    seed: int,
    revision: str,
):
    from datasets import load_dataset

    records = []
    skipped: list[str] = []
    for domain in domains:
        rows = load_dataset(
            DATASET_NAME,
            domain,
            split=split,
            revision=revision,
        ).shuffle(seed=seed)
        if limit is not None:
            if limit < 1:
                raise ValueError("sample limits must be at least 1")
            rows = rows.select(range(min(limit, len(rows))))
        for index, row in enumerate(rows):
            try:
                records.append(adapt_ragbench_row(row, domain))
            except RAGBenchRowError as exc:
                skipped.append(f"{domain}:{row.get('id', index)}: {exc}")
    return records, skipped


def _code_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{commit}+dirty" if dirty else commit


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.calibration_split == args.evaluation_split:
        raise ValueError("calibration and evaluation split names must differ")
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be at least 1")

    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        revision=args.embedding_revision,
    )
    verifier = CrossEncoderNLIVerifier(
        args.entailment_model,
        revision=args.entailment_revision,
    )
    decomposer = DeterministicClaimDecomposer()
    calibration_records, calibration_skips = _load_records(
        args.domains,
        args.calibration_split,
        args.calibration_limit,
        args.seed,
        args.dataset_revision,
    )
    evaluation_records, evaluation_skips = _load_records(
        args.domains,
        args.evaluation_split,
        args.evaluation_limit,
        args.seed,
        args.dataset_revision,
    )
    metadata = GroundingRunMetadata(
        dataset=DATASET_NAME,
        dataset_revision=args.dataset_revision,
        split_strategy="official named splits; thresholds selected on calibration only",
        calibration_split=args.calibration_split,
        evaluation_split=args.evaluation_split,
        seed=args.seed,
        embedding_model=EMBEDDING_MODEL,
        embedding_model_revision=args.embedding_revision,
        entailment_model=args.entailment_model,
        entailment_model_revision=args.entailment_revision,
        claim_decomposer=decomposer.name,
        claim_decomposer_version=decomposer.version,
        similarity_threshold=0.0,
        entailment_threshold=0.0,
        code_commit=_code_commit(),
        calibration_sample_count=len(calibration_records),
        evaluation_sample_count=len(evaluation_records),
        skipped_rows=[
            f"calibration:{item}" for item in calibration_skips
        ]
        + [f"evaluation:{item}" for item in evaluation_skips],
    )
    report = run_experiment(
        calibration_records,
        evaluation_records,
        embedding_model=embedding_model,
        decomposer=decomposer,
        entailment_verifier=verifier,
        metadata=metadata,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    report.methods = {
        method: result
        for method, result in report.methods.items()
        if method in args.methods
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
