"""External-validation CLI for the official RAGTruth JSONL release."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from benchmark.experiment import run_experiment
from benchmark.experiment_cli import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    ENTAILMENT_MODEL,
    ENTAILMENT_REVISION,
    _code_commit,
)
from benchmark.grounding import CrossEncoderNLIVerifier, DeterministicClaimDecomposer
from benchmark.ragtruth import RAGTruthRowError, adapt_ragtruth_row
from models import GroundingRunMetadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run span-to-sentence external validation on RAGTruth."
    )
    parser.add_argument("--source-info", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--embedding-revision", default=EMBEDDING_REVISION)
    parser.add_argument("--entailment-model", default=ENTAILMENT_MODEL)
    parser.add_argument("--entailment-revision", default=ENTAILMENT_REVISION)
    parser.add_argument("--output", required=True)
    return parser


def _jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _adapt_partition(
    responses: list[dict],
    sources: dict[str, dict],
    split: str,
    limit: int | None,
    seed: int,
):
    selected = [
        row
        for row in responses
        if row.get("split") == split and row.get("quality", "good") == "good"
    ]
    random.Random(seed).shuffle(selected)
    if limit is not None:
        if limit < 1:
            raise ValueError("sample limits must be at least 1")
        selected = selected[:limit]
    records = []
    skips = []
    for row in selected:
        row_id = str(row.get("id", "unknown"))
        source = sources.get(str(row.get("source_id")))
        if source is None:
            skips.append(f"{row_id}: missing source_info")
            continue
        try:
            records.append(adapt_ragtruth_row(row, source))
        except RAGTruthRowError as exc:
            skips.append(f"{row_id}: {exc}")
    return records, skips


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_rows = _jsonl(args.source_info)
    response_rows = _jsonl(args.responses)
    sources = {str(row.get("source_id")): row for row in source_rows}
    calibration, calibration_skips = _adapt_partition(
        response_rows, sources, "train", args.train_limit, args.seed
    )
    evaluation, evaluation_skips = _adapt_partition(
        response_rows, sources, "test", args.test_limit, args.seed
    )

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
    metadata = GroundingRunMetadata(
        dataset="ParticleMedia/RAGTruth",
        dataset_revision=args.dataset_revision,
        split_strategy=(
            "official response train/test split; good-quality responses only; "
            "hallucination-span overlap converted to sentence labels"
        ),
        calibration_split="train",
        evaluation_split="test",
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
        calibration_sample_count=len(calibration),
        evaluation_sample_count=len(evaluation),
        skipped_rows=[
            f"calibration:{item}" for item in calibration_skips
        ]
        + [f"evaluation:{item}" for item in evaluation_skips],
    )
    report = run_experiment(
        calibration,
        evaluation,
        embedding_model=embedding_model,
        decomposer=decomposer,
        entailment_verifier=verifier,
        metadata=metadata,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
