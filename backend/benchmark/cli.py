"""Command-line runner for label-preserving RAGBench evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from benchmark.ragbench import (
    DATASET_NAME,
    RAGBenchRowError,
    adapt_ragbench_row,
    evaluate_records,
)
from services.retriever import get_embedding_model

_DOMAINS = ("techqa", "finqa", "covidqa")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic attribution against label-preserved RAGBench responses."
    )
    parser.add_argument("--domain", choices=_DOMAINS, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path. Defaults to stdout.",
    )
    return parser


def run_rows(
    rows: Iterable[Mapping],
    embedding_model,
    domain: str,
    split: str,
    seed: int,
    requested_limit: int | None,
    timestamp: str,
):
    records = []
    skipped_rows: list[str] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id", f"row-{index}"))
        try:
            records.append(adapt_ragbench_row(row, domain=domain))
        except RAGBenchRowError as exc:
            skipped_rows.append(f"{row_id}: {exc}")
    return evaluate_records(
        records=records,
        embedding_model=embedding_model,
        dataset_config=domain,
        split=split,
        seed=seed,
        requested_limit=requested_limit,
        timestamp=timestamp,
        skipped_rows=skipped_rows,
    )


def _load_rows(domain: str, split: str, limit: int | None, seed: int):
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, domain, split=split)
    dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.domain, args.split, args.limit, args.seed)
    report = run_rows(
        rows=rows,
        embedding_model=get_embedding_model(),
        domain=args.domain,
        split=args.split,
        seed=args.seed,
        requested_limit=args.limit,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    payload = report.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
