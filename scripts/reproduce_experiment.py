from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import get_settings
from evaluation.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DOCUMENTS_PATH,
    DEFAULT_INDEX_ROOT,
    DEFAULT_RESULTS_DIR,
)
from evaluation.dataset import generate_dataset
from evaluation.report import render_report, save_report
from evaluation.reproducibility import (
    verify_frozen_inputs,
    write_evaluation_manifest,
)
from evaluation.runner import run_grid_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the evaluation from a frozen Air Bank corpus. By default, "
            "an existing evaluation dataset is reused; pass --regenerate-dataset "
            "to create a new LLM-generated dataset."
        )
    )
    parser.add_argument("--documents", type=Path, default=ROOT_DIR / DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--dataset", type=Path, default=ROOT_DIR / DEFAULT_DATASET_PATH)
    parser.add_argument("--index-root", type=Path, default=ROOT_DIR / DEFAULT_INDEX_ROOT)
    parser.add_argument("--results", type=Path, default=ROOT_DIR / DEFAULT_RESULTS_DIR)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--excerpt-chars", type=int, default=6000)
    parser.add_argument("--regenerate-dataset", action="store_true")
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--judge-model", default=None)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not configured in .env")
    if not args.documents.is_file():
        raise SystemExit(
            f"Processed corpus not found: {args.documents}\n"
            "Run scripts/collect_data.py first or restore the frozen corpus snapshot."
        )

    api_key = settings.openai_api_key.get_secret_value()
    generation_model = args.generation_model or settings.eval_generation_model
    judge_model = args.judge_model or settings.eval_judge_model

    if args.regenerate_dataset or not args.dataset.is_file():
        print("[1/2] Generating evaluation dataset")
        generate_dataset(
            documents_path=args.documents,
            output_path=args.dataset,
            api_key=api_key,
            model=generation_model,
            count=args.count,
            seed=args.seed,
            excerpt_chars=args.excerpt_chars,
        )
    else:
        print("[1/2] Reusing frozen evaluation dataset")
        warnings = verify_frozen_inputs(
            documents_path=args.documents,
            dataset_path=args.dataset,
        )
        for warning in warnings:
            print(f"WARNING: {warning}")

    print("[2/2] Running evaluation grid")
    summaries = await run_grid_evaluation(
        dataset_path=args.dataset,
        documents_path=args.documents,
        index_root=args.index_root,
        results_dir=args.results,
        api_key=api_key,
        judge_model=judge_model,
        with_judge=args.with_judge,
        batch_size=args.batch_size,
        force_rebuild=args.force_rebuild,
        limit=args.limit,
    )

    report = render_report(summaries, with_judge=args.with_judge)
    report_path = save_report(
        summaries=summaries,
        output_dir=args.results,
        with_judge=args.with_judge,
    )
    manifest_path = write_evaluation_manifest(
        output_path=args.results / "reproducibility_manifest.json",
        documents_path=args.documents,
        dataset_path=args.dataset,
        judge_enabled=args.with_judge,
        judge_model=judge_model,
        limit=args.limit,
        retrieval_top_k=10,
    )

    print()
    print(report)
    print(f"Report:       {report_path}")
    print(f"Run manifest: {manifest_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
