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
from evaluation.report import render_report, save_report
from evaluation.reproducibility import verify_frozen_inputs, write_evaluation_manifest
from evaluation.runner import run_grid_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 3x3 embedding/chunking RAG grid and print a comparison report."
    )
    parser.add_argument("--dataset", type=Path, default=ROOT_DIR / DEFAULT_DATASET_PATH)
    parser.add_argument("--documents", type=Path, default=ROOT_DIR / DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--index-root", type=Path, default=ROOT_DIR / DEFAULT_INDEX_ROOT)
    parser.add_argument("--results", type=Path, default=ROOT_DIR / DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Also run the full chatbot and an LLM answer judge for every query/configuration.",
    )
    parser.add_argument("--judge-model", default=None, help="Override EVAL_JUDGE_MODEL")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions; useful for a cheap smoke test.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not configured in .env")
    if not args.dataset.is_file():
        raise SystemExit(
            f"Evaluation dataset not found: {args.dataset}\n"
            "Run scripts/generate_eval_dataset.py first."
        )

    judge_model = args.judge_model or settings.eval_judge_model
    for warning in verify_frozen_inputs(
        documents_path=args.documents, dataset_path=args.dataset
    ):
        print(f"WARNING: {warning}")

    summaries = await run_grid_evaluation(
        dataset_path=args.dataset,
        documents_path=args.documents,
        index_root=args.index_root,
        results_dir=args.results,
        api_key=settings.openai_api_key.get_secret_value(),
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
    print(f"Detailed results: {args.results}")
    print(f"Report:           {report_path}")
    print(f"Run manifest:     {manifest_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
