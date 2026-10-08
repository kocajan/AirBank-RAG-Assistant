from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collection.collector import AirBankCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a controlled corpus of public Air Bank content for the RAG PoC."
    )
    parser.add_argument(
        "--max-html",
        type=int,
        default=200,
        help="Maximum number of Poradna article pages to collect (default: 200).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Minimum delay between HTTP requests in seconds (default: 0.5).",
    )
    parser.add_argument("--skip-html", action="store_true", help="Do not collect HTML pages.")
    parser.add_argument("--skip-pdfs", action="store_true", help="Do not collect PDFs.")
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Do not enforce robots.txt. Use only after checking the site's policy manually.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_html < 0:
        raise SystemExit("--max-html must be >= 0")

    root_dir = ROOT_DIR
    with AirBankCollector(
        root_dir,
        delay_seconds=args.delay,
        respect_robots=not args.ignore_robots,
    ) as collector:
        documents, stats = collector.collect(
            max_html=args.max_html,
            include_html=not args.skip_html and args.max_html > 0,
            include_pdfs=not args.skip_pdfs,
        )
        documents_path = collector.write_documents(documents)
        summary_path = collector.write_summary(documents, stats)

    print()
    print(f"Collected {len(documents)} documents.")
    print(f"JSONL:   {documents_path}")
    print(f"Summary: {summary_path}")
    print(
        "Stats: "
        f"html={stats.html_collected}/{stats.html_discovered}, "
        f"pdf={stats.pdf_collected}/{stats.pdf_discovered}, "
        f"skipped={stats.skipped}, failed={stats.failed}"
    )


if __name__ == "__main__":
    main()
