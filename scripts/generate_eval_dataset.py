from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import get_settings
from evaluation.config import DEFAULT_DATASET_PATH, DEFAULT_DOCUMENTS_PATH
from evaluation.dataset import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM-written QA evaluation pairs from randomly selected Air Bank source documents."
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--excerpt-chars", type=int, default=6000)
    parser.add_argument("--documents", type=Path, default=ROOT_DIR / DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / DEFAULT_DATASET_PATH)
    parser.add_argument("--model", default=None, help="Override EVAL_GENERATION_MODEL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not configured in .env")

    model = args.model or settings.eval_generation_model
    items = generate_dataset(
        documents_path=args.documents,
        output_path=args.output,
        api_key=settings.openai_api_key.get_secret_value(),
        model=model,
        count=args.count,
        seed=args.seed,
        excerpt_chars=args.excerpt_chars,
    )
    print()
    print("Evaluation dataset generated successfully.")
    print(f"Questions: {len(items)}")
    print(f"Model:     {model}")
    print(f"Seed:      {args.seed}")
    print(f"Output:    {args.output}")


if __name__ == "__main__":
    main()
