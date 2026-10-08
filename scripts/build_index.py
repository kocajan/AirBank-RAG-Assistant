from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import get_settings
from rag.embeddings import create_openai_embedder
from rag.index import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk the collected Air Bank corpus, embed it, and build a local vector index."
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "documents.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data" / "index",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    if args.chunk_size < 200:
        raise SystemExit("--chunk-size must be >= 200")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise SystemExit("--chunk-overlap must be >= 0 and smaller than --chunk-size")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not configured in .env")

    embedder = create_openai_embedder(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=settings.embedding_model,
    )
    metadata = await build_index(
        documents_path=args.documents,
        index_dir=args.output,
        embedder=embedder,
        embedding_model_name=settings.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )

    print()
    print("Vector index built successfully.")
    print(f"Documents:  {metadata.documents}")
    print(f"Chunks:     {metadata.chunks}")
    print(f"Dimensions: {metadata.embedding_dimensions}")
    print(f"Model:      {metadata.embedding_model}")
    print(f"Output:     {args.output}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
