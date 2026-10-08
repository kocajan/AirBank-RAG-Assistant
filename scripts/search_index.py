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
from rag.index import LocalVectorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local Air Bank vector retrieval.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not configured in .env")

    embedder = create_openai_embedder(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=settings.embedding_model,
    )
    index = LocalVectorIndex(
        index_dir=settings.rag_index_dir,
        embedder=embedder,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
        max_chunks_per_document=settings.rag_max_chunks_per_document,
    )
    hits = await index.search(args.query, top_k=args.top_k)

    if not hits:
        print("No relevant chunks found.")
        return

    for number, hit in enumerate(hits, start=1):
        print(f"\n#{number} score={hit.score:.4f} — {hit.chunk.title}")
        print(hit.chunk.url)
        print(hit.chunk.text[:700].replace("\n", " "))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
