from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag.index import EMBEDDINGS_FILENAME, METADATA_FILENAME, CHUNKS_FILENAME, LocalVectorIndex
from rag.models import Chunk, IndexMetadata


class FakeEmbeddingResult:
    embeddings = [[1.0, 0.0]]


class FakeEmbedder:
    async def embed_query(self, query: str) -> FakeEmbeddingResult:
        return FakeEmbeddingResult()


def _write_index(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    chunks = [
        Chunk(
            id="a",
            document_id="doc-a",
            chunk_index=0,
            title="Accounts",
            url="https://www.airbank.cz/accounts/",
            source_type="html",
            category="Accounts",
            text="Account information",
        ),
        Chunk(
            id="b",
            document_id="doc-b",
            chunk_index=0,
            title="Cards",
            url="https://www.airbank.cz/cards/",
            source_type="html",
            category="Cards",
            text="Card information",
        ),
    ]
    with (path / CHUNKS_FILENAME).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    np.save(path / EMBEDDINGS_FILENAME, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    metadata = IndexMetadata(
        created_at=datetime.now(timezone.utc),
        embedding_model="fake",
        embedding_dimensions=2,
        chunk_size=1000,
        chunk_overlap=100,
        documents=2,
        chunks=2,
    )
    (path / METADATA_FILENAME).write_text(metadata.model_dump_json(), encoding="utf-8")


async def _search(path: Path):
    index = LocalVectorIndex(
        index_dir=path,
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        top_k=2,
        min_score=-1.0,
    )
    return await index.search("accounts")


def test_vector_index_returns_most_similar_chunk(tmp_path: Path) -> None:
    import asyncio

    _write_index(tmp_path)
    hits = asyncio.run(_search(tmp_path))

    assert hits[0].chunk.id == "a"
    assert hits[0].score > hits[1].score
