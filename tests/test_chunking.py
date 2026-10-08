from datetime import datetime, timezone

from data_collection.models import SourceDocument
from rag.chunking import chunk_document, chunk_text


def test_chunk_text_splits_and_keeps_content() -> None:
    text = "\n".join(f"Paragraph {i}: " + ("x" * 180) for i in range(12))
    chunks = chunk_text(text, chunk_size=600, overlap=120)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 600 for chunk in chunks)
    assert "Paragraph 0" in chunks[0]
    assert any("Paragraph 11" in chunk for chunk in chunks)


def test_chunk_document_preserves_source_metadata() -> None:
    document = SourceDocument(
        id="doc-1",
        title="Test title",
        url="https://www.airbank.cz/test/",
        source_type="html",
        category="Test",
        fetched_at=datetime.now(timezone.utc),
        text="A" * 500,
        content_sha256="abc",
        raw_path="data/raw/html/test.html",
    )

    chunks = chunk_document(document, chunk_size=300, overlap=50)

    assert len(chunks) >= 2
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert all(str(chunk.url) == str(document.url) for chunk in chunks)
