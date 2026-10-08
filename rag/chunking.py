from __future__ import annotations

import hashlib
import re

from data_collection.models import SourceDocument
from rag.models import Chunk


def _paragraphs(text: str) -> list[str]:
    """Turn extracted text into clean paragraph-like units."""

    parts = re.split(r"\n\s*\n|\n", text)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def _fixed_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback for a single paragraph that is larger than one chunk."""

    if len(text) <= chunk_size:
        return [text]

    step = max(chunk_size - overlap, 1)
    windows: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            windows.append(piece)
        if start + chunk_size >= len(text):
            break
        start += step
    return windows


def chunk_text(text: str, *, chunk_size: int = 1600, overlap: int = 250) -> list[str]:
    """Simple paragraph-aware character chunking with small overlap."""

    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    paragraphs = _paragraphs(text)
    chunks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        value = "\n\n".join(buffer).strip()
        if value and (not chunks or chunks[-1] != value):
            chunks.append(value)

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush()
            buffer.clear()
            for piece in _fixed_windows(paragraph, chunk_size, overlap):
                if not chunks or chunks[-1] != piece:
                    chunks.append(piece)
            continue

        candidate = "\n\n".join([*buffer, paragraph]) if buffer else paragraph
        if buffer and len(candidate) > chunk_size:
            flush()

            overlap_buffer: list[str] = []
            overlap_chars = 0
            for previous in reversed(buffer):
                addition = len(previous) + (2 if overlap_buffer else 0)
                if overlap_buffer and overlap_chars + addition > overlap:
                    break
                overlap_buffer.append(previous)
                overlap_chars += addition
            buffer[:] = reversed(overlap_buffer)

            while buffer and len("\n\n".join([*buffer, paragraph])) > chunk_size:
                buffer.pop(0)

        buffer.append(paragraph)

    flush()
    return chunks


def chunk_document(
    document: SourceDocument,
    *,
    chunk_size: int = 1600,
    overlap: int = 250,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, text in enumerate(
        chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
    ):
        digest = hashlib.sha1(
            f"{document.id}:{index}:{text}".encode("utf-8")
        ).hexdigest()[:12]
        chunks.append(
            Chunk(
                id=f"{document.id}-{index:04d}-{digest}",
                document_id=document.id,
                chunk_index=index,
                title=document.title,
                url=document.url,
                source_type=document.source_type,
                category=document.category,
                text=text,
            )
        )
    return chunks
