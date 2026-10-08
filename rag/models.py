from datetime import datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl


class Chunk(BaseModel):
    """One retrievable piece of a normalized source document."""

    id: str
    document_id: str
    chunk_index: int
    title: str
    url: HttpUrl
    source_type: Literal["html", "pdf"]
    category: str
    text: str


class RetrievalHit(BaseModel):
    """A chunk returned by vector similarity search."""

    chunk: Chunk
    score: float


class IndexMetadata(BaseModel):
    """Metadata needed to understand/reproduce a generated vector index."""

    version: int = 1
    created_at: datetime
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    documents: int
    chunks: int
