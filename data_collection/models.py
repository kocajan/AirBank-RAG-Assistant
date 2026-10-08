from datetime import datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl


class SourceDocument(BaseModel):
    """Normalized source document stored before chunking/indexing."""

    id: str
    title: str
    url: HttpUrl
    source_type: Literal["html", "pdf"]
    category: str
    fetched_at: datetime
    text: str
    content_sha256: str
    raw_path: str
