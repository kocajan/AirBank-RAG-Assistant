from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class ChatRequest(BaseModel):
    session_id: UUID
    message: str = Field(min_length=1, max_length=4_000)


class RetrievedSource(BaseModel):
    title: str
    url: HttpUrl
    score: float


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    sources: list[RetrievedSource] = Field(default_factory=list)
    turns_used: int
    turns_remaining: int


class ResetRequest(BaseModel):
    session_id: UUID


class ResetResponse(BaseModel):
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    embedding_model: str
    openai_configured: bool
    rag_index_ready: bool
