import logging

from fastapi import FastAPI, HTTPException, status

from backend.agent import AgentDependencies, get_agent
from backend.config import get_settings
from backend.memory import InMemoryConversationStore
from backend.rag_runtime import get_retriever
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    RetrievedSource,
)
from rag.index import index_is_ready
from rag.models import RetrievalHit


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
conversation_store = InMemoryConversationStore(max_sessions=settings.max_sessions)

app = FastAPI(
    title="Air Bank RAG Assistant API",
    version="1.2.0",
    description="PydanticAI/OpenAI chatbot grounded in indexed public Air Bank data.",
)


def _source_references(hits: list[RetrievalHit]) -> list[RetrievedSource]:
    by_url: dict[str, RetrievedSource] = {}
    for hit in hits:
        url = str(hit.chunk.url)
        current = by_url.get(url)
        if current is None or hit.score > current.score:
            by_url[url] = RetrievedSource(
                title=hit.chunk.title,
                url=hit.chunk.url,
                score=round(hit.score, 4),
            )
    ranked = sorted(by_url.values(), key=lambda source: source.score, reverse=True)
    return ranked[: settings.rag_max_sources]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        model=settings.openai_model,
        embedding_model=settings.embedding_model,
        openai_configured=settings.openai_api_key is not None,
        rag_index_ready=index_is_ready(settings.rag_index_dir),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = str(request.session_id)
    conversation = conversation_store.get(session_id)

    if conversation.turns >= settings.max_turns_per_session:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "This demo conversation reached its message limit. "
                "Start a new conversation to continue."
            ),
        )

    try:
        agent = get_agent()
        deps = AgentDependencies(retriever=get_retriever())
        result = await agent.run(
            request.message,
            message_history=conversation.messages,
            deps=deps,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # Keep provider details out of the public API.
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model or retrieval request failed. Please try again.",
        ) from exc

    conversation = conversation_store.save(session_id, result.all_messages())
    remaining = max(settings.max_turns_per_session - conversation.turns, 0)

    return ChatResponse(
        session_id=request.session_id,
        answer=result.output,
        sources=_source_references(deps.retrieved_hits),
        turns_used=conversation.turns,
        turns_remaining=remaining,
    )


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest) -> ResetResponse:
    conversation_store.reset(str(request.session_id))
    return ResetResponse()
