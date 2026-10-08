from uuid import uuid4

from fastapi.testclient import TestClient

import backend.main as main
from rag.models import Chunk, RetrievalHit


client = TestClient(main.app)


class FakeResult:
    def __init__(self, output: str, messages: list[object]) -> None:
        self.output = output
        self._messages = messages

    def all_messages(self) -> list[object]:
        return self._messages


class FakeAgent:
    def __init__(self) -> None:
        self.histories: list[list[object]] = []

    async def run(self, message: str, message_history: list[object], deps) -> FakeResult:
        self.histories.append(list(message_history))
        deps.retrieved_hits.append(
            RetrievalHit(
                chunk=Chunk(
                    id="source-1",
                    document_id="doc-1",
                    chunk_index=0,
                    title="Spořicí účet",
                    url="https://www.airbank.cz/co-vas-nejvic-zajima/sporici-ucet/",
                    source_type="html",
                    category="Naše účty",
                    text="Test source",
                ),
                score=0.91,
            )
        )
        new_messages = [*message_history, f"user:{message}", f"assistant:Echo: {message}"]
        return FakeResult(f"Echo: {message}", new_messages)


class FakeRetriever:
    pass


def setup_function() -> None:
    main.conversation_store.clear()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["openai_configured"], bool)
    assert isinstance(data["rag_index_ready"], bool)


def test_chat_passes_history_and_returns_sources(monkeypatch) -> None:
    fake_agent = FakeAgent()
    monkeypatch.setattr(main, "get_agent", lambda: fake_agent)
    monkeypatch.setattr(main, "get_retriever", lambda: FakeRetriever())
    session_id = str(uuid4())

    first = client.post(
        "/chat",
        json={"session_id": session_id, "message": "My name is Jan."},
    )
    second = client.post(
        "/chat",
        json={"session_id": session_id, "message": "What is my name?"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert fake_agent.histories[0] == []
    assert fake_agent.histories[1] == [
        "user:My name is Jan.",
        "assistant:Echo: My name is Jan.",
    ]
    assert second.json()["turns_used"] == 2
    assert second.json()["sources"][0]["title"] == "Spořicí účet"


def test_reset_removes_history(monkeypatch) -> None:
    fake_agent = FakeAgent()
    monkeypatch.setattr(main, "get_agent", lambda: fake_agent)
    monkeypatch.setattr(main, "get_retriever", lambda: FakeRetriever())
    session_id = str(uuid4())

    client.post("/chat", json={"session_id": session_id, "message": "Hello"})
    reset = client.post("/reset", json={"session_id": session_id})
    client.post("/chat", json={"session_id": session_id, "message": "Hello again"})

    assert reset.status_code == 200
    assert fake_agent.histories[-1] == []
