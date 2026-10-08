from backend.memory import InMemoryConversationStore


def test_store_saves_and_resets_conversation() -> None:
    store = InMemoryConversationStore(max_sessions=10)

    conversation = store.get("session-1")
    assert conversation.turns == 0
    assert conversation.messages == []

    saved = store.save("session-1", ["message"])
    assert saved.turns == 1
    assert saved.messages == ["message"]

    store.reset("session-1")
    reset = store.get("session-1")
    assert reset.turns == 0
    assert reset.messages == []


def test_store_evicts_oldest_session() -> None:
    store = InMemoryConversationStore(max_sessions=2)
    store.get("one")
    store.get("two")
    store.get("three")

    # At most two sessions are kept; asking for one again creates a fresh conversation.
    assert store.get("one").turns == 0
