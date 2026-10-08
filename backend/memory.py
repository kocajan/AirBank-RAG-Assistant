from dataclasses import dataclass, field
from time import monotonic

from pydantic_ai.messages import ModelMessage


@dataclass
class Conversation:
    messages: list[ModelMessage] = field(default_factory=list)
    turns: int = 0
    updated_at: float = field(default_factory=monotonic)


class InMemoryConversationStore:
    """Small in-process conversation store for a single-instance demo."""

    def __init__(self, max_sessions: int) -> None:
        self._max_sessions = max_sessions
        self._conversations: dict[str, Conversation] = {}

    def get(self, session_id: str) -> Conversation:
        conversation = self._conversations.get(session_id)
        if conversation is None:
            self._evict_if_needed()
            conversation = Conversation()
            self._conversations[session_id] = conversation

        conversation.updated_at = monotonic()
        return conversation

    def save(self, session_id: str, messages: list[ModelMessage]) -> Conversation:
        conversation = self.get(session_id)
        conversation.messages = messages
        conversation.turns += 1
        conversation.updated_at = monotonic()
        return conversation

    def reset(self, session_id: str) -> None:
        self._conversations.pop(session_id, None)

    def clear(self) -> None:
        self._conversations.clear()

    def _evict_if_needed(self) -> None:
        if len(self._conversations) < self._max_sessions:
            return

        oldest_session_id = min(
            self._conversations,
            key=lambda key: self._conversations[key].updated_at,
        )
        self._conversations.pop(oldest_session_id, None)
