from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-large"
    eval_generation_model: str = "gpt-5.6-luna"
    eval_judge_model: str = "gpt-5.6-luna"

    rag_index_dir: Path = Path("data/index")
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_min_score: float = Field(default=0.15, ge=-1.0, le=1.0)
    rag_max_chunks_per_document: int = Field(default=2, ge=1, le=10)
    rag_max_sources: int = Field(default=5, ge=1, le=10)

    max_turns_per_session: int = Field(default=30, ge=1, le=200)
    max_sessions: int = Field(default=500, ge=1, le=10_000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
