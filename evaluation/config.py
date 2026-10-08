from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkingSetting:
    name: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class EvaluationConfig:
    embedding_model: str
    chunking: ChunkingSetting

    @property
    def slug(self) -> str:
        model = self.embedding_model.replace("/", "-")
        return f"{model}__{self.chunking.name}"


EMBEDDING_MODELS = (
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
)

CHUNKING_SETTINGS = (
    ChunkingSetting(name="small", chunk_size=800, chunk_overlap=100),
    ChunkingSetting(name="medium", chunk_size=1600, chunk_overlap=250),
    ChunkingSetting(name="large", chunk_size=2400, chunk_overlap=400),
)

GRID = tuple(
    EvaluationConfig(embedding_model=model, chunking=chunking)
    for model in EMBEDDING_MODELS
    for chunking in CHUNKING_SETTINGS
)

DEFAULT_DATASET_PATH = Path("evaluation/dataset.jsonl")
DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_INDEX_ROOT = Path("data/evaluation/indexes")
DEFAULT_DOCUMENTS_PATH = Path("data/processed/documents.jsonl")
