from functools import lru_cache

from backend.config import get_settings
from rag.embeddings import create_openai_embedder
from rag.index import LocalVectorIndex


@lru_cache
def get_retriever() -> LocalVectorIndex:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    embedder = create_openai_embedder(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=settings.embedding_model,
    )
    index = LocalVectorIndex(
        index_dir=settings.rag_index_dir,
        embedder=embedder,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
        max_chunks_per_document=settings.rag_max_chunks_per_document,
    )
    if index.metadata.embedding_model != settings.embedding_model:
        raise ValueError(
            "Configured EMBEDDING_MODEL does not match the built index. "
            "Run scripts/build_index.py again."
        )
    return index
