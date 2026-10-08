from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider


def create_openai_embedder(*, api_key: str, model_name: str) -> Embedder:
    """Create the same configured embedder for indexing and querying."""

    model = OpenAIEmbeddingModel(
        model_name,
        provider=OpenAIProvider(api_key=api_key),
    )
    return Embedder(model)
