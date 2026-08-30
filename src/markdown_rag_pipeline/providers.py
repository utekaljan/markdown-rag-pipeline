from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    dimension: int
    model_name: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Production embedding provider; construction is explicit and live-only."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for production embeddings")
        if dimension < 8:
            raise ValueError("Embedding dimension must be at least 8")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Live mode needs optional dependencies: pip install -e '.[qdrant]'"
            ) from error

        self._client = OpenAI(api_key=api_key)
        self.model_name = model
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.model_name,
            input=list(texts),
            dimensions=self.dimension,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]
