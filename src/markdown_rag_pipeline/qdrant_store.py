from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .models import Chunk
from .providers import EmbeddingProvider


def build_qdrant_client(*, url: str, api_key: str) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise RuntimeError(
            "Qdrant mode needs optional dependencies: pip install -e '.[qdrant]'"
        ) from error
    return QdrantClient(url=url, api_key=api_key or None, timeout=60)


class QdrantChunkStore:
    """Qdrant storage adapter for the domain-neutral Markdown pipeline."""

    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        vector_dimension: int,
    ) -> None:
        try:
            from qdrant_client import models
        except ImportError as error:
            raise RuntimeError(
                "Qdrant mode needs optional dependencies: pip install -e '.[qdrant]'"
            ) from error
        self._models = models
        self.client = client
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

    def connection_summary(self) -> dict[str, object]:
        collections = self.client.get_collections().collections
        names = {collection.name for collection in collections}
        summary: dict[str, object] = {
            "reachable": True,
            "collection_exists": self.collection_name in names,
        }
        if self.collection_name in names:
            summary["collection_dimension"] = self._existing_dimension()
            summary["dimension_matches"] = (
                summary["collection_dimension"] == self.vector_dimension
            )
        return summary

    def ensure_collection(self) -> str:
        if self.client.collection_exists(self.collection_name):
            existing_dimension = self._existing_dimension()
            if existing_dimension != self.vector_dimension:
                raise ValueError(
                    f"Collection {self.collection_name!r} has vector dimension "
                    f"{existing_dimension}; configured embeddings use {self.vector_dimension}. "
                    "Choose a new collection or restore the matching dimension."
                )
            return "existing"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._models.VectorParams(
                size=self.vector_dimension,
                distance=self._models.Distance.COSINE,
            ),
        )
        return "created"

    def replace_document(
        self,
        *,
        source: str,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector")
        points: list[Any] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if (chunk.source, chunk.document_id) != (source, document_id):
                raise ValueError("All chunks must belong to the replaced document")
            if len(vector) != self.vector_dimension:
                raise ValueError(
                    f"Expected vector dimension {self.vector_dimension}, got {len(vector)}"
                )
            points.append(
                self._models.PointStruct(
                    id=chunk.id,
                    vector=list(vector),
                    payload={
                        "source": chunk.source,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "relative_path": chunk.relative_path,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "metadata": dict(chunk.metadata),
                        "text_sha256": chunk.text_sha256,
                    },
                )
            )

        operations: list[Any] = [
            self._models.DeleteOperation(
                delete=self._models.FilterSelector(
                    filter=self._models.Filter(
                        must=[
                            self._models.FieldCondition(
                                key="source",
                                match=self._models.MatchValue(value=source),
                            ),
                            self._models.FieldCondition(
                                key="document_id",
                                match=self._models.MatchValue(value=document_id),
                            ),
                        ]
                    )
                )
            )
        ]
        if points:
            operations.append(
                self._models.UpsertOperation(
                    upsert=self._models.PointsList(points=points)
                )
            )
        self.client.batch_update_points(
            collection_name=self.collection_name,
            update_operations=operations,
            wait=True,
        )

    def _existing_dimension(self) -> int:
        collection = self.client.get_collection(self.collection_name)
        vectors = collection.config.params.vectors
        size = getattr(vectors, "size", None)
        if not isinstance(size, int):
            raise ValueError(
                "Only Qdrant collections with one unnamed dense vector are supported"
            )
        return size


def upload_chunks(
    *,
    chunks: Sequence[Chunk],
    embedder: EmbeddingProvider,
    store: QdrantChunkStore,
    batch_size: int = 64,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    store.ensure_collection()
    grouped: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[(chunk.source, chunk.document_id)].append(chunk)

    total = 0
    for (source, document_id), document_chunks in grouped.items():
        vectors: list[list[float]] = []
        for offset in range(0, len(document_chunks), batch_size):
            batch = document_chunks[offset : offset + batch_size]
            vectors.extend(embedder.embed_texts([chunk.text for chunk in batch]))
        store.replace_document(
            source=source,
            document_id=document_id,
            chunks=document_chunks,
            vectors=vectors,
        )
        total += len(document_chunks)
    return total
