from __future__ import annotations

from collections.abc import Sequence
from unittest import TestCase

from qdrant_client import QdrantClient

from markdown_rag_pipeline.models import Chunk, MarkdownDocument
from markdown_rag_pipeline.qdrant_store import QdrantChunkStore, upload_chunks


class FixtureEmbedder:
    dimension = 8
    model_name = "deterministic-test-fixture"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for index, text in enumerate(texts):
            vector = [0.0] * self.dimension
            vector[(len(text) + index) % self.dimension] = 1.0
            vectors.append(vector)
        return vectors


def chunks_for(texts: list[str]) -> list[Chunk]:
    document = MarkdownDocument(
        document_id="doc", source="test", title="Test",
        text="\n\n".join(texts), relative_path="doc.md", metadata={"title": "Test"},
    )
    return [
        Chunk.create(document=document, chunk_index=index, text=text)
        for index, text in enumerate(texts)
    ]


class QdrantStoreTests(TestCase):
    def test_collection_creation_upload_and_replace_on_shrink(self) -> None:
        client = QdrantClient(location=":memory:")
        store = QdrantChunkStore(
            client=client, collection_name="test_chunks",
            vector_dimension=FixtureEmbedder.dimension,
        )
        self.assertEqual(
            store.connection_summary(), {"reachable": True, "collection_exists": False}
        )
        self.assertEqual(
            upload_chunks(
                chunks=chunks_for(["First.", "Second."]),
                embedder=FixtureEmbedder(), store=store,
            ), 2,
        )
        self.assertEqual(client.count("test_chunks", exact=True).count, 2)
        self.assertIs(store.connection_summary()["dimension_matches"], True)

        self.assertEqual(
            upload_chunks(
                chunks=chunks_for(["Updated only."]),
                embedder=FixtureEmbedder(), store=store,
            ), 1,
        )
        self.assertEqual(client.count("test_chunks", exact=True).count, 1)
        points, _ = client.scroll(
            "test_chunks", limit=10, with_payload=True, with_vectors=False
        )
        self.assertEqual(points[0].payload["text"], "Updated only.")
