from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from markdown_rag_pipeline.documents import chunk_document, load_documents, read_markdown
from markdown_rag_pipeline.models import MarkdownDocument


class DocumentTests(TestCase):
    def test_chunk_ids_are_stable_and_document_scoped(self) -> None:
        first = MarkdownDocument(
            document_id="one", source="sample", title="One",
            text="The same paragraph.", relative_path="one.md",
            metadata={"title": "One"},
        )
        second = MarkdownDocument(
            document_id="two", source="sample", title="Two",
            text="The same paragraph.", relative_path="two.md",
            metadata={"title": "Two"},
        )
        first_id = chunk_document(first)[0].id
        self.assertEqual(first_id, chunk_document(first)[0].id)
        self.assertNotEqual(first_id, chunk_document(second)[0].id)

        updated = MarkdownDocument(
            document_id="one", source="sample", title="One",
            text="Updated paragraph.", relative_path="one.md",
            metadata={"title": "One"},
        )
        self.assertEqual(chunk_document(updated)[0].id, first_id)

    def test_long_document_is_split_without_losing_words(self) -> None:
        document = MarkdownDocument(
            document_id="long", source="sample", title="Long",
            text=" ".join(f"word-{index}" for index in range(100)),
            relative_path="long.md", metadata={"title": "Long"},
        )
        chunks = chunk_document(document, max_characters=100)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            " ".join(chunk.text for chunk in chunks).split(), document.text.split()
        )

    def test_front_matter_and_relative_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "topic"
            nested.mkdir()
            path = nested / "note.md"
            path.write_text(
                "---\ndocument_id: note-1\nsource: handbook\ntitle: Note\n---\n\nBody.",
                encoding="utf-8",
            )
            document = read_markdown(path, input_directory=root)
            self.assertEqual(document.relative_path, "topic/note.md")
            self.assertEqual(document.document_id, "note-1")
            self.assertEqual(document.source, "handbook")

    def test_duplicate_logical_identity_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("one.md", "two.md"):
                (root / name).write_text(
                    "---\ndocument_id: duplicate\nsource: same\n---\n\nBody.",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_documents(root)
