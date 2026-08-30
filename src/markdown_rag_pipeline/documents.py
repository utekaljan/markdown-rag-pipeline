from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .models import Chunk, MarkdownDocument


def discover_markdown(input_directory: Path) -> list[Path]:
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    paths = sorted(path for path in input_directory.rglob("*.md") if path.is_file())
    if not paths:
        raise ValueError(f"No Markdown files found in: {input_directory}")
    return paths


def read_markdown(path: Path, *, input_directory: Path) -> MarkdownDocument:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(raw)
    relative_path = path.relative_to(input_directory).as_posix()
    document_id = metadata.get("document_id", path.stem).strip()
    source = metadata.get("source", "local-markdown").strip()
    title = metadata.get("title", path.stem.replace("-", " ").title()).strip()
    if not document_id or not source or not title:
        raise ValueError(f"Empty document_id, source, or title in {relative_path}")
    text = body.strip()
    if not text:
        raise ValueError(f"Markdown body is empty: {relative_path}")
    return MarkdownDocument(
        document_id=document_id,
        source=source,
        title=title,
        text=text,
        relative_path=relative_path,
        metadata={**metadata, "title": title},
    )


def load_documents(input_directory: Path) -> list[MarkdownDocument]:
    documents = [
        read_markdown(path, input_directory=input_directory)
        for path in discover_markdown(input_directory)
    ]
    identities: dict[tuple[str, str], str] = {}
    for document in documents:
        identity = (document.source, document.document_id)
        previous = identities.get(identity)
        if previous is not None:
            raise ValueError(
                "Duplicate (source, document_id) identity "
                f"{identity!r} in {previous} and {document.relative_path}"
            )
        identities[identity] = document.relative_path
    return documents


def chunk_document(
    document: MarkdownDocument,
    *,
    max_characters: int = 700,
) -> list[Chunk]:
    if max_characters < 80:
        raise ValueError("max_characters must be at least 80")

    paragraphs = [
        " ".join(part.split())
        for part in document.text.split("\n\n")
        if part.strip()
    ]
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_long_text(paragraph, max_characters))

    groups: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current}\n\n{piece}"
        if current and len(candidate) > max_characters:
            groups.append(current)
            current = piece
        else:
            current = candidate
    if current:
        groups.append(current)

    return [
        Chunk.create(document=document, chunk_index=index, text=text)
        for index, text in enumerate(groups)
    ]


def chunk_documents(
    documents: Iterable[MarkdownDocument],
    *,
    max_characters: int = 700,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, max_characters=max_characters)
    ]


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    normalized = raw.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized
    marker = normalized.find("\n---\n", 4)
    if marker == -1:
        raise ValueError("Markdown front matter is not closed")

    metadata: dict[str, str] = {}
    for line in normalized[4:marker].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid front-matter line: {line!r}")
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError(f"Empty front-matter key: {line!r}")
        metadata[normalized_key] = value.strip()
    return metadata, normalized[marker + 5 :]


def _split_long_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_length + extra > limit:
            pieces.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += extra
    if current:
        pieces.append(" ".join(current))
    return pieces
