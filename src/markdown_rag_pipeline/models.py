from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class MarkdownDocument:
    document_id: str
    source: str
    title: str
    text: str
    relative_path: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Chunk:
    """A source fragment with a stable identity and complete provenance."""

    id: str
    source: str
    document_id: str
    title: str
    relative_path: str
    chunk_index: int
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        document: MarkdownDocument,
        chunk_index: int,
        text: str,
    ) -> Chunk:
        identity = json.dumps(
            {
                "source": document.source,
                "document_id": document.document_id,
                "chunk_index": chunk_index,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return cls(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
            source=document.source,
            document_id=document.document_id,
            title=document.title,
            relative_path=document.relative_path,
            chunk_index=chunk_index,
            text=text,
            metadata=dict(document.metadata),
        )

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["text_sha256"] = self.text_sha256
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> Chunk:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Chunk metadata must be an object")
        return cls(
            id=str(record["id"]),
            source=str(record["source"]),
            document_id=str(record["document_id"]),
            title=str(record["title"]),
            relative_path=str(record["relative_path"]),
            chunk_index=int(record["chunk_index"]),
            text=str(record["text"]),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )
