from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .documents import chunk_documents, load_documents
from .models import Chunk


CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"


def build_artifacts(
    *,
    input_directory: Path,
    output_directory: Path,
    max_characters: int = 700,
) -> tuple[list[Chunk], dict[str, Any]]:
    documents = load_documents(input_directory)
    chunks = chunk_documents(documents, max_characters=max_characters)
    output_directory.mkdir(parents=True, exist_ok=True)

    chunk_path = output_directory / CHUNKS_FILENAME
    with chunk_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(chunk.to_record(), ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    chunks_by_identity: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_identity.setdefault((chunk.source, chunk.document_id), []).append(chunk)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "offline",
        "input": input_directory.name,
        "output": output_directory.name,
        "chunking": {"algorithm": "paragraph-greedy", "max_characters": max_characters},
        "summary": {"documents": len(documents), "chunks": len(chunks)},
        "documents": [
            {
                "path": document.relative_path,
                "source": document.source,
                "document_id": document.document_id,
                "title": document.title,
                "content_sha256": document.content_sha256,
                "chunk_count": len(chunks_by_identity[(document.source, document.document_id)]),
                "chunk_ids": [
                    chunk.id
                    for chunk in chunks_by_identity[(document.source, document.document_id)]
                ],
            }
            for document in documents
        ],
    }
    (output_directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return chunks, manifest


def read_chunks(output_directory: Path) -> list[Chunk]:
    path = output_directory / CHUNKS_FILENAME
    if not path.is_file():
        raise ValueError(f"Missing offline chunks artifact: {path}")
    chunks: list[Chunk] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            chunks.append(Chunk.from_record(record))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid chunk record on line {line_number}: {error}") from error
    return chunks
