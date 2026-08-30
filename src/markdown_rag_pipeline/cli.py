from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import MANIFEST_FILENAME, build_artifacts
from .config import Settings
from .providers import OpenAIEmbedder
from .qdrant_store import QdrantChunkStore, build_qdrant_client, upload_chunks


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="maximum characters in a paragraph-aware chunk",
    )


def _live_settings(env_file: Path) -> Settings:
    settings = Settings.load(env_file)
    errors = settings.validate_live()
    if errors:
        raise RuntimeError("Invalid live configuration: " + "; ".join(errors))
    return settings


def run_ingest(*, input_directory: Path, output_directory: Path, chunk_size: int) -> int:
    chunks, manifest = build_artifacts(
        input_directory=input_directory,
        output_directory=output_directory,
        max_characters=chunk_size,
    )
    print(
        f"Prepared {len(chunks)} chunks from "
        f"{manifest['summary']['documents']} Markdown documents."
    )
    print(f"Chunks: {output_directory / 'chunks.jsonl'}")
    print(f"Manifest: {output_directory / MANIFEST_FILENAME}")
    return 0


def run_preflight(*, env_file: Path) -> int:
    settings = Settings.load(env_file)
    safe = settings.safe_summary()
    print(json.dumps({"configuration": safe}, indent=2, sort_keys=True))
    errors = settings.validate_live()
    if errors:
        raise RuntimeError("Invalid live configuration: " + "; ".join(errors))

    client = build_qdrant_client(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    store = QdrantChunkStore(
        client=client,
        collection_name=settings.qdrant_collection,
        vector_dimension=settings.openai_embedding_dimension,
    )
    connection = store.connection_summary()
    print(json.dumps({"qdrant": connection}, indent=2, sort_keys=True))
    if connection.get("dimension_matches") is False:
        raise RuntimeError(
            "Configured embedding dimension does not match the existing collection"
        )
    return 0


def run_upsert(
    *,
    input_directory: Path,
    output_directory: Path,
    chunk_size: int,
    env_file: Path,
    batch_size: int,
) -> int:
    settings = _live_settings(env_file)
    chunks, manifest = build_artifacts(
        input_directory=input_directory,
        output_directory=output_directory,
        max_characters=chunk_size,
    )
    client = build_qdrant_client(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        dimension=settings.openai_embedding_dimension,
    )
    store = QdrantChunkStore(
        client=client,
        collection_name=settings.qdrant_collection,
        vector_dimension=embedder.dimension,
    )
    count = upload_chunks(
        chunks=chunks,
        embedder=embedder,
        store=store,
        batch_size=batch_size,
    )
    manifest["mode"] = "qdrant"
    manifest["qdrant"] = {
        "collection": settings.qdrant_collection,
        "embedding_provider": "OpenAI",
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "upserted_chunks": count,
    }
    (output_directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Embedded and upserted {count} chunks into "
        f"Qdrant collection {settings.qdrant_collection!r}."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Markdown chunks, inspect provenance, and optionally upsert to Qdrant"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest", help="offline: write chunks.jsonl and manifest.json"
    )
    _paths(ingest)

    preflight = subparsers.add_parser(
        "preflight", help="validate live configuration and Qdrant connectivity"
    )
    preflight.add_argument("--env-file", type=Path, default=Path(".env"))

    upsert = subparsers.add_parser(
        "upsert", help="prepare artifacts, create/check a collection, and upload vectors"
    )
    _paths(upsert)
    upsert.add_argument("--env-file", type=Path, default=Path(".env"))
    upsert.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ingest":
            return run_ingest(
                input_directory=args.input,
                output_directory=args.output,
                chunk_size=args.chunk_size,
            )
        if args.command == "preflight":
            return run_preflight(env_file=args.env_file)
        if args.command == "upsert":
            return run_upsert(
                input_directory=args.input,
                output_directory=args.output,
                chunk_size=args.chunk_size,
                env_file=args.env_file,
                batch_size=args.batch_size,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    return 2
