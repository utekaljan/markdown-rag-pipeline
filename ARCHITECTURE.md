# Architecture

This repository is a standalone, domain-neutral Markdown ingestion project. It is an
independent reference implementation rather than an adapter for a particular source
system, application, or retrieval product.

## Three boundaries

1. **Generic core** reads Markdown, validates stable `(source, document_id)` identities,
   chunks paragraphs, assigns deterministic UUIDs, and preserves provenance.
2. **Ingestion** always writes `chunks.jsonl` and `manifest.json` first. Live mode then
   embeds the exact prepared chunk text, creates or validates one Qdrant collection,
   and replaces each logical document atomically with delete plus upsert operations.
3. **Application integration** is intentionally outside the generic pipeline.
   Downstream systems may have their own source schemas, access controls, application
   context, and retrieval consumers. Their adapters, data models, prompts, credentials,
   and deployment configurations are outside this repository. The project boundary
   starts with ordinary Markdown and ends with generic Qdrant points.

## Core guarantees

The implementation provides stable chunk identity, paragraph-aware chunking, an
explicit embedding provider, Qdrant collection initialization, vector-dimension
validation, metadata payloads, and replace-on-reingest behavior. The interface is
deliberately small and source agnostic: Markdown is the only input contract.

## Why offline output comes first

Chunking is independently inspectable and deterministic. Reviewers can see exactly
which text and provenance would be embedded before any network call or paid API use.
The offline command has no runtime dependencies beyond Python and never creates fake
vectors. Unit tests use deterministic vectors only to exercise the Qdrant adapter; they
are test fixtures, not a production-quality semantic model and are not available from
the live CLI.

## Reingestion semantics

Chunk IDs depend on `source`, `document_id`, and `chunk_index`. Reingesting an edited
document therefore overwrites stable positions. A delete filtered by `source` and
`document_id` runs in the same ordered Qdrant batch as the new points, so shrinking a
document does not leave stale trailing chunks.
