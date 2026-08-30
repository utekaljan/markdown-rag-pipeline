# Markdown RAG Pipeline

Turn Markdown documents into a retrieval-ready Qdrant collection. The pipeline keeps the
whole path inspectable: it reads your Markdown files, preserves their provenance while
splitting them into chunks, creates embeddings when credentials are configured, and
upserts the resulting vectors into Qdrant.

It can also stop after the local preparation step. That makes the exact payload intended
for embedding and upload visible before any cloud call.

The checked-in `examples/synthetic-document.md` is a first-party fixture whose
metadata and prose were written only for this repository. It contains no source text
or identifiers copied from an external or private project.

**The complete flow is:**

```text
input/*.md  ->  offline chunking  ->  output/chunks.jsonl + output/manifest.json
                                           |
                                           +-> optional OpenAI embeddings
                                               -> create/check Qdrant collection
                                               -> replace document vectors
```

`ingest` is the credential-free preparation mode. It reads Markdown and produces
inspectable chunks plus a provenance manifest. `upsert` performs that same preparation,
creates real OpenAI embeddings, and uploads the vectors to Qdrant. Nothing in `ingest`
pretends to be a semantic embedding.

## 1. Prepare Markdown for retrieval

Python 3.11 or newer is required. Offline mode has no third-party runtime dependency.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
markdown-rag ingest
```

Put one or more `.md` files anywhere under `input/`, then run `ingest`. The public
repository keeps `input/` empty so you always provide your own documents.

To verify a fresh clone without adding a document, run the same path against the
synthetic fixture:

```bash
markdown-rag ingest --input examples --output /tmp/markdown-rag-output
```

Open these two generated files:

- `output/chunks.jsonl` — one full chunk per line, including stable ID, text hash,
  source, document ID, chunk index, title, relative source path, and metadata;
- `output/manifest.json` — chunking configuration, input hashes, document-to-chunk
  mapping, and totals.

Front matter is optional:

```markdown
---
document_id: stable-article-id
source: my-handbook
title: Human-readable title
---

# Content starts here
```

`source` plus `document_id` must be unique. If front matter is absent, the filename is
the document ID and `local-markdown` is the source. Paragraphs stay together when they
fit; long paragraphs split on word boundaries. Change the limit with
`--chunk-size 1000`.

## 2. Create embeddings and upload to Qdrant

Install the live dependencies, create local configuration, and start Qdrant locally or
point to a hosted instance:

```bash
python -m pip install -e ".[qdrant]"
cp .env.example .env
docker compose up -d
```

Set `OPENAI_API_KEY` in `.env`. Set `QDRANT_URL` to your Qdrant instance and
`QDRANT_COLLECTION` to the collection name. A local Qdrant instance normally needs no
`QDRANT_API_KEY`; set it only for a hosted Qdrant service that requires one. The
`.env.example` file shows every supported setting and `.env` is ignored by Git.

Then validate the setup without printing either key:

```bash
markdown-rag preflight
```

Preflight checks required settings, Qdrant connectivity, collection existence, and an
existing collection's vector dimension. It does not create the collection or call the
embedding API. Upload only after reviewing the offline artifacts:

```bash
markdown-rag upsert
```

`upsert` rebuilds the same artifacts, requests embeddings from the configured OpenAI
model, creates the Qdrant collection if missing, checks its dimension if present, and
replaces each logical document.

Changing the embedding model or dimension requires a new empty collection (or a matching
existing one). Reusing `(source, document_id)` intentionally replaces that document;
changing either value creates a separate logical document.

## Test

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

Tests cover deterministic chunk identities, provenance, the complete offline command,
secret-safe configuration summaries, Qdrant collection creation, vector upload, and
replace-on-shrink behavior. The Qdrant integration test uses the client's in-memory
transport and deterministic fixture vectors. Those vectors exist only to test storage
behavior and are not exposed as a usable embedding mode.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the exact document-identity and reingestion
semantics.

## Security

Report suspected vulnerabilities through GitHub's private vulnerability reporting
form. Do not open a public discussion or include credentials or private documents in a
report. GitHub Issues are intentionally disabled for this repository. See
[SECURITY.md](SECURITY.md) for the reporting policy.

## License

Copyright (c) 2026 Jan Utěkal. This project is licensed under the
[MIT License](LICENSE). The first-party synthetic fixture is included under the
same terms.
