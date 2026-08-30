from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    openai_api_key: str
    openai_embedding_model: str
    openai_embedding_dimension: int

    @classmethod
    def load(cls, env_file: Path = Path(".env")) -> Settings:
        values = _read_env_file(env_file)

        def get(name: str, default: str = "") -> str:
            return os.environ.get(name, values.get(name, default)).strip()

        raw_dimension = get("OPENAI_EMBEDDING_DIMENSION", "1536")
        try:
            dimension = int(raw_dimension)
        except ValueError as error:
            raise ValueError("OPENAI_EMBEDDING_DIMENSION must be an integer") from error
        return cls(
            qdrant_url=get("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=get("QDRANT_API_KEY"),
            qdrant_collection=get("QDRANT_COLLECTION", "markdown_knowledge"),
            openai_api_key=get("OPENAI_API_KEY"),
            openai_embedding_model=get(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_embedding_dimension=dimension,
        )

    def validate_live(self) -> list[str]:
        errors: list[str] = []
        if not self.qdrant_url:
            errors.append("QDRANT_URL is required")
        if not self.qdrant_collection:
            errors.append("QDRANT_COLLECTION is required")
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for production embeddings")
        if not self.openai_embedding_model:
            errors.append("OPENAI_EMBEDDING_MODEL is required")
        if self.openai_embedding_dimension < 8:
            errors.append("OPENAI_EMBEDDING_DIMENSION must be at least 8")
        return errors

    def safe_summary(self) -> dict[str, str | int | bool]:
        return {
            "qdrant_url": self.qdrant_url,
            "qdrant_api_key_configured": bool(self.qdrant_api_key),
            "qdrant_collection": self.qdrant_collection,
            "embedding_provider": "OpenAI",
            "openai_api_key_configured": bool(self.openai_api_key),
            "embedding_model": self.openai_embedding_model,
            "embedding_dimension": self.openai_embedding_dimension,
        }


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid .env line {line_number}")
        normalized = value.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in "\"'":
            normalized = normalized[1:-1]
        values[key.strip()] = normalized
    return values
