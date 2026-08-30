from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from markdown_rag_pipeline.config import Settings


class ConfigTests(TestCase):
    def test_env_file_is_loaded_and_summary_never_contains_keys(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "QDRANT_URL=https://example.invalid\n"
                "QDRANT_API_KEY=qdrant-secret-value\n"
                "QDRANT_COLLECTION=test_collection\n"
                "OPENAI_API_KEY=openai-secret-value\n"
                "OPENAI_EMBEDDING_MODEL=text-embedding-3-small\n"
                "OPENAI_EMBEDDING_DIMENSION=1536\n", encoding="utf-8",
            )
            settings = Settings.load(env_file)
            serialized = json.dumps(settings.safe_summary())
            self.assertEqual(settings.validate_live(), [])
            self.assertNotIn("qdrant-secret-value", serialized)
            self.assertNotIn("openai-secret-value", serialized)
            self.assertIs(settings.safe_summary()["qdrant_api_key_configured"], True)
            self.assertIs(settings.safe_summary()["openai_api_key_configured"], True)

    def test_live_validation_requires_real_embedding_key(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            settings = Settings.load(Path(directory) / "missing.env")
        self.assertIn(
            "OPENAI_API_KEY is required for production embeddings",
            settings.validate_live(),
        )
