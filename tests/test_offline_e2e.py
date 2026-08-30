from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from markdown_rag_pipeline.cli import main


class OfflineEndToEndTests(TestCase):
    def test_offline_command_writes_inspectable_deterministic_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_directory = root / "input"
            output_directory = root / "output"
            input_directory.mkdir()
            (input_directory / "sample.md").write_text(
                "---\ndocument_id: sample\nsource: test\ntitle: Sample\n---\n\n"
                "First paragraph.\n\nSecond paragraph.", encoding="utf-8",
            )
            arguments = [
                "ingest", "--input", str(input_directory),
                "--output", str(output_directory), "--chunk-size", "80",
            ]
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(arguments), 0)
            first_chunks = (output_directory / "chunks.jsonl").read_text(encoding="utf-8")
            first_manifest = (output_directory / "manifest.json").read_text(encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)

            self.assertEqual(
                (output_directory / "chunks.jsonl").read_text(encoding="utf-8"), first_chunks
            )
            self.assertEqual(
                (output_directory / "manifest.json").read_text(encoding="utf-8"), first_manifest
            )
            manifest = json.loads(first_manifest)
            record = json.loads(first_chunks.splitlines()[0])
            self.assertEqual(manifest["summary"], {"chunks": 1, "documents": 1})
            self.assertEqual(manifest["documents"][0]["path"], "sample.md")
            self.assertEqual(record["source"], "test")
            self.assertEqual(record["relative_path"], "sample.md")
            self.assertNotIn(str(root), first_manifest)
            self.assertNotIn(str(root), first_chunks)
            self.assertIn("Prepared 1 chunks", stdout.getvalue())
