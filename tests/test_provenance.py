from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lib.organism.provenance import ExperimentProvenance, sha256_file


class ProvenanceTests(unittest.TestCase):
    def test_sha256_file_is_stable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vector.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223"
                "b00361a396177a9cb410ff61f20015ad",
            )

    def test_provenance_serializes_required_fields(self):
        record = ExperimentProvenance(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            model_revision="deadbeef",
            tokenizer_revision="deadbeef",
            dtype="bfloat16",
            backend="transformers",
            vector_source="Pain Axis",
            vector_sha256="abc123",
            layer=20,
            dose_ratio=0.5,
            seed=42,
            prompt_id="neutral-001",
            code_commit="cafebabe",
            hardware="test-gpu",
        )
        blob = record.to_dict()
        self.assertEqual(blob["layer"], 20)
        self.assertEqual(blob["dose_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
