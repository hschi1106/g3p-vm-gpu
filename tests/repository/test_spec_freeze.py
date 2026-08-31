import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestSpecFreeze(unittest.TestCase):
    def test_spec_freeze_hashes_match(self):
        manifest_path = ROOT / "benchmarks/spec_freeze.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["format_version"], "spec-freeze")
        self.assertEqual(manifest["status"], "frozen")
        self.assertEqual(manifest["authoritative_entrypoint"], "spec/grammar.md")

        paths = [entry["path"] for entry in manifest["specs"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("spec/grammar.md", paths)
        self.assertIn("spec/grammar.md", paths)

        for entry in manifest["specs"]:
            spec_path = ROOT / entry["path"]
            digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
            self.assertEqual(
                digest,
                entry["sha256"],
                msg=f"spec freeze hash mismatch for {entry['path']}",
            )
