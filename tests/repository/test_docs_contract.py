import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestDocsContract(unittest.TestCase):
    def read_doc(self, path: str) -> str:
        return (ROOT / path).read_text(encoding="utf-8")

    def test_entrypoint_docs_name_current_as_current_contract(self):
        readme = self.read_doc("README.md")
        architecture = self.read_doc("docs/ARCHITECTURE.md")
        development = self.read_doc("docs/DEVELOPMENT.md")
        agents = self.read_doc("AGENTS.md")

        self.assertIn("fixture schema: `fitness-cases`", readme)
        self.assertIn("--format-version fitness-cases", readme)
        self.assertIn("These are the current invariants", architecture)
        self.assertIn("`fitness-cases` fixtures for current validation", architecture)
        self.assertIn("input fitness-cases file; current validation uses", development)
        self.assertIn("`grammar.md`", agents)
        self.assertIn("Release details are\n  recorded only in `VERSION.md`", agents)

    def test_stale_default_language_does_not_reappear(self):
        docs = {
            "README.md": self.read_doc("README.md"),
            "docs/ARCHITECTURE.md": self.read_doc("docs/ARCHITECTURE.md"),
            "docs/DEVELOPMENT.md": self.read_doc("docs/DEVELOPMENT.md"),
            "AGENTS.md": self.read_doc("AGENTS.md"),
        }
        stale_phrases = [
            "scores them against `fitness-cases`,",
            "Convert PSB1/PSB2 tasks into `fitness-cases`",
            "Treat these files as the behavioral source of truth.",
            "Historical old",
            "python/src/g3p_vm_gpu",
            "PYTHONPATH=python",
            "Python reference implementation",
        ]

        for path, text in docs.items():
            for phrase in stale_phrases:
                self.assertNotIn(phrase, text, msg=f"{path} contains stale phrase")
