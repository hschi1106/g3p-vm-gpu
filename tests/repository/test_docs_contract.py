import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestDocsContract(unittest.TestCase):
    def read_doc(self, path: str) -> str:
        return (ROOT / path).read_text(encoding="utf-8")

    def test_entrypoints_route_semantics_to_specs(self):
        readme = self.read_doc("README.md")
        architecture = self.read_doc("docs/design/architecture.md")
        development = self.read_doc("docs/guides/development.md")
        agents = self.read_doc("AGENTS.md")

        self.assertIn("spec/README.md", readme)
        self.assertIn("specifications", architecture)
        self.assertIn("reference/cli.md", development)
        self.assertNotIn("numeric expected + numeric actual", readme)
        self.assertNotIn("Scalar builtins:", architecture)
        self.assertNotIn("Operational summary:", architecture)
        self.assertIn("`grammar.md`", agents)
        self.assertIn("Release details are\n  recorded only in `VERSION.md`", agents)

    def test_stale_default_language_does_not_reappear(self):
        docs = {
            "README.md": self.read_doc("README.md"),
            "docs/design/architecture.md": self.read_doc("docs/design/architecture.md"),
            "docs/guides/development.md": self.read_doc("docs/guides/development.md"),
            "AGENTS.md": self.read_doc("AGENTS.md"),
        }
        stale_phrases = [
            "scores them against `fitness-cases`,",
            "Convert PSB1/PSB2 tasks into `fitness-cases`",
            "Historical old",
            "python/src/gagp",
            "PYTHONPATH=python",
            "Python reference implementation",
        ]

        for path, text in docs.items():
            for phrase in stale_phrases:
                self.assertNotIn(phrase, text, msg=f"{path} contains stale phrase")

    def test_gagp_identity_is_canonical(self):
        readme = self.read_doc("README.md")
        version = self.read_doc("VERSION.md")
        cmake = self.read_doc("cpp/CMakeLists.txt")
        pyproject = self.read_doc("tools/pyproject.toml")

        self.assertTrue(readme.startswith("# GAGP\n"))
        self.assertIn(
            "GPU-Accelerated Genetic Programming for Program Synthesis",
            readme,
        )
        self.assertIn("Current release: 1.0.0", version)
        self.assertIn("project(gagp LANGUAGES CXX)", cmake)
        self.assertIn('name = "gagp-tools"', pyproject)
        self.assertTrue((ROOT / "cpp" / "include" / "gagp").is_dir())
        self.assertTrue((ROOT / "tools" / "gagp_tools").is_dir())
