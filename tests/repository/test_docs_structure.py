from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
TABLE_LINK = re.compile(r"^\| \[`[^`]+`\]\(([^)]+)\) \|", re.MULTILINE)
LAYOUT_PATH = re.compile(r"^\| `([^`]+)` \|", re.MULTILINE)


def maintained_markdown() -> list[Path]:
    roots = [ROOT / "docs", ROOT / "spec", ROOT / "tools", ROOT / "benchmarks"]
    files = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "VERSION.md"]
    for directory in roots:
        files.extend(directory.rglob("*.md"))
    return sorted(set(files))


class TestDocumentationStructure(unittest.TestCase):
    def test_all_relative_markdown_links_resolve(self) -> None:
        failures: list[str] = []
        for document in maintained_markdown():
            text = document.read_text(encoding="utf-8")
            for raw_target in MARKDOWN_LINK.findall(text):
                target = raw_target.strip().strip("<>")
                if target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                if any(character in target for character in " []{}()"):
                    continue
                path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not path_part:
                    continue
                resolved = (document.parent / path_part).resolve()
                if not resolved.exists():
                    failures.append(
                        f"{document.relative_to(ROOT)} -> {target}"
                    )
        self.assertEqual([], failures, "broken local Markdown links")

    def test_document_indexes_cover_every_maintained_document(self) -> None:
        docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        primary_targets = TABLE_LINK.findall(docs_index)
        self.assertEqual(len(primary_targets), len(set(primary_targets)))

        indexed = {(ROOT / "docs" / target).resolve() for target in primary_targets}
        for secondary in (ROOT / "spec" / "README.md", ROOT / "docs" / "refactor" / "README.md"):
            text = secondary.read_text(encoding="utf-8")
            indexed.update(
                (secondary.parent / target).resolve()
                for target in MARKDOWN_LINK.findall(text)
                if target.endswith(".md") and "/" not in target
            )

        expected = set((ROOT / "docs").rglob("*.md"))
        expected.update((ROOT / "spec").glob("*.md"))
        expected.update({ROOT / "tools" / "README.md", ROOT / "benchmarks" / "README.md"})
        expected.discard(ROOT / "docs" / "README.md")
        self.assertEqual(expected, indexed)

    def test_repository_layout_paths_exist(self) -> None:
        reference = ROOT / "docs" / "reference" / "repository-layout.md"
        paths = LAYOUT_PATH.findall(reference.read_text(encoding="utf-8"))
        self.assertGreater(len(paths), 15)
        missing = [path for path in paths if not (ROOT / path).exists()]
        self.assertEqual([], missing)

    def test_experiment_protocol_has_one_maintained_location(self) -> None:
        canonical = ROOT / "docs" / "guides" / "experiment-protocol.md"
        self.assertTrue(canonical.is_file())
        self.assertFalse((ROOT / "docs" / "EXPERIMENT_SPEC.md").exists())
        tracked_docs = [path for path in maintained_markdown() if path.name != "experiment-protocol.md"]
        for document in tracked_docs:
            text = document.read_text(encoding="utf-8")
            self.assertNotIn("docs/EXPERIMENT_SPEC.md", text, str(document))

    def test_retired_root_doc_paths_do_not_reappear(self) -> None:
        retired = (
            "docs/ARCHITECTURE.md",
            "docs/CPP_RUNTIME_PAYLOAD.md",
            "docs/DEVELOPMENT.md",
            "docs/FILE_STRUCTURE.md",
            "docs/GPU_REPRODUCTION.md",
            "docs/GRAMMAR_CONFIG.md",
            "docs/TIMING.md",
            "docs/TOOLING_INVENTORY.md",
        )
        current = [
            ROOT / "README.md",
            ROOT / "AGENTS.md",
            *ROOT.joinpath("docs").rglob("*.md"),
        ]
        for path in current:
            text = path.read_text(encoding="utf-8")
            for stale in retired:
                self.assertNotIn(stale, text, str(path))


if __name__ == "__main__":
    unittest.main()
