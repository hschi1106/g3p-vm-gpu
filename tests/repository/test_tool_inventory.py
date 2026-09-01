from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestToolInventory(unittest.TestCase):
    def test_every_top_level_tool_is_classified(self) -> None:
        inventory = (ROOT / "docs" / "TOOLING_INVENTORY.md").read_text(encoding="utf-8")
        scripts = sorted((ROOT / "tools").glob("*.py"))
        self.assertTrue(scripts)
        for script in scripts:
            self.assertIn(f"`{script.name}`", inventory)

    def test_legacy_plot_surface_is_absent(self) -> None:
        self.assertFalse((ROOT / "draw").exists())
        for directory in (ROOT / "tools", ROOT / "docs"):
            for path in directory.rglob("*"):
                if path.suffix not in {".py", ".md", ".toml", ".txt"}:
                    continue
                text = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("import matplotlib", text, str(path))
                self.assertNotIn("import seaborn", text, str(path))

    def test_auxiliary_native_targets_are_opt_in(self) -> None:
        cmake = (ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('option(G3PVM_BUILD_BENCHMARKS', cmake)
        self.assertIn('option(G3PVM_BUILD_EXPERIMENTS', cmake)
        self.assertIn("if(G3PVM_BUILD_BENCHMARKS)", cmake)
        self.assertIn("if(G3PVM_BUILD_EXPERIMENTS)", cmake)
        self.assertTrue(
            (ROOT / "cpp" / "src" / "experiments" / "simple_exp_population_probe.cpp").is_file()
        )
        self.assertFalse(
            (ROOT / "cpp" / "tests" / "parity" / "simple_exp_population_probe.cpp").exists()
        )


if __name__ == "__main__":
    unittest.main()
