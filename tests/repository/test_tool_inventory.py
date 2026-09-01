from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestToolInventory(unittest.TestCase):
    def test_every_top_level_tool_is_classified(self) -> None:
        inventory = (ROOT / "docs" / "reference" / "tooling.md").read_text(
            encoding="utf-8"
        )
        scripts = sorted((ROOT / "tools").glob("*.py"))
        self.assertTrue(scripts)
        for script in scripts:
            self.assertIn(f"`{script.name}`", inventory)

    def test_tool_package_has_one_dependency_free_entrypoint(self) -> None:
        pyproject = (ROOT / "tools" / "pyproject.toml").read_text(encoding="utf-8")
        setup_cfg = (ROOT / "tools" / "setup.cfg").read_text(encoding="utf-8")
        self.assertIn('g3pvm-tools = "g3pvm_tools.cli:main"', pyproject)
        self.assertIn("g3pvm-tools = g3pvm_tools.cli:main", setup_cfg)
        self.assertIn("dependencies = []", pyproject)
        expected_modules = {
            "datasets/convert_psb.py",
            "datasets/fetch_psb.py",
            "datasets/materialize_psb.py",
            "experiments/grammar_profiles.py",
            "experiments/population_seeds.py",
            "experiments/run_psb.py",
            "reports/compare_psb.py",
            "reports/psb_manifest.py",
            "reports/simple_manifest.py",
            "shared/hashing.py",
            "shared/json_io.py",
            "shared/metrics.py",
            "shared/schemas.py",
        }
        package = ROOT / "tools" / "g3pvm_tools"
        self.assertTrue(all((package / path).is_file() for path in expected_modules))

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
