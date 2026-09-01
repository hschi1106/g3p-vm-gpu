from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


class TestToolCli(unittest.TestCase):
    def run_unified(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", "-m", "g3pvm_tools", *args],
            cwd=TOOLS,
            text=True,
            capture_output=True,
        )

    def test_unified_help_lists_every_command(self) -> None:
        result = self.run_unified("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in (
            "psb fetch",
            "psb convert",
            "psb materialize",
            "psb run",
            "psb compare",
            "benchmark population-seeds",
            "report psb-manifest",
            "report simple-manifest",
            "grammar profile",
        ):
            self.assertIn(command, result.stdout)

    def test_every_unified_subcommand_dispatches_help(self) -> None:
        commands = (
            ("psb", "fetch"),
            ("psb", "convert"),
            ("psb", "materialize"),
            ("psb", "run"),
            ("psb", "compare"),
            ("benchmark", "population-seeds"),
            ("report", "psb-manifest"),
            ("report", "simple-manifest"),
            ("grammar", "profile"),
        )
        for command in commands:
            with self.subTest(command=command):
                result = self.run_unified(*command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_population_seed_wrapper_and_unified_output_match(self) -> None:
        wrapper_out = TOOLS / "tests" / ".population-wrapper.json"
        unified_out = TOOLS / "tests" / ".population-unified.json"
        try:
            common = [
                "--cases",
                "data/fixtures/simple_exp_1024.json",
                "--count",
                "3",
                "--start-seed",
                "9",
            ]
            wrapper = subprocess.run(
                ["python3", "tools/make_population_seeds.py", *common, "--out", str(wrapper_out)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            unified = self.run_unified(
                "benchmark", "population-seeds", *common, "--out", str(unified_out)
            )
            self.assertEqual(wrapper.returncode, 0, wrapper.stderr)
            self.assertEqual(unified.returncode, 0, unified.stderr)
            self.assertEqual(
                json.loads(wrapper_out.read_text(encoding="utf-8")),
                json.loads(unified_out.read_text(encoding="utf-8")),
            )
        finally:
            wrapper_out.unlink(missing_ok=True)
            unified_out.unlink(missing_ok=True)

    def test_compatibility_wrappers_are_thin(self) -> None:
        for path in sorted(TOOLS.glob("*.py")):
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertLessEqual(len(lines), 5, path.name)
            self.assertIn("g3pvm_tools", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
