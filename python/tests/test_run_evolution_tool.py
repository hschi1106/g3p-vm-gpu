import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestRunEvolutionTool(unittest.TestCase):
    def test_cli_runs_and_writes_json(self):
        with tempfile.TemporaryDirectory(prefix="g3p_run_evo_") as td:
            td_path = Path(td)
            cases_path = td_path / "cases.json"
            out_path = td_path / "summary.json"
            payload = {
                "cases": [
                    {"inputs": {"x": 1, "y": 2}, "expected": 3},
                    {"inputs": {"x": -1, "y": 4}, "expected": 3},
                    {"inputs": {"x": 3, "y": -2}, "expected": 1},
                ]
            }
            cases_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

            cmd = [
                "python3",
                "tools/run_evolution.py",
                "--cases",
                str(cases_path),
                "--population-size",
                "16",
                "--generations",
                "4",
                "--selection",
                "tournament",
                "--crossover-method",
                "hybrid",
                "--out-json",
                str(out_path),
            ]
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": "python"},
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("GEN 000", proc.stdout)
            self.assertIn("FINAL best=", proc.stdout)
            self.assertTrue(out_path.exists())

            out_payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("history", out_payload)
            self.assertEqual(len(out_payload["history"]), 4)
            self.assertIn("final", out_payload)
            self.assertIn("best_fitness", out_payload["final"])

    def test_cli_accepts_psb2_fixture_cases(self):
        with tempfile.TemporaryDirectory(prefix="g3p_run_evo_psb2_") as td:
            td_path = Path(td)
            fixture_path = td_path / "psb2_fixture.json"
            payload = {
                "bytecode_program_inputs": {
                    "format_version": "bytecode-json-v0.1",
                    "fuel": 64,
                    "programs": [],
                    "shared_cases": [
                        [
                            {"idx": 0, "value": {"type": "int", "value": 3}},
                            {"idx": 1, "value": {"type": "int", "value": 10}},
                        ],
                        [
                            {"idx": 0, "value": {"type": "int", "value": 7}},
                            {"idx": 1, "value": {"type": "int", "value": 20}},
                        ],
                    ],
                    "shared_answer": [
                        {"type": "int", "value": 3},
                        {"type": "int", "value": 7},
                    ],
                }
            }
            fixture_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

            cmd = [
                "python3",
                "tools/run_evolution.py",
                "--cases",
                str(fixture_path),
                "--cases-format",
                "psb2_fixture",
                "--input-indices",
                "1",
                "--input-names",
                "x",
                "--population-size",
                "12",
                "--generations",
                "3",
                "--selection",
                "random",
            ]
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": "python"},
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("GEN 000", proc.stdout)
            self.assertIn("FINAL best=", proc.stdout)


if __name__ == "__main__":
    unittest.main()
