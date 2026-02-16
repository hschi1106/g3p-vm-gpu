import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestRunCppEvolutionTool(unittest.TestCase):
    def test_tool_runs_and_writes_detailed_logs(self):
        with tempfile.TemporaryDirectory(prefix="g3p_run_cpp_evo_") as td:
            td_path = Path(td)
            cases_path = td_path / "cases.json"
            logs_dir = td_path / "logs"
            fake_cli = td_path / "fake_evolve_cli.sh"
            fake_out_json = td_path / "fake_out.json"

            cases_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"inputs": {"x": 1, "y": 2}, "expected": 3},
                            {"inputs": {"x": -1, "y": 4}, "expected": 3},
                        ]
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            fake_cli.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "out_json=\"\"\n"
                "while [[ $# -gt 0 ]]; do\n"
                "  if [[ \"$1\" == \"--out-json\" ]]; then\n"
                "    out_json=\"$2\"\n"
                "    shift 2\n"
                "    continue\n"
                "  fi\n"
                "  shift\n"
                "done\n"
                "echo 'GEN 000 best=1.000000 mean=0.250000 hash=abc123abc123abcd'\n"
                "echo 'GEN 001 best=2.000000 mean=0.750000 hash=def456def456def4'\n"
                "echo 'FINAL best=2.000000 hash=def456def456def4 selection=tournament crossover=hybrid'\n"
                "echo 'TIMING phase=init_population ms=1.234'\n"
                "echo 'TIMING phase=total ms=9.876'\n"
                "echo 'TIMING gen=000 eval_ms=3.210 repro_ms=1.230 total_ms=4.440'\n"
                "echo 'TIMING gpu_gen=000 compile_ms=0.900 pack_upload_ms=0.500 kernel_ms=1.200 copyback_ms=0.300'\n"
                "if [[ -n \"${out_json}\" ]]; then\n"
                "  cat > \"${out_json}\" <<'JSON'\n"
                "{\"history\":[{\"generation\":0}],\"final\":{\"best_fitness\":2}}\n"
                "JSON\n"
                "fi\n",
                encoding="utf-8",
            )
            fake_cli.chmod(fake_cli.stat().st_mode | stat.S_IXUSR)

            cmd = [
                "python3",
                "tools/run_cpp_evolution.py",
                "--cases",
                str(cases_path),
                "--cpp-cli",
                str(fake_cli),
                "--generations",
                "2",
                "--population-size",
                "8",
                "--log-dir",
                str(logs_dir),
                "--run-tag",
                "unittest",
            ]
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
                env={**os.environ, "PYTHONPATH": "python"},
            )

            self.assertIn("SUMMARY_JSON", proc.stdout)
            self.assertIn("TIMINGS_LOG", proc.stdout)
            self.assertIn("FINAL_METRIC best=2.000000", proc.stdout)

            summary_line = next(x for x in proc.stdout.splitlines() if x.startswith("SUMMARY_JSON "))
            summary_path = Path(summary_line.split(" ", 1)[1].strip())
            self.assertTrue(summary_path.exists())
            timing_line = next(x for x in proc.stdout.splitlines() if x.startswith("TIMINGS_LOG "))
            timing_path = Path(timing_line.split(" ", 1)[1].strip())
            self.assertTrue(timing_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("timings", summary)
            self.assertGreaterEqual(len(summary["timings"]), 5)
            self.assertIn("parsed", summary)
            self.assertEqual(len(summary["parsed"]["history"]), 2)
            self.assertEqual(summary["parsed"]["final"]["best"], 2.0)
            self.assertIn("timing_summary", summary["parsed"])
            self.assertEqual(summary["parsed"]["timing_summary"]["init_population"], 1.234)
            self.assertEqual(summary["parsed"]["timing_summary"]["total"], 9.876)
            self.assertEqual(len(summary["parsed"]["timing_per_gen"]), 1)
            self.assertEqual(len(summary["parsed"]["timing_gpu_per_gen"]), 1)

            timing_stages = {row["stage"] for row in summary["timings"]}
            self.assertIn("run_cpp_cli", timing_stages)
            self.assertIn("parse_cli_output", timing_stages)
            self.assertIn("write_artifacts", timing_stages)

            timing_text = timing_path.read_text(encoding="utf-8")
            self.assertIn("[outer_python]", timing_text)
            self.assertIn("[inner_cpp_summary]", timing_text)
            self.assertIn("[inner_cpp_per_gen]", timing_text)
            self.assertIn("[inner_cpp_gpu_per_gen]", timing_text)


if __name__ == "__main__":
    unittest.main()
