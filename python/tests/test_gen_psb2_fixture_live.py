import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "gen_psb2_fitness_multi_bench_inputs.py"


class TestGenPsb2FixtureLive(unittest.TestCase):
    def test_generate_fixture_with_live_psb2_fetch(self) -> None:
        if os.environ.get("G3PVM_RUN_PSB2_LIVE") != "1":
            self.skipTest("set G3PVM_RUN_PSB2_LIVE=1 to run live psb2 fetch test")
        if importlib.util.find_spec("psb2") is None:
            self.skipTest("psb2 module not installed")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            psb2_root = tmp / "psb2_datasets"
            out_path = tmp / "fitness_multi_bench_inputs_psb2_live.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--psb2-root",
                    str(psb2_root),
                    "--out",
                    str(out_path),
                    "--task-train-count",
                    "200",
                    "--task-test-count",
                    "2000",
                    "--require-psb2-fetch",
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["meta"]["source_mode"]["bouncing-balls"], "psb2.fetch_examples")
            self.assertEqual(payload["meta"]["tasks"], ["bouncing-balls"])
            self.assertEqual(payload["meta"]["pass_programs"], 2048)
            self.assertEqual(payload["meta"]["fail_programs"], 1024)
            self.assertEqual(payload["meta"]["timeout_programs"], 1024)


if __name__ == "__main__":
    unittest.main()
