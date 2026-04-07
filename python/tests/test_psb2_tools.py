import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestPsb2Tools(unittest.TestCase):
    def test_unified_convert_with_psb2_basic(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_conv_") as td:
            td_path = Path(td)
            edge = td_path / "edge.json"
            rnd = td_path / "random.json"
            out = td_path / "train.json"
            out_test = td_path / "test.json"
            summary = td_path / "summary.json"

            edge.write_text(
                '{"input1":1,"input2":2.5,"output1":3}\n'
                '{"input1":2,"input2":3.0,"output1":5}\n',
                encoding="utf-8",
            )
            rnd.write_text(
                '{"input1":4,"input2":1.0,"output1":5}\n'
                '{"input1":5,"input2":2.0,"output1":7}\n',
                encoding="utf-8",
            )

            cmd = [
                "python3",
                "tools/convert_psb_to_fitness_cases.py",
                "--suite",
                "psb2",
                "--edge-file",
                str(edge),
                "--random-file",
                str(rnd),
                "--n-train",
                "3",
                "--n-test",
                "2",
                "--seed",
                "7",
                "--out",
                str(out),
                "--out-test",
                str(out_test),
                "--summary-json",
                str(summary),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("CONVERT_RUNTIME_COMPATIBLE 1", proc.stdout)
            train = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(train["format_version"], "fitness-cases-v1")
            self.assertEqual(len(train["cases"]), 3)
            one = train["cases"][0]
            self.assertIn("inputs", one)
            self.assertEqual(sorted(one["inputs"].keys()), ["input1", "input2"])
            self.assertIn("expected", one)
            self.assertIn("type", one["expected"])
            s = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(s["runtime_compatible"])

    def test_convert_multi_output_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_multiout_") as td:
            td_path = Path(td)
            edge = td_path / "edge.json"
            rnd = td_path / "random.json"
            out = td_path / "train.json"
            edge.write_text('{"input1":1,"output1":2,"output2":3}\n', encoding="utf-8")
            rnd.write_text('{"input1":1,"output1":2}\n', encoding="utf-8")

            cmd = [
                "python3",
                "tools/convert_psb_to_fitness_cases.py",
                "--suite",
                "psb2",
                "--edge-file",
                str(edge),
                "--random-file",
                str(rnd),
                "--n-train",
                "1",
                "--n-test",
                "1",
                "--out",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("multi-output", proc.stderr)

    def test_convert_empty_list_uses_field_schema(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_list_schema_") as td:
            td_path = Path(td)
            edge = td_path / "edge.json"
            rnd = td_path / "random.json"
            out = td_path / "train.json"
            out_test = td_path / "test.json"
            edge.write_text(
                '{"input1":[],"output1":[]}\n'
                '{"input1":[1,2],"output1":[3]}\n',
                encoding="utf-8",
            )
            rnd.write_text(
                '{"input1":[4.5],"output1":[4.5]}\n'
                '{"input1":[],"output1":[]}\n',
                encoding="utf-8",
            )

            cmd = [
                "python3",
                "tools/convert_psb_to_fitness_cases.py",
                "--suite",
                "psb2",
                "--edge-file",
                str(edge),
                "--random-file",
                str(rnd),
                "--n-train",
                "2",
                "--n-test",
                "2",
                "--out",
                str(out),
                "--out-test",
                str(out_test),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("CONVERT_RUNTIME_COMPATIBLE 1", proc.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(all(row["inputs"]["input1"]["type"] == "num_list" for row in payload["cases"]))
            self.assertTrue(all(row["expected"]["type"] == "num_list" for row in payload["cases"]))

    def test_unified_convert_with_psb2_problem_to_fitness_cases(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_unified_conv_") as td:
            td_path = Path(td)
            problem_dir = td_path / "dice-game"
            problem_dir.mkdir(parents=True)
            edge = problem_dir / "dice-game-edge.json"
            rnd = problem_dir / "dice-game-random.json"
            out = td_path / "train.json"
            summary = td_path / "summary.json"

            edge.write_text(
                '{"input1":1,"input2":2,"output1":0.0}\n',
                encoding="utf-8",
            )
            rnd.write_text(
                '{"input1":2,"input2":1,"output1":0.5}\n'
                '{"input1":3,"input2":2,"output1":0.5}\n',
                encoding="utf-8",
            )

            cmd = [
                "python3",
                "tools/convert_psb_to_fitness_cases.py",
                "--suite",
                "psb2",
                "--problem",
                "dice-game",
                "--datasets-root",
                str(td_path),
                "--n-train",
                "2",
                "--n-test",
                "1",
                "--seed",
                "3",
                "--out",
                str(out),
                "--summary-json",
                str(summary),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("CONVERT_SUITE psb2", proc.stdout)
            self.assertIn("CONVERT_PROBLEM dice-game", proc.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"]["suite"], "psb2")
            self.assertEqual(payload["source"]["problem"], "dice-game")
            self.assertEqual(len(payload["cases"]), 2)
            s = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(s["runtime_compatible"])

if __name__ == "__main__":
    unittest.main()
