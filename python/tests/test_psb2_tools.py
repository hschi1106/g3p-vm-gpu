import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestPsb2Tools(unittest.TestCase):
    def test_convert_psb2_to_fitness_cases_basic(self):
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
                "tools/convert_psb2_to_fitness_cases.py",
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

    def test_convert_multi_output_to_list_expected(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_multiout_") as td:
            td_path = Path(td)
            edge = td_path / "edge.json"
            rnd = td_path / "random.json"
            out = td_path / "train.json"
            edge.write_text('{"input1":1,"output1":2,"output2":3}\n', encoding="utf-8")
            rnd.write_text('{"input1":1,"output1":2}\n', encoding="utf-8")

            cmd = [
                "python3",
                "tools/convert_psb2_to_fitness_cases.py",
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
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("CONVERT_RUNTIME_COMPATIBLE 1", proc.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            case0 = payload["cases"][0]
            self.assertEqual(case0["expected"]["type"], "list")
            self.assertEqual(len(case0["expected"]["value"]), 2)

    def test_run_psb2_all_tasks_mixed_status(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb2_all_") as td:
            td_path = Path(td)
            datasets = td_path / "datasets"
            num_task = datasets / "num-task"
            mo_task = datasets / "multi-out-task"
            bad_task = datasets / "bad-object-task"
            num_task.mkdir(parents=True)
            mo_task.mkdir(parents=True)
            bad_task.mkdir(parents=True)

            (num_task / "num-task-edge.json").write_text('{"input1":1,"output1":2}\n', encoding="utf-8")
            (num_task / "num-task-random.json").write_text(
                '{"input1":2,"output1":3}\n{"input1":3,"output1":4}\n',
                encoding="utf-8",
            )
            (mo_task / "multi-out-task-edge.json").write_text('{"input1":1,"output1":2,"output2":3}\n', encoding="utf-8")
            (mo_task / "multi-out-task-random.json").write_text('{"input1":4,"output1":5,"output2":6}\n', encoding="utf-8")
            (bad_task / "bad-object-task-edge.json").write_text('{"input1":{"a":1},"output1":2}\n', encoding="utf-8")
            (bad_task / "bad-object-task-random.json").write_text('{"input1":{"b":2},"output1":3}\n', encoding="utf-8")

            fake_run = td_path / "fake_run_cpp.py"
            fake_run.write_text(
                "import json,sys\n"
                "args=sys.argv[1:]\n"
                "log_dir='.'\n"
                "run_tag='x'\n"
                "for i,a in enumerate(args):\n"
                "  if a=='--log-dir': log_dir=args[i+1]\n"
                "  if a=='--run-tag': run_tag=args[i+1]\n"
                "from pathlib import Path\n"
                "ld=Path(log_dir); ld.mkdir(parents=True,exist_ok=True)\n"
                "sj=ld/f'{run_tag}.summary.json'\n"
                "ej=ld/f'{run_tag}.evolution.json'\n"
                "sj.write_text(json.dumps({'parsed':{'timing_summary':{}},'timings':[]}),encoding='utf-8')\n"
                "ej.write_text(json.dumps({'history':[{'best_fitness':1.0,'mean_fitness':0.0},{'best_fitness':2.0,'mean_fitness':1.0}], 'final':{'best_fitness':2.0}}),encoding='utf-8')\n"
                "print('SUMMARY_JSON', sj)\n"
                "print('EVOLUTION_JSON', ej)\n",
                encoding="utf-8",
            )
            fake_run.chmod(fake_run.stat().st_mode | stat.S_IXUSR)

            out_dir = td_path / "out"
            cmd = [
                "python3",
                "tools/run_psb2_all_tasks.py",
                "--datasets-root",
                str(datasets),
                "--tasks",
                "all",
                "--engine",
                "cpu",
                "--n-train",
                "1",
                "--n-test",
                "1",
                "--run-cpp-tool",
                str(fake_run),
                "--log-dir",
                str(out_dir),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, env={**os.environ, "PYTHONPATH": "python"})
            self.assertEqual(proc.returncode, 1)
            self.assertIn("PSB2_SUMMARY_JSON", proc.stdout)
            summary_path = None
            for line in proc.stdout.splitlines():
                if line.startswith("PSB2_SUMMARY_JSON "):
                    summary_path = Path(line.split(" ", 1)[1].strip())
            self.assertIsNotNone(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["tasks_total"], 3)
            self.assertEqual(summary["ok"], 2)
            self.assertEqual(summary["unsupported"], 0)
            self.assertEqual(summary["failed"], 1)
            failed = [r for r in summary["results"] if r["status"] == "failed"]
            self.assertEqual(len(failed), 1)
            self.assertIn("convert failed", failed[0]["reason"])


if __name__ == "__main__":
    unittest.main()
