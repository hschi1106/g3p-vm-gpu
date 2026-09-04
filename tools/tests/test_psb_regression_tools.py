import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_summary(path: Path, runs):
    payload = {
        "format_version": "psb-regression-summary",
        "metadata": {"suite": "psb1", "profile": "test"},
        "aggregate": {},
        "runs": runs,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def native_cli_ready() -> bool:
    binary = ROOT / "cpp/build/gagp_evolve_cli"
    source = ROOT / "cpp/src/cli/evolve_cli.cpp"
    return binary.exists() and (not source.exists() or source.stat().st_mtime <= binary.stat().st_mtime)


class TestPsbRegressionTools(unittest.TestCase):
    def test_run_psb_regression_dry_run_writes_summary(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_regression_") as td:
            td_path = Path(td)
            cases_root = td_path / "cases"
            cases_root.mkdir()
            fixture = {
                "format_version": "fitness-cases",
                "meta": {"schema_hash": "sha256:test-schema"},
                "schema": {"inputs": {"input1": "int_list"}, "expected": "int"},
                "cases": [
                    {
                        "inputs": {"input1": {"type": "int_list", "value": [1, 2]}},
                        "expected": {"type": "int", "value": 2},
                    },
                    {
                        "inputs": {"input1": {"type": "int_list", "value": [2, 4]}},
                        "expected": {"type": "int", "value": 0},
                    },
                ],
            }
            (cases_root / "count-odds.train.json").write_text(json.dumps(fixture), encoding="utf-8")
            out_dir = td_path / "out"

            cmd = [
                "python3",
                "tools/run_psb_regression.py",
                "--suite",
                "psb1",
                "--cases-root",
                str(cases_root),
                "--problems",
                "count-odds",
                "--seeds",
                "0,1",
                "--engine",
                "gpu",
                "--repro-backend",
                "gpu",
                "--repro-overlap",
                "on",
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("PSB_SUMMARY", proc.stdout)

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["format_version"], "psb-regression-summary")
            self.assertEqual(summary["metadata"]["repro_overlap"], True)
            self.assertEqual(summary["aggregate"]["dry_runs"], 2)
            self.assertEqual(len(summary["runs"]), 2)
            self.assertEqual(summary["runs"][0]["target"]["target_fitness"], 0.0)
            self.assertEqual(summary["runs"][0]["schema_hash"], "sha256:test-schema")
            self.assertEqual(summary["runs"][0]["target"]["format_version"], "fitness-cases")
            self.assertEqual(summary["runs"][0]["status"], "dry_run")
            self.assertFalse((out_dir / "count-odds" / "seed_0" / "run.json").exists())

    def test_run_psb_regression_generates_compat_config(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_compat_") as td:
            td_path = Path(td)
            cases_root = td_path / "cases"
            cases_root.mkdir()
            fixture = {
                "format_version": "fitness-cases",
                "meta": {"schema_hash": "sha256:test-schema"},
                "schema": {"inputs": {"input1": "int_list"}, "expected": "int"},
                "cases": [
                    {
                        "inputs": {"input1": {"type": "int_list", "value": [1, 2, 3]}},
                        "expected": {"type": "int", "value": 2},
                    }
                ],
            }
            (cases_root / "count-odds.train.json").write_text(json.dumps(fixture), encoding="utf-8")
            out_dir = td_path / "out"

            cmd = [
                "python3",
                "tools/run_psb_regression.py",
                "--suite",
                "psb1",
                "--profile",
                "compat",
                "--base-grammar-config",
                "configs/grammar/num_list.json",
                "--cases-root",
                str(cases_root),
                "--problems",
                "count-odds",
                "--seeds",
                "0",
                "--engine",
                "cpu",
                "--repro-backend",
                "cpu",
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ]
            subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            generated = summary["metadata"]["generated_grammar_configs"]["count-odds"]
            generated_path = ROOT / generated["path"]
            self.assertTrue(generated_path.exists())
            profile = json.loads(generated_path.read_text(encoding="utf-8"))
            self.assertEqual(profile["format_version"], "grammar-config")
            self.assertEqual(profile["compat"]["fixture_schema_hash"], "sha256:test-schema")
            self.assertTrue(profile["values"]["int_list"])
            self.assertFalse(profile["values"]["float_list"])
            self.assertFalse(profile["values"]["char"])
            run = summary["runs"][0]
            self.assertEqual(run["grammar_config"]["kind"], "generated_compat")
            self.assertIn("--grammar-config", run["command"])
            self.assertIn(generated["path"], run["command"])

    def test_run_psb_regression_generates_compact_config_with_old_num_list_shape(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_compact_") as td:
            td_path = Path(td)
            cases_root = td_path / "cases"
            cases_root.mkdir()
            fixture = {
                "format_version": "fitness-cases",
                "meta": {"schema_hash": "sha256:test-schema"},
                "schema": {"inputs": {"input1": "int_list"}, "expected": "int"},
                "cases": [
                    {
                        "inputs": {"input1": {"type": "int_list", "value": [1, 2, 3]}},
                        "expected": {"type": "int", "value": 2},
                    }
                ],
            }
            (cases_root / "count-odds.train.json").write_text(json.dumps(fixture), encoding="utf-8")
            out_dir = td_path / "out"

            cmd = [
                "python3",
                "tools/run_psb_regression.py",
                "--suite",
                "psb1",
                "--profile",
                "compact",
                "--base-grammar-config",
                "configs/grammar/num_list.json",
                "--cases-root",
                str(cases_root),
                "--problems",
                "count-odds",
                "--seeds",
                "0",
                "--engine",
                "cpu",
                "--repro-backend",
                "cpu",
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ]
            subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            generated = summary["metadata"]["generated_grammar_configs"]["count-odds"]
            profile = json.loads((ROOT / generated["path"]).read_text(encoding="utf-8"))
            self.assertEqual(profile["profile"], "compact")
            self.assertEqual(profile["compat"]["num_list_mode"], "both")
            self.assertTrue(profile["values"]["int_list"])
            self.assertTrue(profile["values"]["float_list"])
            self.assertEqual(summary["runs"][0]["grammar_config"]["kind"], "generated_compact")

    def test_make_population_seeds_writes_replayable_seed_set(self):
        with tempfile.TemporaryDirectory(prefix="g3p_population_seeds_") as td:
            td_path = Path(td)
            out = td_path / "population.seeds.json"
            cmd = [
                "python3",
                "tools/make_population_seeds.py",
                "--cases",
                "data/fixtures/simple_exp_1024.json",
                "--grammar-config",
                "configs/grammar/scalar.json",
                "--count",
                "3",
                "--start-seed",
                "10",
                "--stride",
                "5",
                "--out",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("POPULATION_SEEDS_COUNT 3", proc.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["format_version"], "population-seeds")
            self.assertEqual(payload["cases_path"], "data/fixtures/simple_exp_1024.json")
            self.assertEqual([row["seed"] for row in payload["seeds"]], [10, 15, 20])
            self.assertEqual(payload["limits"]["max_expr_depth"], 7)
            self.assertEqual(payload["grammar_config"]["path"], "configs/grammar/scalar.json")
            self.assertTrue(payload["grammar_config"]["hash"].startswith("fnv1a64:"))

    def test_materialize_psb_fixtures_writes_manifest_with_exclusions(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_materialize_") as td:
            td_path = Path(td)
            datasets_root = td_path / "datasets"
            count_dir = datasets_root / "count-odds"
            count_dir.mkdir(parents=True)
            count_dir.joinpath("count-odds-edge.json").write_text(
                '{"input1":[],"output1":0}\n'
                '{"input1":[1,2,3],"output1":2}\n',
                encoding="utf-8",
            )
            count_dir.joinpath("count-odds-random.json").write_text(
                '{"input1":[4,5],"output1":1}\n',
                encoding="utf-8",
            )
            multi_dir = datasets_root / "replace-space-with-newline"
            multi_dir.mkdir(parents=True)
            multi_dir.joinpath("replace-space-with-newline-edge.json").write_text(
                '{"input1":"a b","output1":"a\\nb","output2":1}\n',
                encoding="utf-8",
            )
            multi_dir.joinpath("replace-space-with-newline-random.json").write_text(
                '{"input1":"c d","output1":"c\\nd","output2":1}\n',
                encoding="utf-8",
            )
            out_dir = td_path / "fixtures"

            cmd = [
                "python3",
                "tools/materialize_psb_fixtures.py",
                "--suite",
                "psb1",
                "--format-version",
                "fitness-cases",
                "--datasets-root",
                str(datasets_root),
                "--problems",
                "count-odds,replace-space-with-newline",
                "--n-train",
                "2",
                "--n-test",
                "1",
                "--out-dir",
                str(out_dir),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("PSB_FIXTURES_OK 1", proc.stdout)
            self.assertIn("PSB_FIXTURES_FAILED 1", proc.stdout)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format_version"], "psb-fixtures-manifest")
            self.assertEqual(manifest["fixture_format_version"], "fitness-cases")
            self.assertEqual(manifest["ok_problems"], ["count-odds"])
            self.assertEqual(
                manifest["excluded_problems"]["replace-space-with-newline"]["category"],
                "multi_output",
            )
            train = json.loads((out_dir / "count-odds.train.json").read_text(encoding="utf-8"))
            test = json.loads((out_dir / "count-odds.test.json").read_text(encoding="utf-8"))
            self.assertEqual(train["format_version"], "fitness-cases")
            self.assertEqual(train["schema"]["inputs"]["input1"], "int_list")
            self.assertEqual(test["meta"]["schema_hash"], train["meta"]["schema_hash"])
            self.assertTrue((out_dir / "_summaries" / "count-odds.summary.json").exists())

    def test_release_psb_exclusion_manifest_is_consistent(self):
        manifest_path = ROOT / "benchmarks/psb_release_exclusions.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["format_version"], "psb-release-exclusions")
        self.assertEqual(manifest["status"], "recorded")
        self.assertEqual(manifest["summary"]["suite_count"], 2)
        self.assertEqual(manifest["summary"]["supported_problem_count"], 49)
        self.assertEqual(manifest["summary"]["excluded_problem_count"], 5)
        self.assertEqual(manifest["summary"]["total_problem_count"], 54)

        psb1 = manifest["suites"]["psb1"]
        psb2 = manifest["suites"]["psb2"]
        self.assertEqual(psb1["supported_count"], 28)
        self.assertEqual(psb1["excluded_count"], 1)
        self.assertEqual(psb2["supported_count"], 21)
        self.assertEqual(psb2["excluded_count"], 4)
        self.assertEqual(set(psb1["exclusions"]), {"replace-space-with-newline"})
        self.assertEqual(set(psb2["exclusions"]), {"coin-sums", "cut-vector", "find-pair", "mastermind"})
        for suite in (psb1, psb2):
            for row in suite["exclusions"].values():
                self.assertEqual(row["category"], "multi_output")
                self.assertIn("multi-output rows", row["reason"])

        for suite in (psb1, psb2):
            source_path = ROOT / suite["manifest"]
            if not source_path.exists():
                continue
            source = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(source["ok_count"], suite["supported_count"])
            self.assertEqual(source["failed_count"], suite["excluded_count"])
            self.assertEqual(set(source["excluded_problems"]), set(suite["exclusions"]))

    def test_supported_psb1_baseline_manifest_is_consistent(self):
        manifest_path = ROOT / "benchmarks/psb1_supported_all_config_fullbudget_baseline.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["format_version"], "psb-baseline-manifest")
        self.assertEqual(manifest["status"], "recorded")
        self.assertEqual(manifest["suite"], "psb1")
        self.assertEqual(manifest["profile"], "all_config_supported_fullbudget_merged")
        self.assertEqual(manifest["scope"]["problem_count"], 28)
        self.assertEqual(manifest["scope"]["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(manifest["scope"]["population_size"], 8192)
        self.assertEqual(manifest["scope"]["generations"], 100)
        self.assertEqual(manifest["scope"]["engine"], "gpu")
        self.assertEqual(manifest["scope"]["repro_backend"], "gpu")
        self.assertTrue(manifest["scope"]["repro_overlap"])
        self.assertTrue(manifest["scope"]["eval_test"])
        self.assertEqual(manifest["aggregate"]["runs"], 140)
        self.assertEqual(manifest["aggregate"]["ok_runs"], 140)
        self.assertEqual(manifest["aggregate"]["failed_runs"], 0)
        self.assertEqual(manifest["aggregate"]["test_ok_runs"], 140)
        self.assertEqual(manifest["merge_provenance"]["replacement_problem"], "wallis-pi")
        self.assertEqual(manifest["merge_provenance"]["replacement_seed"], 0)

        summary_path = ROOT / manifest["raw_evidence"]["summary"]
        if not summary_path.exists():
            return
        summary_bytes = summary_path.read_bytes()
        self.assertEqual(hashlib.sha256(summary_bytes).hexdigest(), manifest["raw_evidence"]["summary_sha256"])
        summary = json.loads(summary_bytes.decode("utf-8"))
        self.assertEqual(summary["metadata"]["profile"], manifest["profile"])
        self.assertEqual(summary["metadata"]["problems"], manifest["scope"]["problems"])
        self.assertEqual(summary["metadata"]["seeds"], manifest["scope"]["seeds"])
        self.assertEqual(summary["aggregate"]["runs"], manifest["aggregate"]["runs"])
        self.assertEqual(summary["aggregate"]["ok_runs"], manifest["aggregate"]["ok_runs"])
        self.assertEqual(summary["aggregate"]["failed_runs"], manifest["aggregate"]["failed_runs"])
        self.assertEqual(len(summary["aggregate"]["problems"]), manifest["scope"]["problem_count"])

    def test_run_psb_regression_eval_test_records_test_metrics(self):
        if not native_cli_ready():
            self.skipTest("native CLI is not built or is stale")
        with tempfile.TemporaryDirectory(prefix="g3p_psb_eval_test_") as td:
            td_path = Path(td)
            cases_root = td_path / "cases"
            cases_root.mkdir()
            train = {
                "format_version": "fitness-cases",
                "meta": {"schema_hash": "sha256:test-schema"},
                "schema": {"inputs": {"x": "int"}, "expected": "int"},
                "cases": [{"inputs": {"x": {"type": "int", "value": 1}}, "expected": {"type": "int", "value": 1}}],
            }
            test = {
                "format_version": "fitness-cases",
                "meta": {"schema_hash": "sha256:test-schema"},
                "schema": {"inputs": {"x": "int"}, "expected": "int"},
                "cases": [{"inputs": {"x": {"type": "int", "value": 2}}, "expected": {"type": "int", "value": 2}}],
            }
            (cases_root / "toy.train.json").write_text(json.dumps(train), encoding="utf-8")
            (cases_root / "toy.test.json").write_text(json.dumps(test), encoding="utf-8")
            out_dir = td_path / "out"
            cmd = [
                "python3",
                "tools/run_psb_regression.py",
                "--suite",
                "psb1",
                "--cases-root",
                str(cases_root),
                "--problems",
                "toy",
                "--seeds",
                "0",
                "--engine",
                "cpu",
                "--repro-backend",
                "cpu",
                "--population-size",
                "4",
                "--generations",
                "1",
                "--eval-test",
                "--out-dir",
                str(out_dir),
            ]
            subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            run = summary["runs"][0]
            self.assertEqual(run["test_status"], "ok")
            self.assertIn("test_best_fitness", run)
            self.assertEqual(summary["aggregate"]["problems"]["toy"]["test_ok_runs"], 1)

    def test_compare_psb_baseline_passes_with_close_candidate(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_compare_pass_") as td:
            td_path = Path(td)
            baseline = td_path / "baseline.json"
            candidate = td_path / "candidate.json"
            out = td_path / "comparison.json"

            write_summary(
                baseline,
                [
                    {
                        "status": "ok",
                        "problem": "median",
                        "seed": i,
                        "solved": i < 4,
                        "best_fitness": 0.0,
                        "total_ms": 100.0 + i,
                        "timing": {"generation_gpu_eval_kernel_ms": [10.0 + i]},
                    }
                    for i in range(5)
                ],
            )
            write_summary(
                candidate,
                [
                    {
                        "status": "ok",
                        "problem": "median",
                        "seed": i,
                        "solved": i < 3,
                        "best_fitness": 0.0,
                        "total_ms": 110.0 + i,
                        "timing": {"generation_gpu_eval_kernel_ms": [11.0 + i]},
                    }
                    for i in range(5)
                ],
            )

            cmd = [
                "python3",
                "tools/compare_psb_baseline.py",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--out",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("passed=true", proc.stdout)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])

    def test_compare_psb_baseline_fails_large_quality_regression(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_compare_fail_") as td:
            td_path = Path(td)
            baseline = td_path / "baseline.json"
            candidate = td_path / "candidate.json"
            out = td_path / "comparison.json"

            write_summary(
                baseline,
                [
                    {
                        "status": "ok",
                        "problem": "smallest",
                        "seed": i,
                        "solved": True,
                        "best_fitness": 0.0,
                        "total_ms": 100.0,
                        "timing": {},
                    }
                    for i in range(5)
                ],
            )
            write_summary(
                candidate,
                [
                    {
                        "status": "ok",
                        "problem": "smallest",
                        "seed": i,
                        "solved": i == 0,
                        "best_fitness": -1.0,
                        "total_ms": 100.0,
                        "timing": {},
                    }
                    for i in range(5)
                ],
            )

            cmd = [
                "python3",
                "tools/compare_psb_baseline.py",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--out",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("passed=false", proc.stdout)
            result = json.loads(out.read_text(encoding="utf-8"))
            problem = result["problems"][0]
            self.assertIn("stable_train_solved", problem["quality_failure_categories"])
            self.assertEqual(problem["speed_failure_attribution"], [])
            self.assertEqual(problem["failure_context"]["seeds"], [0, 1, 2, 3, 4])

    def test_compare_psb_baseline_fails_test_quality_regression(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_compare_test_fail_") as td:
            td_path = Path(td)
            baseline = td_path / "baseline.json"
            candidate = td_path / "candidate.json"
            out = td_path / "comparison.json"

            write_summary(
                baseline,
                [
                    {
                        "status": "ok",
                        "problem": "toy",
                        "seed": i,
                        "solved": False,
                        "best_fitness": 0.0,
                        "test_solved": True,
                        "test_best_fitness": 1.0,
                        "total_ms": 100.0,
                        "timing": {},
                    }
                    for i in range(3)
                ],
            )
            write_summary(
                candidate,
                [
                    {
                        "status": "ok",
                        "problem": "toy",
                        "seed": i,
                        "solved": False,
                        "best_fitness": 0.0,
                        "test_solved": False,
                        "test_best_fitness": -1.0,
                        "total_ms": 100.0,
                        "timing": {},
                    }
                    for i in range(3)
                ],
            )

            cmd = [
                "python3",
                "tools/compare_psb_baseline.py",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--out",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("passed=false", proc.stdout)
            result = json.loads(out.read_text(encoding="utf-8"))
            problem = result["problems"][0]
            self.assertIn("test_solved_count", problem["quality_failure_categories"])
            self.assertIn("test_median_best_fitness", problem["quality_failure_categories"])
            self.assertEqual(problem["speed_failure_attribution"], [])

    def test_compare_psb_baseline_uses_problem_specific_quality_tolerances(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_compare_problem_tol_") as td:
            td_path = Path(td)
            baseline = td_path / "baseline.json"
            candidate = td_path / "candidate.json"
            tolerances = td_path / "tolerances.json"
            strict_out = td_path / "comparison.strict.json"
            tolerant_out = td_path / "comparison.tolerant.json"

            write_summary(
                baseline,
                [
                    {
                        "status": "ok",
                        "problem": "super-anagrams",
                        "seed": i,
                        "solved": False,
                        "best_fitness": 193.0,
                        "test_solved": False,
                        "test_best_fitness": 191.0,
                        "total_ms": 100.0,
                        "timing": {},
                    }
                    for i in range(5)
                ],
            )
            write_summary(
                candidate,
                [
                    {
                        "status": "ok",
                        "problem": "super-anagrams",
                        "seed": i,
                        "solved": False,
                        "best_fitness": 192.0,
                        "test_solved": False,
                        "test_best_fitness": 187.0,
                        "total_ms": 100.0,
                        "timing": {},
                    }
                    for i in range(5)
                ],
            )
            tolerances.write_text(
                json.dumps(
                    {
                        "format_version": "psb-problem-tolerances",
                        "default": {
                            "median_fitness_tolerance": 0.0,
                            "test_median_fitness_tolerance": 0.0,
                        },
                        "problems": {
                            "super-anagrams": {
                                "median_fitness_tolerance": 1.0,
                                "test_median_fitness_tolerance": 4.0,
                                "reason": "known full-budget compact stochastic envelope",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            strict_cmd = [
                "python3",
                "tools/compare_psb_baseline.py",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--out",
                str(strict_out),
            ]
            strict_proc = subprocess.run(strict_cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(strict_proc.returncode, 0)
            self.assertIn("passed=false", strict_proc.stdout)

            tolerant_cmd = [
                "python3",
                "tools/compare_psb_baseline.py",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--problem-tolerances",
                str(tolerances),
                "--out",
                str(tolerant_out),
            ]
            tolerant_proc = subprocess.run(tolerant_cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("passed=true", tolerant_proc.stdout)
            result = json.loads(tolerant_out.read_text(encoding="utf-8"))
            problem = result["problems"][0]
            self.assertTrue(problem["passed"])
            self.assertEqual(problem["quality_thresholds"]["median_fitness_tolerance"], 1.0)
            self.assertEqual(problem["quality_thresholds"]["test_median_fitness_tolerance"], 4.0)
            self.assertEqual(result["thresholds"]["problem_tolerances"]["format_version"], "psb-problem-tolerances")

    def test_compare_psb_baseline_attributes_gpu_kernel_speed_regression(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_compare_speed_fail_") as td:
            td_path = Path(td)
            baseline = td_path / "baseline.json"
            candidate = td_path / "candidate.json"
            out = td_path / "comparison.json"

            write_summary(
                baseline,
                [
                    {
                        "status": "ok",
                        "problem": "strings",
                        "seed": i,
                        "schema_hash": "sha256:schema",
                        "solved": False,
                        "best_fitness": 1.0,
                        "total_ms": 100.0,
                        "timing": {
                            "generation_eval_ms": [20.0],
                            "generation_gpu_eval_kernel_ms": [10.0],
                        },
                    }
                    for i in range(2)
                ],
            )
            write_summary(
                candidate,
                [
                    {
                        "status": "ok",
                        "problem": "strings",
                        "seed": i,
                        "schema_hash": "sha256:schema",
                        "solved": False,
                        "best_fitness": 1.0,
                        "total_ms": 105.0,
                        "timing": {
                            "generation_eval_ms": [21.0],
                            "generation_gpu_eval_kernel_ms": [13.0],
                        },
                        "grammar_config": {
                            "kind": "generated_compact",
                            "path": "logs/strings.compact.json",
                        },
                    }
                    for i in range(2)
                ],
            )

            cmd = [
                "python3",
                "tools/compare_psb_baseline.py",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--out",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("passed=false", proc.stdout)
            result = json.loads(out.read_text(encoding="utf-8"))
            problem = result["problems"][0]
            self.assertEqual(problem["failures"], ["gpu_kernel_ms"])
            self.assertEqual(problem["speed_failure_attribution"], ["gpu_kernel"])
            self.assertEqual(problem["quality_failure_categories"], [])
            self.assertEqual(problem["failure_context"]["schema_hashes"], ["sha256:schema"])
            self.assertEqual(
                problem["failure_context"]["candidate_grammar_config"]["kind"],
                "generated_compact",
            )
            self.assertEqual(problem["timing_phase_ratios"]["gpu_kernel"]["ratio"], 1.3)

    def test_write_psb_manifest_compacts_comparison_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="g3p_psb_manifest_") as td:
            td_path = Path(td)
            baseline = td_path / "baseline.json"
            candidate = td_path / "candidate.json"
            comparison = td_path / "comparison.json"
            exclusions = td_path / "fixture_manifest.json"
            out = td_path / "manifest.json"

            baseline.write_text(
                json.dumps(
                    {
                        "format_version": "psb-regression-summary",
                        "metadata": {
                            "suite": "psb1",
                            "profile": "baseline",
                            "cases_root": "logs/baseline_cases",
                            "grammar_config": {
                                "path": "configs/grammar/all.json",
                                "hash": "sha256:old",
                            },
                            "seeds": [0, 1],
                            "population_size": 512,
                            "generations": 10,
                            "engine": "gpu",
                            "repro_backend": "gpu",
                            "repro_overlap": True,
                            "eval_test": True,
                        },
                        "aggregate": {"runs": 2, "ok_runs": 2},
                        "runs": [],
                    }
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    {
                        "format_version": "psb-regression-summary",
                        "metadata": {
                            "suite": "psb1",
                            "profile": "compact",
                            "cases_root": "logs/current_cases",
                            "base_grammar_config": {
                                "path": "configs/grammar/all.json",
                                "hash": "sha256:old",
                            },
                            "generated_grammar_configs": {
                                "count-odds": {"kind": "generated_compact"}
                            },
                            "seeds": [0, 1],
                            "population_size": 512,
                            "generations": 10,
                            "engine": "gpu",
                            "repro_backend": "gpu",
                            "repro_overlap": True,
                            "eval_test": True,
                        },
                        "aggregate": {"runs": 2, "ok_runs": 2},
                        "runs": [],
                    }
                ),
                encoding="utf-8",
            )
            comparison.write_text(
                json.dumps(
                    {
                        "format_version": "psb-baseline-comparison",
                        "passed": True,
                        "thresholds": {"median_runtime_ratio": 1.15},
                        "problems": [
                            {
                                "problem": "count-odds",
                                "baseline_runs": 2,
                                "candidate_runs": 2,
                                "baseline_median_best_fitness": -10.0,
                                "candidate_median_best_fitness": -9.0,
                                "fitness_regression": -1.0,
                                "baseline_median_test_best_fitness": -12.0,
                                "candidate_median_test_best_fitness": -11.0,
                                "test_fitness_regression": -1.0,
                                "median_total_ratio": 1.05,
                                "p90_total_ratio": 1.08,
                                "gpu_kernel_ratio": 1.02,
                                "failures": [],
                                "passed": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            exclusions.write_text(
                json.dumps(
                    {
                        "format_version": "psb-fixtures-manifest",
                        "excluded_problems": {
                            "replace-space-with-newline": {
                                "category": "multi_output",
                                "error": "multi-output",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cmd = [
                "python3",
                "tools/write_psb_manifest.py",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--comparison",
                str(comparison),
                "--excluded-manifest",
                str(exclusions),
                "--out",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("PSB_MANIFEST_STATUS passed", proc.stdout)

            manifest = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format_version"], "psb-comparison-manifest")
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["scope"]["problem_count"], 1)
            self.assertEqual(manifest["scope"]["baseline_cases_root"], "logs/baseline_cases")
            self.assertEqual(
                manifest["scope"]["excluded_problems"]["replace-space-with-newline"]["category"],
                "multi_output",
            )
            self.assertEqual(manifest["fairness"]["baseline_grammar_config"]["hash"], "sha256:old")
            self.assertEqual(manifest["fairness"]["candidate_generated_config_count"], 1)
            self.assertEqual(manifest["aggregate"]["max_median_total_ratio"], 1.05)
            self.assertEqual(manifest["aggregate"]["min_gpu_kernel_ratio"], 1.02)
            self.assertEqual(
                manifest["problems"]["count-odds"]["candidate_median_test_best_fitness"],
                -11.0,
            )


if __name__ == "__main__":
    unittest.main()
