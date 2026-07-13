import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.g3p_vm_gpu.core.ast import BUILTIN_NODE_BY_NAME, Char, FloatList, IntList, NodeKind, StringList
from src.g3p_vm_gpu.core.errors import Failed, Returned
from src.g3p_vm_gpu.evolution.crossover import crossover
from src.g3p_vm_gpu.evolution.genome import Limits, compile_for_eval
from src.g3p_vm_gpu.evolution.grammar_config import load_grammar_config
from src.g3p_vm_gpu.evolution.mutation import mutate
from src.g3p_vm_gpu.evolution.random_genome import make_random_genome
from src.g3p_vm_gpu.evolution.stmt_codec import top_level_statements
from src.g3p_vm_gpu.runtime.compiler import compile_program
from src.g3p_vm_gpu.runtime.interp import run_program
from src.g3p_vm_gpu.runtime.vm import ExecError, ExecReturn, exec_bytecode


def _spec_current_grammar_config():
    return {
        "format_version": "grammar-config",
        "profile": "test_current",
        "values": {
            "int": True,
            "float": True,
            "bool": True,
            "char": True,
            "string": True,
            "int_list": True,
            "float_list": True,
            "string_list": True,
        },
        "statements": {
            "assign": True,
            "if_stmt": True,
            "for_range": True,
            "return": True,
        },
        "expressions": {
            "const": True,
            "var": True,
            "bound_var": True,
            "unary": True,
            "binary": True,
            "if_expr": True,
            "call": True,
            "map_list": False,
            "filter_list": False,
            "linear_rec": False,
            "asgp_dc": False,
            "asgp_dp1d": False,
            "asgp_dp2d": False,
        },
        "builtins": {
            "abs": True,
            "min": True,
            "max": True,
            "clip": True,
            "idiv0": True,
            "imod0": True,
            "len": True,
            "concat": True,
            "slice": True,
            "index": True,
            "append": True,
            "prepend": True,
            "reverse": True,
            "find": True,
            "contains": True,
            "singleton": True,
            "char_to_string": True,
            "string_to_char": True,
            "ord": True,
            "chr": True,
            "is_letter": True,
            "is_digit": True,
            "is_space": True,
            "is_vowel": True,
            "to_lower": True,
            "to_upper": True,
            "to_string": True,
        },
        "structured": {
            "max_nested_binders": 0,
            "max_map_body_depth": 0,
            "max_filter_pred_depth": 0,
            "max_linear_rec_body_depth": 0,
        },
        "asgp": {
            "max_scheme_nesting": 0,
            "dc": {"enabled_source_elems": [], "max_depth": 0},
            "dp1d": {"max_states": 0, "max_step": 0, "dependency_patterns": []},
            "dp2d": {"max_cells": 0, "dependency_patterns": []},
        },
        "limits": {
            "max_expr_depth": 7,
            "max_stmts_per_block": 6,
            "max_total_nodes": 80,
            "max_for_k": 16,
            "max_call_args": 3,
        },
        "compat": None,
    }


class TestEvolutionOps(unittest.TestCase):
    def _assert_no_sequence_features(self, genome):
        disabled_calls = {
            node
            for name, node in BUILTIN_NODE_BY_NAME.items()
            if name not in {"abs", "min", "max", "clip"}
        }
        disabled_structured = {NodeKind.MAP_LIST, NodeKind.FILTER_LIST, NodeKind.LINEAR_REC}
        for node in genome.ast.nodes:
            self.assertNotIn(node.kind, disabled_calls)
            self.assertNotIn(node.kind, disabled_structured)
        for value in genome.ast.consts:
            self.assertNotIsInstance(value, (Char, str, IntList, FloatList, StringList))

    def test_random_genome_compile_rate(self):
        limits = Limits()
        compiled = 0
        total = 400
        for i in range(total):
            genome = make_random_genome(seed=i, limits=limits)
            compile_program(genome.ast)
            compiled += 1
        self.assertGreaterEqual(compiled / total, 0.99)

    def test_mutation_produces_compilable_genomes(self):
        limits = Limits()
        base = make_random_genome(seed=123, limits=limits)
        for i in range(120):
            child = mutate(base, seed=i, limits=limits, mutation_subtree_prob=0.8)
            self.assertLessEqual(child.meta.node_count, limits.max_total_nodes)
            compile_for_eval(child)

    def test_crossover_produces_compilable_genomes(self):
        limits = Limits()
        parent_a = make_random_genome(seed=1, limits=limits)
        parent_b = make_random_genome(seed=2, limits=limits)
        for i in range(120):
            child = crossover(parent_a, parent_b, seed=i, limits=limits)
            self.assertLessEqual(child.meta.node_count, limits.max_total_nodes)
            compile_for_eval(child)

    def test_random_generator_uses_bounded_constant_for_loops(self):
        limits = Limits(max_for_k=8)
        def check_block(statements):
            for stmt in statements:
                if stmt[0] == "for":
                    self.assertEqual(stmt[2][0], "const")
                    self.assertIsInstance(stmt[2][1], int)
                    self.assertLessEqual(stmt[2][1], limits.max_for_k)
                    check_block(stmt[3])
                elif stmt[0] == "if":
                    check_block(stmt[2])
                    check_block(stmt[3])
        for i in range(100):
            genome = make_random_genome(seed=88 + i, limits=limits)
            check_block(top_level_statements(genome.ast))

    def test_random_generator_emits_structured_expressions_when_enabled(self):
        limits = Limits(max_expr_depth=7, max_stmts_per_block=6, max_total_nodes=160)
        seen = set()
        for i in range(120):
            genome = make_random_genome(seed=9000 + i, limits=limits)
            kinds = {node.kind for node in genome.ast.nodes}
            if NodeKind.MAP_LIST in kinds:
                seen.add(NodeKind.MAP_LIST)
            if NodeKind.FILTER_LIST in kinds:
                seen.add(NodeKind.FILTER_LIST)
            if NodeKind.LINEAR_REC in kinds:
                seen.add(NodeKind.LINEAR_REC)
            compile_for_eval(genome)
            if len(seen) == 3:
                break
        self.assertEqual(seen, {NodeKind.MAP_LIST, NodeKind.FILTER_LIST, NodeKind.LINEAR_REC})

    def test_mutation_emits_structured_expressions_when_enabled(self):
        limits = Limits(max_expr_depth=7, max_stmts_per_block=6, max_total_nodes=160)
        base = make_random_genome(seed=9100, limits=limits)
        seen = set()
        for i in range(160):
            child = mutate(base, seed=9200 + i, limits=limits, mutation_subtree_prob=1.0)
            kinds = {node.kind for node in child.ast.nodes}
            if NodeKind.MAP_LIST in kinds:
                seen.add(NodeKind.MAP_LIST)
            if NodeKind.FILTER_LIST in kinds:
                seen.add(NodeKind.FILTER_LIST)
            if NodeKind.LINEAR_REC in kinds:
                seen.add(NodeKind.LINEAR_REC)
            compile_for_eval(child)
            if seen:
                break
        self.assertTrue(seen)

    def test_eval_parity_sample(self):
        limits = Limits()
        for i in range(80):
            genome = make_random_genome(seed=2000 + i, limits=limits)
            interp_env, interp_out = run_program(genome.ast, {}, fuel=20_000)
            vm_out = exec_bytecode(compile_for_eval(genome), {}, fuel=20_000)

            if isinstance(interp_out, Returned):
                self.assertIsInstance(vm_out, ExecReturn)
                self.assertEqual(interp_out.value, vm_out.value)
            elif isinstance(interp_out, Failed):
                self.assertIsInstance(vm_out, ExecError)
                self.assertEqual(interp_out.err.code, vm_out.err.code)
            else:
                self.fail("unexpected top-level interpreter outcome")

            self.assertIsInstance(interp_env, dict)

    def test_scalar_grammar_config_restricts_random_generation(self):
        limits = Limits()
        grammar = load_grammar_config("configs/grammar/scalar.json")
        for i in range(160):
            genome = make_random_genome(seed=5000 + i, limits=limits, grammar_config=grammar)
            self._assert_no_sequence_features(genome)
            compile_for_eval(genome)

    def test_scalar_grammar_config_restricts_mutation(self):
        limits = Limits()
        grammar = load_grammar_config("configs/grammar/scalar.json")
        base = make_random_genome(seed=6111, limits=limits, grammar_config=grammar)
        for i in range(80):
            child = mutate(base, seed=7000 + i, limits=limits, mutation_subtree_prob=0.8, grammar_config=grammar)
            self._assert_no_sequence_features(child)
            compile_for_eval(child)

    def test_checked_in_grammar_presets_parse(self):
        for name in ("all", "scalar", "string", "num_list", "string_list", "sequence"):
            with self.subTest(name=name):
                load_grammar_config(f"configs/grammar/{name}.json")

    def test_spec_current_grammar_config_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grammar_current.json"
            path.write_text(json.dumps(_spec_current_grammar_config()), encoding="utf-8")
            grammar = load_grammar_config(path)
        self.assertTrue(grammar.allow_value("char"))
        self.assertTrue(grammar.allow_value("int_list"))
        self.assertTrue(grammar.allow_value("float_list"))
        self.assertTrue(grammar.allow_builtin("prepend"))

    def test_spec_current_grammar_config_rejects_legacy_value_keys(self):
        raw = _spec_current_grammar_config()
        raw["values"]["num_list"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_grammar_current.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_grammar_config(path)

    def test_grammar_config_rejects_unknown_keys(self):
        raw = json.loads(Path("configs/grammar/scalar.json").read_text(encoding="utf-8"))
        raw["values"]["generic_list"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_grammar.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_grammar_config(path)

    def test_native_cli_accepts_spec_current_grammar_config(self):
        binary = Path("cpp/build/g3pvm_evolve_cli")
        if not binary.exists():
            self.skipTest("native CLI is not built")
        source = Path("cpp/src/cli/evolve_cli.cpp")
        if source.exists() and source.stat().st_mtime > binary.stat().st_mtime:
            self.skipTest("native CLI binary is older than evolve_cli.cpp")
        cases = {
            "format_version": "fitness-cases",
            "cases": [
                {
                    "inputs": {"x": {"type": "int", "value": 1}},
                    "expected": {"type": "int", "value": 1},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            grammar_path = tmp_path / "grammar_current.json"
            cases_path = tmp_path / "cases_current.json"
            out_path = tmp_path / "run.json"
            grammar_path.write_text(json.dumps(_spec_current_grammar_config()), encoding="utf-8")
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            result = subprocess.run(
                [
                    str(binary),
                    "--cases",
                    str(cases_path),
                    "--grammar-config",
                    str(grammar_path),
                    "--engine",
                    "cpu",
                    "--repro-backend",
                    "cpu",
                    "--population-size",
                    "4",
                    "--generations",
                    "1",
                    "--out-json",
                    str(out_path),
                ],
                cwd=Path.cwd(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
