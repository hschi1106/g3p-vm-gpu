import json
import tempfile
import unittest
from pathlib import Path

from src.g3p_vm_gpu.core.ast import NodeKind, NumList, StringList
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


class TestEvolutionOps(unittest.TestCase):
    def _assert_no_sequence_features(self, genome):
        disabled_calls = {
            NodeKind.CALL_LEN,
            NodeKind.CALL_CONCAT,
            NodeKind.CALL_SLICE,
            NodeKind.CALL_INDEX,
            NodeKind.CALL_APPEND,
            NodeKind.CALL_REVERSE,
            NodeKind.CALL_FIND,
            NodeKind.CALL_CONTAINS,
        }
        for node in genome.ast.nodes:
            self.assertNotIn(node.kind, disabled_calls)
        for value in genome.ast.consts:
            self.assertNotIsInstance(value, (str, NumList, StringList))

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

    def test_grammar_config_rejects_unknown_keys(self):
        raw = json.loads(Path("configs/grammar/scalar.json").read_text(encoding="utf-8"))
        raw["values"]["generic_list"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_grammar.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_grammar_config(path)


if __name__ == "__main__":
    unittest.main()
