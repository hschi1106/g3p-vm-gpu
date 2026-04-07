from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..core.ast import Val
from ..runtime.vm import ExecError, ExecReturn, exec_bytecode
from .crossover import crossover
from .genome import Limits, ProgramGenome, compile_for_eval
from .grammar_config import DEFAULT_GRAMMAR_CONFIG, GrammarConfig
from .mutation import mutate
from .random_genome import make_random_genome


@dataclass(frozen=True)
class FitnessCase:
    inputs: Dict[str, Val]
    expected: Val


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 64
    generations: int = 40
    mutation_rate: float = 0.5
    mutation_subtree_prob: float = 0.8
    penalty: float = 1.0
    selection_pressure: int = 3
    seed: int = 0
    fuel: int = 20_000
    limits: Limits = Limits()
    grammar_config: GrammarConfig = field(default_factory=lambda: DEFAULT_GRAMMAR_CONFIG)


@dataclass(frozen=True)
class ScoredGenome:
    genome: ProgramGenome
    fitness: float


@dataclass(frozen=True)
class EvolutionResult:
    best: ScoredGenome
    history_best: List[ScoredGenome]
    history_best_fitness: List[float]
    history_mean_fitness: List[float]
    final_population: List[ScoredGenome]


def _is_numeric(value: Val) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def score_output(actual: Val, expected: Val) -> float:
    return score_output_with_penalty(actual, expected, penalty=1.0)


def score_output_with_penalty(actual: Val, expected: Val, penalty: float) -> float:
    actual_is_numeric = _is_numeric(actual)
    expected_is_numeric = _is_numeric(expected)

    if expected_is_numeric:
        if actual_is_numeric:
            return -abs(float(actual) - float(expected))
        return -abs(float(penalty))

    if type(actual) is not type(expected):
        return -abs(float(penalty))

    return 1.0 if actual == expected else 0.0


def evaluate_genome(genome: ProgramGenome, cases: Sequence[FitnessCase], cfg: EvolutionConfig) -> float:
    program = compile_for_eval(genome)
    score = 0.0
    for case in cases:
        out = exec_bytecode(program, inputs=case.inputs, fuel=cfg.fuel)
        if isinstance(out, ExecReturn):
            score += score_output_with_penalty(out.value, case.expected, cfg.penalty)
        elif isinstance(out, ExecError):
            score -= abs(float(cfg.penalty))
    return score


def evaluate_population(population: Sequence[ProgramGenome], cases: Sequence[FitnessCase], cfg: EvolutionConfig) -> List[ScoredGenome]:
    scored = [ScoredGenome(genome=genome, fitness=evaluate_genome(genome, cases, cfg)) for genome in population]
    scored.sort(key=lambda item: item.fitness, reverse=True)
    return scored


def select_parent_tournament(scored: Sequence[ScoredGenome], rng: random.Random, selection_pressure: int) -> ProgramGenome:
    if not scored:
        raise ValueError("scored population is empty")
    tournament_size = max(1, min(selection_pressure, len(scored)))
    pool = [scored[rng.randrange(len(scored))] for _ in range(tournament_size)]
    return max(pool, key=lambda item: item.fitness).genome


def _init_population(cfg: EvolutionConfig) -> List[ProgramGenome]:
    return [
        make_random_genome(seed=cfg.seed + i, limits=cfg.limits, grammar_config=cfg.grammar_config)
        for i in range(cfg.population_size)
    ]


def evolve_population(
    cases: Sequence[FitnessCase],
    cfg: EvolutionConfig,
    initial_population: Sequence[ProgramGenome] | None = None,
) -> EvolutionResult:
    if not cases:
        raise ValueError("cases must not be empty")
    if cfg.population_size <= 0:
        raise ValueError("population_size must be > 0")
    if cfg.generations <= 0:
        raise ValueError("generations must be > 0")
    if cfg.selection_pressure <= 0:
        raise ValueError("selection_pressure must be > 0")
    if cfg.penalty < 0:
        raise ValueError("penalty must be >= 0")

    rng = random.Random(cfg.seed)
    population = list(initial_population) if initial_population is not None else _init_population(cfg)
    if len(population) != cfg.population_size:
        raise ValueError("initial_population size must match population_size")

    history_best: List[ScoredGenome] = []
    history_best_fitness: List[float] = []
    history_mean_fitness: List[float] = []

    for _ in range(cfg.generations):
        scored = evaluate_population(population, cases, cfg)
        best = scored[0]
        history_best.append(best)
        history_best_fitness.append(best.fitness)
        history_mean_fitness.append(sum(item.fitness for item in scored) / len(scored))

        pair_count = (cfg.population_size + 1) // 2
        selected_parents: List[ProgramGenome] = [
            select_parent_tournament(scored, rng, cfg.selection_pressure) for _ in range(pair_count * 2)
        ]
        next_population = list(selected_parents)
        if len(selected_parents) > 1:
            rng.shuffle(next_population)
            for i in range(0, len(next_population) - 1, 2):
                parent_a = next_population[i]
                parent_b = next_population[i + 1]
                next_population[i] = crossover(
                    parent_a,
                    parent_b,
                    seed=rng.randint(0, 2_000_000_000),
                    limits=cfg.limits,
                    grammar_config=cfg.grammar_config,
                )
                next_population[i + 1] = crossover(
                    parent_b,
                    parent_a,
                    seed=rng.randint(0, 2_000_000_000),
                    limits=cfg.limits,
                    grammar_config=cfg.grammar_config,
                )
        for i, child in enumerate(next_population):
            if rng.random() < cfg.mutation_rate:
                next_population[i] = mutate(
                    child,
                    seed=rng.randint(0, 2_000_000_000),
                    limits=cfg.limits,
                    mutation_subtree_prob=cfg.mutation_subtree_prob,
                    grammar_config=cfg.grammar_config,
                )
        population = next_population[: cfg.population_size]

    final_scored = evaluate_population(population, cases, cfg)
    return EvolutionResult(
        best=final_scored[0],
        history_best=history_best,
        history_best_fitness=history_best_fitness,
        history_mean_fitness=history_mean_fitness,
        final_population=final_scored,
    )
