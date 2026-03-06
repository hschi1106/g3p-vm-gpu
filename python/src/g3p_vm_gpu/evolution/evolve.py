from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..core.ast import Val
from ..runtime.vm import ExecError, ExecReturn, exec_bytecode
from .crossover import crossover
from .genome import Limits, ProgramGenome, compile_for_eval
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
    elitism: int = 2
    mutation_rate: float = 0.5
    mutation_subtree_prob: float = 0.8
    crossover_rate: float = 0.9
    selection_pressure: int = 3
    seed: int = 0
    fuel: int = 20_000
    limits: Limits = Limits()


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


def score_output(actual: Val, expected: Val) -> float:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return 1.0 if actual == expected else 0.0
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return -abs(float(actual) - float(expected))
    return 1.0 if actual == expected else 0.0


def evaluate_genome(genome: ProgramGenome, cases: Sequence[FitnessCase], cfg: EvolutionConfig) -> float:
    program = compile_for_eval(genome)
    score = 0.0
    for case in cases:
        out = exec_bytecode(program, inputs=case.inputs, fuel=cfg.fuel)
        if isinstance(out, ExecReturn):
            score += score_output(out.value, case.expected)
        elif isinstance(out, ExecError):
            score += 0.0
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
    return [make_random_genome(seed=cfg.seed + i, limits=cfg.limits) for i in range(cfg.population_size)]


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
    if cfg.elitism < 0 or cfg.elitism > cfg.population_size:
        raise ValueError("elitism must be in [0, population_size]")
    if cfg.selection_pressure <= 0:
        raise ValueError("selection_pressure must be > 0")

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

        next_population: List[ProgramGenome] = [item.genome for item in scored[: cfg.elitism]]
        while len(next_population) < cfg.population_size:
            parent_a = select_parent_tournament(scored, rng, cfg.selection_pressure)
            child = parent_a
            if rng.random() < cfg.crossover_rate:
                parent_b = select_parent_tournament(scored, rng, cfg.selection_pressure)
                child = crossover(parent_a, parent_b, seed=rng.randint(0, 2_000_000_000), limits=cfg.limits)
            if rng.random() < cfg.mutation_rate:
                child = mutate(
                    child,
                    seed=rng.randint(0, 2_000_000_000),
                    limits=cfg.limits,
                    mutation_subtree_prob=cfg.mutation_subtree_prob,
                )
            next_population.append(child)
        population = next_population

    final_scored = evaluate_population(population, cases, cfg)
    return EvolutionResult(
        best=final_scored[0],
        history_best=history_best,
        history_best_fitness=history_best_fitness,
        history_mean_fitness=history_mean_fitness,
        final_population=final_scored,
    )
