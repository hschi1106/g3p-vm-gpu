from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence

from .ast import Val
from .evo_encoding import Limits, ProgramGenome, compile_for_eval, crossover, make_random_genome, mutate
from .vm import VMError, VMReturn, run_bytecode


class SelectionMethod(str, Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    TRUNCATION = "truncation"
    RANDOM = "random"


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
    crossover_rate: float = 0.9
    crossover_method: str = "hybrid"
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_k: int = 3
    truncation_ratio: float = 0.5
    seed: int = 0
    fuel: int = 20_000
    limits: Limits = Limits()
    float_abs_tol: float = 1e-12
    float_rel_tol: float = 1e-12
    reward_match: float = 1.0
    penalty_mismatch: float = 0.0
    penalty_error: float = -1.0


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


def _is_close(a: Val, b: Val, abs_tol: float, rel_tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        af = float(a) if isinstance(a, (int, float)) else None
        bf = float(b) if isinstance(b, (int, float)) else None
        if af is None or bf is None:
            return False
        diff = abs(af - bf)
        return diff <= max(abs_tol, rel_tol * max(abs(af), abs(bf)))
    return a == b


def evaluate_genome(genome: ProgramGenome, cases: Sequence[FitnessCase], cfg: EvolutionConfig) -> float:
    program = compile_for_eval(genome)
    score = 0.0
    for case in cases:
        out = run_bytecode(program, inputs=case.inputs, fuel=cfg.fuel)
        if isinstance(out, VMReturn):
            if _is_close(out.value, case.expected, cfg.float_abs_tol, cfg.float_rel_tol):
                score += cfg.reward_match
            else:
                score += cfg.penalty_mismatch
        elif isinstance(out, VMError):
            score += cfg.penalty_error
    return score


def evaluate_population(population: Sequence[ProgramGenome], cases: Sequence[FitnessCase], cfg: EvolutionConfig) -> List[ScoredGenome]:
    scored = [ScoredGenome(genome=g, fitness=evaluate_genome(g, cases, cfg)) for g in population]
    scored.sort(key=lambda x: x.fitness, reverse=True)
    return scored


def select_parent(
    scored: Sequence[ScoredGenome],
    rng: random.Random,
    method: SelectionMethod,
    tournament_k: int = 3,
    truncation_ratio: float = 0.5,
) -> ProgramGenome:
    if not scored:
        raise ValueError("scored population is empty")

    if method == SelectionMethod.RANDOM:
        return rng.choice(scored).genome

    if method == SelectionMethod.TOURNAMENT:
        k = max(1, min(tournament_k, len(scored)))
        pool = rng.sample(list(scored), k)
        return max(pool, key=lambda x: x.fitness).genome

    if method == SelectionMethod.ROULETTE:
        min_fit = min(s.fitness for s in scored)
        shift = -min_fit + 1e-9 if min_fit <= 0 else 0.0
        weights = [s.fitness + shift for s in scored]
        total = sum(weights)
        if total <= 0:
            return rng.choice(scored).genome
        pick = rng.uniform(0.0, total)
        acc = 0.0
        for s, w in zip(scored, weights):
            acc += w
            if acc >= pick:
                return s.genome
        return scored[-1].genome

    if method == SelectionMethod.RANK:
        ranked = sorted(scored, key=lambda x: x.fitness)
        n = len(ranked)
        weights = [i + 1 for i in range(n)]
        total = sum(weights)
        pick = rng.uniform(0.0, float(total))
        acc = 0.0
        for s, w in zip(ranked, weights):
            acc += float(w)
            if acc >= pick:
                return s.genome
        return ranked[-1].genome

    if method == SelectionMethod.TRUNCATION:
        ratio = min(max(truncation_ratio, 0.05), 1.0)
        keep_n = max(1, int(len(scored) * ratio))
        return rng.choice(list(scored[:keep_n])).genome

    raise ValueError(f"unknown selection method: {method}")


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

    rng = random.Random(cfg.seed)
    population = list(initial_population) if initial_population is not None else _init_population(cfg)
    if len(population) != cfg.population_size:
        raise ValueError("initial_population size must match population_size")

    history_best_fitness: List[float] = []
    history_mean_fitness: List[float] = []
    history_best: List[ScoredGenome] = []

    for gen in range(cfg.generations):
        scored = evaluate_population(population, cases, cfg)
        best = scored[0]
        history_best.append(best)
        history_best_fitness.append(best.fitness)
        history_mean_fitness.append(sum(s.fitness for s in scored) / len(scored))

        next_population: List[ProgramGenome] = [s.genome for s in scored[: cfg.elitism]]

        while len(next_population) < cfg.population_size:
            p1 = select_parent(
                scored,
                rng=rng,
                method=cfg.selection_method,
                tournament_k=cfg.tournament_k,
                truncation_ratio=cfg.truncation_ratio,
            )
            child = p1
            if rng.random() < cfg.crossover_rate:
                p2 = select_parent(
                    scored,
                    rng=rng,
                    method=cfg.selection_method,
                    tournament_k=cfg.tournament_k,
                    truncation_ratio=cfg.truncation_ratio,
                )
                child = crossover(
                    p1,
                    p2,
                    seed=rng.randint(0, 2_000_000_000),
                    limits=cfg.limits,
                    method=cfg.crossover_method,
                )
            if rng.random() < cfg.mutation_rate:
                child = mutate(child, seed=rng.randint(0, 2_000_000_000), limits=cfg.limits)
            next_population.append(child)

        population = next_population

        # Keep deterministic generations even if caller probes per generation.
        _ = gen

    final_scored = evaluate_population(population, cases, cfg)
    return EvolutionResult(
        best=final_scored[0],
        history_best=history_best,
        history_best_fitness=history_best_fitness,
        history_mean_fitness=history_mean_fitness,
        final_population=final_scored,
    )
