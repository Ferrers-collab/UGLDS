from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from pathlib import Path
import random

import pandas as pd

from .core import (
    RoutingContext,
    Solution,
    base_permutation,
    decode_permutation,
    data_quality_audit,
    load_routing_context,
    monte_carlo_solution_stats,
    physical_vehicle_mix,
    route_rows,
    stop_rows,
)


@dataclass
class Problem1Config:
    population_size: int = 60
    generations: int = 120
    elite_size: int = 8
    tournament_size: int = 3
    mutation_rate: int | float = 0.25
    max_segment_length: int = 20
    mc_candidates: int = 6
    mc_runs: int = 100
    seed: int = 20260425
    progress_callback: Callable[[int, int, float], None] | None = None


def _ordered_crossover(parent_a: list[int], parent_b: list[int], rng: random.Random) -> list[int]:
    if len(parent_a) < 2:
        return list(parent_a)
    left, right = sorted(rng.sample(range(len(parent_a)), 2))
    child = [-1] * len(parent_a)
    child[left:right] = parent_a[left:right]
    fill_values = [gene for gene in parent_b if gene not in child]
    fill_cursor = 0
    for index in range(len(child)):
        if child[index] == -1:
            child[index] = fill_values[fill_cursor]
            fill_cursor += 1
    return child


def _mutate(permutation: list[int], rng: random.Random) -> list[int]:
    child = list(permutation)
    if len(child) < 2:
        return child
    left, right = sorted(rng.sample(range(len(child)), 2))
    child[left:right] = reversed(child[left:right])
    return child


def _perturb(base: list[int], rng: random.Random, strength: int = 3) -> list[int]:
    perm = list(base)
    for _ in range(max(1, strength)):
        perm = _mutate(perm, rng)
    return perm


def _seed_population(context: RoutingContext, config: Problem1Config) -> list[list[int]]:
    rng = random.Random(config.seed)
    base = base_permutation(context)
    population = [base]
    while len(population) < config.population_size:
        candidate = _perturb(base, rng, strength=max(2, len(base) // 40))
        population.append(candidate)
    return population


def _evaluate(
    context: RoutingContext,
    permutation: list[int],
    config: Problem1Config,
    cache: dict[tuple[int, ...], Solution],
) -> Solution:
    key = tuple(permutation)
    if key not in cache:
        cache[key] = decode_permutation(
            context=context,
            permutation=permutation,
            variant="problem1",
            tradeoff_lambda=0.0,
            max_segment_length=config.max_segment_length,
        )
    return cache[key]


def _tournament(
    population: list[list[int]],
    solutions: dict[tuple[int, ...], Solution],
    config: Problem1Config,
    rng: random.Random,
) -> list[int]:
    candidates = rng.sample(population, config.tournament_size)
    ranked = sorted(candidates, key=lambda perm: solutions[tuple(perm)].total_cost)
    return list(ranked[0])


def solve_problem1(context: RoutingContext, config: Problem1Config | None = None) -> Solution:
    config = config or Problem1Config()
    rng = random.Random(config.seed)
    population = _seed_population(context, config)
    cache: dict[tuple[int, ...], Solution] = {}
    convergence = []

    for _ in range(config.generations):
        solutions = {tuple(perm): _evaluate(context, perm, config, cache) for perm in population}
        ranked = sorted(population, key=lambda perm: solutions[tuple(perm)].total_cost)
        best_solution = solutions[tuple(ranked[0])]
        convergence.append(best_solution.total_cost)
        if config.progress_callback and ((len(convergence) == 1) or (len(convergence) % 10 == 0) or (len(convergence) == config.generations)):
            config.progress_callback(len(convergence), config.generations, best_solution.total_cost)

        next_population = [list(perm) for perm in ranked[: config.elite_size]]
        while len(next_population) < config.population_size:
            parent_a = _tournament(ranked, solutions, config, rng)
            parent_b = _tournament(ranked, solutions, config, rng)
            child = _ordered_crossover(parent_a, parent_b, rng)
            if rng.random() < float(config.mutation_rate):
                child = _mutate(child, rng)
            next_population.append(child)
        population = next_population

    final_solutions = {tuple(perm): _evaluate(context, perm, config, cache) for perm in population}
    ranked = sorted(population, key=lambda perm: final_solutions[tuple(perm)].total_cost)
    top_candidates = ranked[: min(config.mc_candidates, len(ranked))]

    robust_records = []
    for perm in top_candidates:
        solution = final_solutions[tuple(perm)]
        mc_stats = monte_carlo_solution_stats(
            context=context,
            solution=solution,
            runs=config.mc_runs,
            seed=config.seed,
        )
        robust_records.append((mc_stats["expected_cost"], solution, mc_stats))

    robust_records.sort(key=lambda item: item[0])
    best_expected_cost, best_solution, mc_stats = robust_records[0]
    best_solution.metadata["convergence"] = convergence
    best_solution.metadata["mc_stats"] = mc_stats
    best_solution.metadata["vehicle_mix"] = physical_vehicle_mix(best_solution)
    best_solution.metadata["expected_cost"] = best_expected_cost
    return best_solution


def write_problem1_outputs(solution: Solution, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(route_rows(solution)).to_csv(output_dir / "route_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stop_rows(solution)).to_csv(output_dir / "route_stops.csv", index=False, encoding="utf-8-sig")

    if "convergence" in solution.metadata:
        pd.DataFrame(
            {
                "generation": range(1, len(solution.metadata["convergence"]) + 1),
                "best_cost": solution.metadata["convergence"],
            }
        ).to_csv(output_dir / "convergence.csv", index=False, encoding="utf-8-sig")

    audit_context = solution.metadata.get("context")
    if audit_context is None:
        try:
            audit_context = load_routing_context()
        except Exception:
            audit_context = None
    audit = data_quality_audit(audit_context) if audit_context is not None else None
    summary_lines = [
        "问题1 静态调度结果",
        f"总成本: {solution.total_cost:.2f}",
        f"期望成本(MC): {solution.metadata.get('expected_cost', solution.total_cost):.2f}",
        f"总碳排放: {solution.total_carbon:.2f} kg",
        f"总里程: {solution.total_distance:.2f} km",
        f"路线数: {len(solution.routes)}",
        f"可行性: {'是' if solution.feasible else '否'}",
        f"车辆结构: {solution.metadata.get('vehicle_mix', {})}",
        "启动成本计费口径: 按当日实际启用物理车辆计，同一车辆多趟配送不重复计启动成本。",
        f"MC统计: {solution.metadata.get('mc_stats', {})}",
    ]
    if audit is not None:
        summary_lines.extend(
            [
                f"客户总数: {audit['customer_count']}",
                f"有效需求客户数: {audit['active_customer_count']}",
                f"绿色区客户数(附件坐标≤10km): {audit['green_customer_count']}",
                f"有需求绿色区客户数: {audit['active_green_customer_count']}",
            ]
        )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
