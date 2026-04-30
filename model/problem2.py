from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable
from pathlib import Path
import math
import random

import numpy as np
import pandas as pd

from .core import (
    LARGE_PENALTY,
    BAN_END,
    BAN_START,
    DeliveryJob,
    RoutingContext,
    RoutePlan,
    Solution,
    VehicleType,
    base_permutation,
    build_route_plan,
    data_quality_audit,
    decode_permutation,
    evaluate_solution,
    physical_vehicle_mix,
    route_selection_score,
    route_rows,
    stop_rows,
)


@dataclass
class Problem2Config:
    population_size: int = 60
    generations: int = 120
    mutation_rate: float = 0.24
    tradeoff_mutation_sigma: float = 0.08
    max_segment_length: int = 20
    representative_count: int = 6
    seed: int = 20260426
    progress_callback: Callable[[int, int, int, float, float], None] | None = None


def _tradeoff_lambda(gene: float) -> float:
    return 0.25 + 3.25 * float(np.clip(gene, 0.0, 1.0))


def _is_mandatory_green(job: DeliveryJob) -> bool:
    return job.green_zone == 1 and job.earliest < BAN_END and job.latest > BAN_START


def _sorted_mandatory_jobs(mandatory_jobs: list[DeliveryJob], mode: str) -> list[DeliveryJob]:
    if mode == "largest":
        return sorted(mandatory_jobs, key=lambda job: (-job.weight, job.earliest, job.customer_id, job.split_index))
    if mode == "tightest":
        return sorted(
            mandatory_jobs,
            key=lambda job: ((job.latest - job.earliest), job.latest, -job.weight, job.customer_id, job.split_index),
        )
    return sorted(mandatory_jobs, key=lambda job: (job.earliest, job.latest, -job.weight, job.customer_id, job.split_index))


def _best_inserted_route(
    context: RoutingContext,
    route: RoutePlan,
    job: DeliveryJob,
    tradeoff_lambda: float,
):
    best_route = None
    best_delta = math.inf
    for position in range(len(route.jobs) + 1):
        candidate_jobs = route.jobs[:position] + [job] + route.jobs[position:]
        if sum(item.weight for item in candidate_jobs) - route.vehicle.capacity_weight > 1e-6:
            continue
        if sum(item.volume for item in candidate_jobs) - route.vehicle.capacity_volume > 1e-6:
            continue
        candidate = build_route_plan(context, route.vehicle, candidate_jobs, "problem2")
        delta = route_selection_score(candidate, "problem2", tradeoff_lambda) - route_selection_score(
            route,
            "problem2",
            tradeoff_lambda,
        )
        if delta < best_delta:
            best_delta = delta
            best_route = candidate
    return best_route, best_delta


def _select_new_ev_vehicle(
    electric_types: dict[str, VehicleType],
    used_counts: Counter,
    current_job: DeliveryJob,
) -> VehicleType | None:
    del used_counts
    electric_list = sorted(
        electric_types.values(),
        key=lambda vehicle: (vehicle.capacity_weight, vehicle.capacity_volume),
    )
    for vehicle in electric_list:
        if current_job.weight <= vehicle.capacity_weight + 1e-6 and current_job.volume <= vehicle.capacity_volume + 1e-6:
            return vehicle
    return None


def _build_mandatory_ev_routes(
    context: RoutingContext,
    mandatory_jobs: list[DeliveryJob],
    tradeoff_lambda: float,
    mode: str,
) -> list[RoutePlan]:
    if not mandatory_jobs:
        return []

    electric_types = {vehicle.name: vehicle for vehicle in context.vehicle_types if vehicle.is_electric}
    routes: list[RoutePlan] = []
    used_counts: Counter = Counter()
    ordered_jobs = _sorted_mandatory_jobs(mandatory_jobs, mode)

    for job in ordered_jobs:
        best_option = None
        for route_index, route in enumerate(routes):
            candidate_route, delta = _best_inserted_route(context, route, job, tradeoff_lambda)
            if candidate_route is None:
                continue
            option = (delta, 0, route_index, candidate_route)
            if best_option is None or option < best_option:
                best_option = option

        chosen_vehicle = _select_new_ev_vehicle(electric_types, used_counts, job)
        if chosen_vehicle is not None:
            new_route = build_route_plan(context, chosen_vehicle, [job], "problem2")
            option = (
                route_selection_score(new_route, "problem2", tradeoff_lambda),
                1,
                len(routes),
                new_route,
            )
            if best_option is None or option < best_option:
                best_option = option

        if best_option is None:
            return []

        _, option_type, route_index, candidate = best_option
        if option_type == 0:
            routes[route_index] = candidate
        else:
            routes.append(candidate)
            used_counts[candidate.vehicle.name] += 1

    return routes


def _build_optional_ev_routes(
    context: RoutingContext,
    candidate_jobs: list[DeliveryJob],
    reserved_routes: list[RoutePlan],
    tradeoff_gene: float,
) -> list[RoutePlan]:
    if tradeoff_gene < 0.65 or not candidate_jobs:
        return []

    electric_small = next((vehicle for vehicle in context.vehicle_types if vehicle.is_electric), None)
    if electric_small is None:
        return []

    scheduled = evaluate_solution(context, list(reserved_routes), "problem2") if reserved_routes else None
    used_counts = Counter(scheduled.metadata.get("physical_vehicle_usage", {})) if scheduled is not None else Counter()
    spare = max(0, electric_small.count - used_counts[electric_small.name])
    if spare <= 0:
        return []

    ordered_jobs = sorted(
        [job for job in candidate_jobs if job.green_zone == 1],
        key=lambda job: (job.earliest, -job.weight, job.customer_id, job.split_index),
    )
    optional_routes: list[list[DeliveryJob]] = []
    route_loads: list[tuple[float, float]] = []
    for job in ordered_jobs:
        placed = False
        for idx, (weight, volume) in enumerate(route_loads):
            if weight + job.weight - electric_small.capacity_weight > 1e-6:
                continue
            if volume + job.volume - electric_small.capacity_volume > 1e-6:
                continue
            optional_routes[idx].append(job)
            route_loads[idx] = (weight + job.weight, volume + job.volume)
            placed = True
            break
        if placed:
            continue
        if len(optional_routes) >= spare:
            continue
        optional_routes.append([job])
        route_loads.append((job.weight, job.volume))

    return [build_route_plan(context, electric_small, jobs, "problem2") for jobs in optional_routes]


def _remaining_vehicle_context(
    context: RoutingContext,
    reserved_routes: list[RoutePlan],
    remaining_jobs: list[DeliveryJob],
) -> RoutingContext | None:
    scheduled = evaluate_solution(context, list(reserved_routes), "problem2") if reserved_routes else None
    used_counts = Counter(scheduled.metadata.get("physical_vehicle_usage", {})) if scheduled is not None else Counter()
    vehicle_types = []
    for vehicle in context.vehicle_types:
        remaining_count = vehicle.count - used_counts[vehicle.name]
        if remaining_count < 0:
            return None
        vehicle_types.append(replace(vehicle, count=remaining_count))
    return replace(context, jobs=remaining_jobs, vehicle_types=vehicle_types)


def _policy_candidate(
    context: RoutingContext,
    individual: dict,
    config: Problem2Config,
) -> Solution | None:
    tradeoff_gene = float(individual["tradeoff"])
    tradeoff_lambda = _tradeoff_lambda(tradeoff_gene)
    mandatory_mode = individual.get("mandatory_mode", "earliest")
    mandatory_jobs = [job for job in context.jobs if _is_mandatory_green(job)]
    mandatory_job_keys = {job.job_key for job in mandatory_jobs}

    reserved_routes = _build_mandatory_ev_routes(
        context=context,
        mandatory_jobs=mandatory_jobs,
        tradeoff_lambda=tradeoff_lambda,
        mode=mandatory_mode,
    )
    if mandatory_jobs and not reserved_routes:
        return None

    remaining_jobs = [job for job in context.jobs if job.job_key not in mandatory_job_keys]
    optional_routes = _build_optional_ev_routes(context, remaining_jobs, reserved_routes, tradeoff_gene)
    optional_job_keys = {job.job_key for route in optional_routes for job in route.jobs}
    remaining_jobs = [job for job in remaining_jobs if job.job_key not in optional_job_keys]

    remaining_context = _remaining_vehicle_context(context, reserved_routes + optional_routes, remaining_jobs)
    if remaining_context is None:
        return None

    key_to_index = {job.job_key: idx for idx, job in enumerate(remaining_context.jobs)}
    remaining_permutation = []
    for original_index in individual["perm"]:
        job_key = context.jobs[original_index].job_key
        if job_key in key_to_index:
            remaining_permutation.append(key_to_index[job_key])
    if len(remaining_permutation) != len(remaining_context.jobs):
        seen = set(remaining_permutation)
        for idx in range(len(remaining_context.jobs)):
            if idx not in seen:
                remaining_permutation.append(idx)

    tail_solution = decode_permutation(
        context=remaining_context,
        permutation=remaining_permutation,
        variant="problem2",
        tradeoff_lambda=tradeoff_lambda,
        max_segment_length=config.max_segment_length,
    )
    combined_solution = evaluate_solution(context, reserved_routes + optional_routes + tail_solution.routes, "problem2")
    combined_solution.metadata["tradeoff_gene"] = round(tradeoff_gene, 4)
    combined_solution.metadata["tradeoff_lambda"] = tradeoff_lambda
    combined_solution.metadata["mandatory_mode"] = mandatory_mode
    combined_solution.metadata["vehicle_mix"] = physical_vehicle_mix(combined_solution)
    combined_solution.metadata["permutation"] = list(individual["perm"])
    return combined_solution


def _ordered_crossover(parent_a: list[int], parent_b: list[int], rng: random.Random) -> list[int]:
    if len(parent_a) < 2:
        return list(parent_a)
    left, right = sorted(rng.sample(range(len(parent_a)), 2))
    child = [-1] * len(parent_a)
    child[left:right] = parent_a[left:right]
    filler = [gene for gene in parent_b if gene not in child]
    cursor = 0
    for idx in range(len(child)):
        if child[idx] == -1:
            child[idx] = filler[cursor]
            cursor += 1
    return child


def _mutate_permutation(permutation: list[int], rng: random.Random) -> list[int]:
    child = list(permutation)
    if len(child) < 2:
        return child
    left, right = sorted(rng.sample(range(len(child)), 2))
    child[left:right] = reversed(child[left:right])
    return child


def _perturb(base: list[int], rng: random.Random, strength: int) -> list[int]:
    candidate = list(base)
    for _ in range(max(1, strength)):
        candidate = _mutate_permutation(candidate, rng)
    return candidate


def _seed_population(
    context: RoutingContext,
    baseline_solution: Solution | None,
    config: Problem2Config,
) -> list[dict]:
    rng = random.Random(config.seed)
    base = baseline_solution.metadata.get("permutation") if baseline_solution is not None else None
    if not base:
        base = base_permutation(context)

    genes = np.linspace(0.05, 0.95, num=max(6, config.population_size))
    modes = ["earliest", "largest"]
    population = []
    for idx in range(config.population_size):
        strength = 2 + idx % 5 + len(base) // 45
        perm = _perturb(base, rng, strength=strength) if idx > 0 else list(base)
        if idx >= config.population_size // 2:
            gene = float(np.clip(genes[idx % len(genes)] + rng.uniform(-0.05, 0.08), 0.0, 1.0))
        else:
            gene = float(np.clip(0.55 + 0.4 * rng.random(), 0.0, 1.0))
        population.append({"perm": perm, "tradeoff": gene, "mandatory_mode": modes[idx % len(modes)]})
    return population


def _evaluate(
    context: RoutingContext,
    individual: dict,
    config: Problem2Config,
    cache: dict[tuple[tuple[int, ...], float, str], Solution],
) -> Solution:
    gene = round(float(individual["tradeoff"]), 4)
    mode = str(individual.get("mandatory_mode", "earliest"))
    key = (tuple(individual["perm"]), gene, mode)
    if key not in cache:
        solution = _policy_candidate(context, individual, config)
        if solution is None:
            solution = decode_permutation(
                context=context,
                permutation=individual["perm"],
                variant="problem2",
                tradeoff_lambda=_tradeoff_lambda(gene),
                max_segment_length=config.max_segment_length,
            )
        solution.metadata["tradeoff_gene"] = gene
        solution.metadata["mandatory_mode"] = mode
        solution.metadata["tradeoff_lambda"] = _tradeoff_lambda(gene)
        solution.metadata["vehicle_mix"] = physical_vehicle_mix(solution)
        cache[key] = solution
    return cache[key]


def _objective_pair(solution: Solution) -> tuple[float, float]:
    if solution.feasible:
        return solution.total_cost, solution.total_carbon
    penalty = LARGE_PENALTY + solution.total_cost
    return penalty, penalty


def _dominates(obj_a: tuple[float, float], obj_b: tuple[float, float]) -> bool:
    return (obj_a[0] <= obj_b[0] and obj_a[1] <= obj_b[1]) and (obj_a[0] < obj_b[0] or obj_a[1] < obj_b[1])


def _fast_non_dominated_sort(objectives: list[tuple[float, float]]) -> list[list[int]]:
    dominates = [[] for _ in objectives]
    domination_count = [0 for _ in objectives]
    fronts: list[list[int]] = [[]]

    for i, objective_i in enumerate(objectives):
        for j, objective_j in enumerate(objectives):
            if i == j:
                continue
            if _dominates(objective_i, objective_j):
                dominates[i].append(j)
            elif _dominates(objective_j, objective_i):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[int] = []
        for i in fronts[front_index]:
            for j in dominates[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return fronts


def _crowding_distance(front: list[int], objectives: list[tuple[float, float]]) -> dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: math.inf for idx in front}

    distances = {idx: 0.0 for idx in front}
    for objective_dim in (0, 1):
        sorted_front = sorted(front, key=lambda idx: objectives[idx][objective_dim])
        distances[sorted_front[0]] = math.inf
        distances[sorted_front[-1]] = math.inf
        objective_min = objectives[sorted_front[0]][objective_dim]
        objective_max = objectives[sorted_front[-1]][objective_dim]
        if abs(objective_max - objective_min) < 1e-9:
            continue
        for position in range(1, len(sorted_front) - 1):
            prev_idx = sorted_front[position - 1]
            next_idx = sorted_front[position + 1]
            current_idx = sorted_front[position]
            distances[current_idx] += (
                objectives[next_idx][objective_dim] - objectives[prev_idx][objective_dim]
            ) / (objective_max - objective_min)
    return distances


def _rank_population(objectives: list[tuple[float, float]]) -> tuple[list[list[int]], dict[int, int], dict[int, float]]:
    fronts = _fast_non_dominated_sort(objectives)
    rank_map: dict[int, int] = {}
    distance_map: dict[int, float] = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            rank_map[idx] = rank
        distance_map.update(_crowding_distance(front, objectives))
    return fronts, rank_map, distance_map


def _tournament(
    population: list[dict],
    rank_map: dict[int, int],
    distance_map: dict[int, float],
    rng: random.Random,
) -> dict:
    a_idx, b_idx = rng.sample(range(len(population)), 2)
    a_rank, b_rank = rank_map[a_idx], rank_map[b_idx]
    if a_rank < b_rank:
        return population[a_idx]
    if b_rank < a_rank:
        return population[b_idx]
    if distance_map.get(a_idx, 0.0) >= distance_map.get(b_idx, 0.0):
        return population[a_idx]
    return population[b_idx]


def _make_child(parent_a: dict, parent_b: dict, config: Problem2Config, rng: random.Random) -> dict:
    child_perm = _ordered_crossover(parent_a["perm"], parent_b["perm"], rng)
    if rng.random() < config.mutation_rate:
        child_perm = _mutate_permutation(child_perm, rng)

    tradeoff = 0.5 * (parent_a["tradeoff"] + parent_b["tradeoff"]) + rng.uniform(-0.04, 0.04)
    if rng.random() < config.mutation_rate:
        tradeoff += rng.gauss(0.0, config.tradeoff_mutation_sigma)
    tradeoff = float(np.clip(tradeoff, 0.0, 1.0))
    modes = ["earliest", "largest"]
    mandatory_mode = parent_a.get("mandatory_mode", "earliest") if rng.random() < 0.5 else parent_b.get("mandatory_mode", "earliest")
    if rng.random() < 0.15:
        mandatory_mode = rng.choice(modes)
    return {"perm": child_perm, "tradeoff": tradeoff, "mandatory_mode": mandatory_mode}


def _environmental_selection(
    combined_population: list[dict],
    combined_solutions: list[Solution],
    config: Problem2Config,
) -> tuple[list[dict], list[Solution], dict]:
    objectives = [_objective_pair(solution) for solution in combined_solutions]
    fronts, _, distance_map = _rank_population(objectives)

    next_population: list[dict] = []
    next_solutions: list[Solution] = []
    for front in fronts:
        if len(next_population) + len(front) <= config.population_size:
            for idx in front:
                next_population.append(combined_population[idx])
                next_solutions.append(combined_solutions[idx])
            continue
        ordered_front = sorted(front, key=lambda idx: distance_map.get(idx, 0.0), reverse=True)
        remaining = config.population_size - len(next_population)
        for idx in ordered_front[:remaining]:
            next_population.append(combined_population[idx])
            next_solutions.append(combined_solutions[idx])
        break

    info = {"fronts": fronts, "distance_map": distance_map}
    return next_population, next_solutions, info


def _kmeans_representatives(points: np.ndarray, k: int, seed: int) -> list[int]:
    if len(points) <= k:
        return list(range(len(points)))
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=k, replace=False)]
    labels = np.zeros(len(points), dtype=int)
    for _ in range(40):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centers = []
        for idx in range(k):
            members = points[labels == idx]
            if len(members) == 0:
                new_centers.append(centers[idx])
            else:
                new_centers.append(members.mean(axis=0))
        new_centers = np.asarray(new_centers)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    representatives = []
    for idx in range(k):
        members = np.where(labels == idx)[0]
        if len(members) == 0:
            continue
        distances = np.linalg.norm(points[members] - centers[idx], axis=1)
        representatives.append(int(members[int(np.argmin(distances))]))
    return sorted(set(representatives))


def _entropy_topsis(records: list[dict]) -> tuple[list[dict], dict]:
    matrix = np.asarray([[record["total_cost"], record["total_carbon"]] for record in records], dtype=float)
    eps = 1e-9
    max_values = matrix.max(axis=0)
    min_values = matrix.min(axis=0)
    normalized = (max_values - matrix) / (max_values - min_values + eps)
    proportion = normalized / (normalized.sum(axis=0, keepdims=True) + eps)
    entropy_scale = 1.0 / math.log(max(2, len(records)))
    entropy = -entropy_scale * np.sum(proportion * np.log(proportion + eps), axis=0)
    weights = (1.0 - entropy) / np.sum(1.0 - entropy)

    weighted = normalized * weights
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    dist_best = np.linalg.norm(weighted - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    closeness = dist_worst / (dist_best + dist_worst + eps)

    enriched = []
    for record, score in zip(records, closeness):
        enriched_record = dict(record)
        enriched_record["topsis_score"] = float(score)
        enriched.append(enriched_record)
    return enriched, {"cost_weight": float(weights[0]), "carbon_weight": float(weights[1])}


def solve_problem2(
    context: RoutingContext,
    baseline_solution: Solution | None = None,
    config: Problem2Config | None = None,
) -> dict:
    config = config or Problem2Config()
    rng = random.Random(config.seed)
    population = _seed_population(context, baseline_solution, config)
    cache: dict[tuple[tuple[int, ...], float, str], Solution] = {}
    archive: dict[tuple[float, float, float, int], Solution] = {}
    history = []

    population_solutions = [_evaluate(context, individual, config, cache) for individual in population]
    for generation in range(config.generations):
        objectives = [_objective_pair(solution) for solution in population_solutions]
        fronts, rank_map, distance_map = _rank_population(objectives)

        feasible_population = [solution for solution in population_solutions if solution.feasible]
        pareto_feasible = 0
        if fronts:
            pareto_feasible = sum(1 for idx in fronts[0] if population_solutions[idx].feasible)
        history.append(
            {
                "generation": generation + 1,
                "feasible_count": len(feasible_population),
                "pareto_size": pareto_feasible,
                "best_cost": min((solution.total_cost for solution in feasible_population), default=float("inf")),
                "best_carbon": min((solution.total_carbon for solution in feasible_population), default=float("inf")),
            }
        )

        if config.progress_callback and ((generation + 1 == 1) or ((generation + 1) % 10 == 0) or (generation + 1 == config.generations)):
            config.progress_callback(
                generation + 1,
                config.generations,
                len(feasible_population),
                min((solution.total_cost for solution in feasible_population), default=float("inf")),
                min((solution.total_carbon for solution in feasible_population), default=float("inf")),
            )

        for solution in feasible_population:
            record_key = (
                round(solution.total_cost, 2),
                round(solution.total_carbon, 2),
                round(solution.metadata.get("tradeoff_gene", 0.0), 4),
                len(solution.routes),
            )
            archive[record_key] = solution

        offspring = []
        while len(offspring) < config.population_size:
            parent_a = _tournament(population, rank_map, distance_map, rng)
            parent_b = _tournament(population, rank_map, distance_map, rng)
            offspring.append(_make_child(parent_a, parent_b, config, rng))

        offspring_solutions = [_evaluate(context, individual, config, cache) for individual in offspring]
        combined_population = population + offspring
        combined_solutions = population_solutions + offspring_solutions
        population, population_solutions, _ = _environmental_selection(combined_population, combined_solutions, config)

    archived_solutions = list(archive.values())
    if not archived_solutions:
        archived_solutions = [solution for solution in population_solutions if solution.feasible]
    if not archived_solutions:
        raise RuntimeError("问题2未找到任何可行解，请检查模型参数或约束实现。")

    archive_objectives = [(solution.total_cost, solution.total_carbon) for solution in archived_solutions]
    archive_fronts = _fast_non_dominated_sort(archive_objectives)
    pareto_indices = archive_fronts[0] if archive_fronts else list(range(len(archived_solutions)))

    pareto_records = []
    for idx in pareto_indices:
        solution = archived_solutions[idx]
        pareto_records.append(
            {
                "idx": idx,
                "total_cost": solution.total_cost,
                "total_carbon": solution.total_carbon,
                "num_routes": len(solution.routes),
                "tradeoff_gene": solution.metadata.get("tradeoff_gene", 0.0),
                "vehicle_mix": solution.metadata.get("vehicle_mix", physical_vehicle_mix(solution)),
                "solution": solution,
            }
        )
    pareto_records.sort(key=lambda item: (item["total_cost"], item["total_carbon"]))

    points = np.asarray([[record["total_cost"], record["total_carbon"]] for record in pareto_records], dtype=float)
    normalized_points = (points - points.mean(axis=0)) / (points.std(axis=0) + 1e-9)
    representative_indices = _kmeans_representatives(
        normalized_points,
        k=min(config.representative_count, len(pareto_records)),
        seed=config.seed,
    )
    representatives = [pareto_records[index] for index in representative_indices]
    scored_representatives, weight_record = _entropy_topsis(representatives)
    scored_representatives.sort(key=lambda item: item["topsis_score"], reverse=True)
    selected_record = scored_representatives[0]
    selected_solution = selected_record["solution"]
    selected_solution.metadata["history"] = history
    selected_solution.metadata["entropy_weights"] = weight_record
    selected_solution.metadata["vehicle_mix"] = physical_vehicle_mix(selected_solution)

    baseline_comparison = None
    if baseline_solution is not None:
        baseline_comparison = {
            "baseline_cost": baseline_solution.total_cost,
            "policy_cost": selected_solution.total_cost,
            "cost_change_pct": (selected_solution.total_cost - baseline_solution.total_cost)
            / max(1e-9, baseline_solution.total_cost)
            * 100.0,
            "baseline_carbon": baseline_solution.total_carbon,
            "policy_carbon": selected_solution.total_carbon,
            "carbon_change_pct": (selected_solution.total_carbon - baseline_solution.total_carbon)
            / max(1e-9, baseline_solution.total_carbon)
            * 100.0,
        }

    return {
        "pareto_records": pareto_records,
        "representatives": scored_representatives,
        "selected_solution": selected_solution,
        "entropy_weights": weight_record,
        "history": history,
        "baseline_comparison": baseline_comparison,
        "data_audit": data_quality_audit(context),
        "config": config,
    }


def write_problem2_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pareto_rows = []
    for record in result["pareto_records"]:
        mix = record["vehicle_mix"]
        pareto_rows.append(
            {
                "total_cost": record["total_cost"],
                "total_carbon": record["total_carbon"],
                "num_routes": record["num_routes"],
                "tradeoff_gene": record["tradeoff_gene"],
                "fuel_vehicles": mix.get("燃油车", 0),
                "ev_vehicles": mix.get("新能源车", 0),
            }
        )
    pd.DataFrame(pareto_rows).to_csv(output_dir / "pareto_front.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(result["representatives"]).drop(columns=["solution"], errors="ignore").to_csv(
        output_dir / "representatives_topsis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(result["history"]).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")

    if result["baseline_comparison"] is not None:
        pd.DataFrame([result["baseline_comparison"]]).to_csv(
            output_dir / "policy_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )

    selected = result["selected_solution"]
    pd.DataFrame(route_rows(selected)).to_csv(output_dir / "selected_route_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stop_rows(selected)).to_csv(output_dir / "selected_route_stops.csv", index=False, encoding="utf-8-sig")

    audit = result.get("data_audit", {})
    summary_lines = [
        "问题2 绿色配送区限行双目标结果",
        f"Pareto可行解数量: {len(result['pareto_records'])}",
        f"折中方案总成本: {selected.total_cost:.2f}",
        f"折中方案总碳排放: {selected.total_carbon:.2f} kg",
        f"路线数: {len(selected.routes)}",
        f"车辆结构: {selected.metadata.get('vehicle_mix', {})}",
        f"熵权: {selected.metadata.get('entropy_weights', {})}",
        f"绿色区客户数(附件坐标≤10km): {audit.get('green_customer_count', '未知')}",
        f"有需求绿色区客户数: {audit.get('active_green_customer_count', '未知')}",
        "绿色区口径说明: 附件坐标按距市中心≤10km计算；题面30个与附件计算不一致时，以附件数据为准。",
        "启动成本计费口径: 按当日实际启用物理车辆计，同一车辆多趟配送不重复计启动成本。",
    ]
    if result["baseline_comparison"] is not None:
        summary_lines.extend(
            [
                f"相对问题1成本变化率: {result['baseline_comparison']['cost_change_pct']:.2f}%",
                f"相对问题1碳排变化率: {result['baseline_comparison']['carbon_change_pct']:.2f}%",
            ]
        )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
