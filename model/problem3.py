from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
import random
import time
import re

import pandas as pd

from .core import (
    DeliveryJob,
    RoutingContext,
    RoutePlan,
    Solution,
    build_route_plan,
    create_virtual_jobs,
    decode_permutation,
    evaluate_solution,
    improve_route_two_opt,
    route_rows,
    stop_rows,
)


@dataclass
class Problem3Config:
    event_time: float = 810.0
    route_change_weight: float = 1200.0
    # 旧版默认 50 次 LNS 会导致问题3耗时达到数百秒；
    # 修正后默认走秒级快速修复，必要时才在短时间限额内做 LNS 改善。
    search_iterations: int = 8
    destroy_fraction: float = 0.20
    max_segment_length: int = 16
    time_limit_seconds: float = 8.0
    fast_repair_job_threshold: int = 35
    seed: int = 20260427


@dataclass
class DynamicEvent:
    kind: str
    customer_id: int
    weight: float = 0.0
    volume: float = 0.0
    earliest: float | None = None
    latest: float | None = None
    new_customer_id: int | None = None
    description: str = ""


@dataclass
class VehicleState:
    assigned_vehicle_id: str
    vehicle_name: str
    current_node: int
    available_time: float
    startup_required: bool
    source: str
    remaining_jobs: list[DeliveryJob]


# 论文/报告用：描述事件对系统的“影响强度”（可与题目文字一致）
EVENT_IMPACT_SCORE = {
    "取消": 0.9,
    "新增": 0.6,
    "地址变更": 0.5,
    "时间窗调整": 0.3,
}

# 策略切换用：与阈值 S_E=0.5、D=0.2 配套，使「单取消 + 低扰动」走贪婪插入，
# 「单新增 / 多事件」仍倾向 LNS（与第三问文档的自适应逻辑一致）
STRATEGY_URGENCY_SCORE = {
    "取消": 0.35,
    "新增": 0.55,
    "地址变更": 0.55,
    "时间窗调整": 0.25,
}

JOB_KEY_PATTERN = re.compile(r"(?:DYN-)?C(\d+)-S\d+")


def _copy_job(job: DeliveryJob, **updates) -> DeliveryJob:
    data = {
        "index": job.index,
        "job_key": job.job_key,
        "customer_id": job.customer_id,
        "split_index": job.split_index,
        "split_count": job.split_count,
        "x": job.x,
        "y": job.y,
        "green_zone": job.green_zone,
        "weight": job.weight,
        "volume": job.volume,
        "earliest": job.earliest,
        "latest": job.latest,
    }
    data.update(updates)
    return DeliveryJob(**data)


def _job_at_customer(
    context: RoutingContext,
    job: DeliveryJob,
    customer_id: int,
    earliest: float | None = None,
    latest: float | None = None,
) -> DeliveryJob:
    customer = context.customer_info(customer_id)
    return _copy_job(
        job,
        customer_id=int(customer_id),
        x=float(customer["X (km)"]),
        y=float(customer["Y (km)"]),
        green_zone=int(customer["绿色区"]),
        earliest=float(customer["最早_min"] if earliest is None else earliest),
        latest=float(customer["最晚_min"] if latest is None else latest),
    )


def _route_current_node(route: RoutePlan, event_time: float) -> int:
    for job, stop in zip(route.jobs, route.metrics.stops):
        if stop.arrival <= event_time < stop.service_start:
            return job.customer_id

    last_completed = route.start_node
    for job, stop in zip(route.jobs, route.metrics.stops):
        if stop.departure <= event_time:
            last_completed = job.customer_id
        else:
            break
    return last_completed


def _remaining_jobs_from_route(route: RoutePlan, event_time: float) -> list[DeliveryJob]:
    remaining = []
    for job, stop in zip(route.jobs, route.metrics.stops):
        if stop.service_start >= event_time:
            remaining.append(job)
    return remaining


def _extract_dynamic_context(
    context: RoutingContext,
    baseline_solution: Solution,
    event_time: float,
) -> tuple[list[VehicleState], list[VehicleState], list[DeliveryJob], list[str], dict[str, str]]:
    active_states: list[VehicleState] = []
    depot_states: list[VehicleState] = []
    flexible_jobs: list[DeliveryJob] = []
    flexible_order: list[str] = []
    baseline_vehicle_lookup: dict[str, str] = {}

    routes_by_vehicle: dict[str, list[RoutePlan]] = {}
    for route in baseline_solution.routes:
        if route.assigned_vehicle_id is None:
            continue
        routes_by_vehicle.setdefault(route.assigned_vehicle_id, []).append(route)

    used_counts = Counter()
    next_id_by_type: dict[str, int] = {}
    for vehicle_id, vehicle_routes in routes_by_vehicle.items():
        vehicle_routes.sort(key=lambda item: (item.start_time, item.metrics.route_end_time))
        vehicle_name = vehicle_routes[0].vehicle.name
        used_counts[vehicle_name] += 1
        suffix = vehicle_id.rsplit("-", maxsplit=1)[-1]
        if suffix.isdigit():
            next_id_by_type[vehicle_name] = max(next_id_by_type.get(vehicle_name, 1), int(suffix) + 1)

        active_route = next(
            (route for route in vehicle_routes if route.start_time <= event_time < route.metrics.route_end_time),
            None,
        )
        started_before_event = any(route.start_time < event_time for route in vehicle_routes)

        for route in vehicle_routes:
            for job in _remaining_jobs_from_route(route, event_time):
                baseline_vehicle_lookup[job.job_key] = vehicle_id

        if active_route is not None:
            remaining_jobs = _remaining_jobs_from_route(active_route, event_time)
            if remaining_jobs:
                active_states.append(
                    VehicleState(
                        assigned_vehicle_id=vehicle_id,
                        vehicle_name=vehicle_name,
                        current_node=_route_current_node(active_route, event_time),
                        available_time=event_time,
                        startup_required=False,
                        source="active",
                        remaining_jobs=remaining_jobs,
                    )
                )
            active_index = vehicle_routes.index(active_route)
            for route in vehicle_routes[active_index + 1 :]:
                for job in route.jobs:
                    flexible_jobs.append(job)
                    flexible_order.append(job.job_key)
        else:
            depot_states.append(
                VehicleState(
                    assigned_vehicle_id=vehicle_id,
                    vehicle_name=vehicle_name,
                    current_node=0,
                    available_time=event_time,
                    startup_required=not started_before_event,
                    source="planned" if not started_before_event else "depot_reuse",
                    remaining_jobs=[],
                )
            )
            for route in vehicle_routes:
                if route.start_time >= event_time:
                    for job in route.jobs:
                        flexible_jobs.append(job)
                        flexible_order.append(job.job_key)

    for vehicle in context.vehicle_types:
        next_instance = next_id_by_type.get(vehicle.name, 1)
        remaining = vehicle.count - used_counts.get(vehicle.name, 0)
        for _ in range(max(0, remaining)):
            depot_states.append(
                VehicleState(
                    assigned_vehicle_id=f"{vehicle.name}-{next_instance}",
                    vehicle_name=vehicle.name,
                    current_node=0,
                    available_time=event_time,
                    startup_required=True,
                    source="unused",
                    remaining_jobs=[],
                )
            )
            next_instance += 1

    return active_states, depot_states, flexible_jobs, flexible_order, baseline_vehicle_lookup


def _apply_events_to_jobs(
    context: RoutingContext,
    jobs: list[DeliveryJob],
    events: list[DynamicEvent],
    include_new_jobs: bool = True,
) -> tuple[list[DeliveryJob], dict]:
    cancel_customers = {event.customer_id for event in events if event.kind == "取消"}
    time_updates = {event.customer_id: event for event in events if event.kind == "时间窗调整"}
    address_updates = {event.customer_id: event for event in events if event.kind == "地址变更"}

    affected_customers = set()
    updated_jobs = []
    for job in jobs:
        if job.customer_id in cancel_customers:
            affected_customers.add(job.customer_id)
            continue

        candidate = job
        if job.customer_id in address_updates:
            event = address_updates[job.customer_id]
            target_customer = event.new_customer_id if event.new_customer_id is not None else job.customer_id
            candidate = _job_at_customer(
                context=context,
                job=candidate,
                customer_id=int(target_customer),
                earliest=event.earliest if event.earliest is not None else candidate.earliest,
                latest=event.latest if event.latest is not None else candidate.latest,
            )
            affected_customers.add(job.customer_id)

        if job.customer_id in time_updates:
            event = time_updates[job.customer_id]
            candidate = _copy_job(
                candidate,
                earliest=candidate.earliest if event.earliest is None else float(event.earliest),
                latest=candidate.latest if event.latest is None else float(event.latest),
            )
            affected_customers.add(job.customer_id)

        updated_jobs.append(candidate)

    new_jobs = []
    if include_new_jobs:
        next_index = max((job.index for job in updated_jobs), default=-1) + 1
        for event in events:
            if event.kind != "新增":
                continue
            created = create_virtual_jobs(
                context=context,
                customer_id=event.customer_id,
                weight=event.weight,
                volume=event.volume,
                earliest=event.earliest,
                latest=event.latest,
                prefix="DYN",
                index_start=next_index,
            )
            next_index += len(created)
            new_jobs.extend(created)
            affected_customers.add(event.customer_id)

    return updated_jobs + new_jobs, {"affected_customers": sorted(affected_customers)}


def _sequence_penalty(order: list[str], baseline_pos: dict[str, int]) -> float:
    if not order:
        return 0.0
    total = 0.0
    for index, key in enumerate(order):
        if key in baseline_pos:
            total += abs(index - baseline_pos[key])
        else:
            total += len(order) * 0.5
    return total / len(order)


def _assignment_change_penalty(
    baseline_lookup: dict[str, str],
    final_lookup: dict[str, str],
    events: list[DynamicEvent],
    route_change_weight: float,
) -> float:
    """按车辆分配变化计入扰动惩罚，避免 fallback 后“实际改线但惩罚为0”。

    取消订单不视为路线调整惩罚；未取消且仍需服务的历史任务若换车或漏服务，
    以及新增动态任务，都计入轻量惩罚。
    """
    cancel_customers = {event.customer_id for event in events if event.kind == "取消"}
    comparable = [
        key
        for key in baseline_lookup
        if (job_customer_id(key) is None or job_customer_id(key) not in cancel_customers)
    ]
    changed = sum(
        1
        for key in comparable
        if key in final_lookup and baseline_lookup.get(key, "") != final_lookup.get(key, "")
    )
    missing = sum(1 for key in comparable if key not in final_lookup)
    new_jobs = sum(1 for key in final_lookup if key not in baseline_lookup)
    # 使用“变动任务数 × 权重”的口径，直观表示调度扰动成本；
    # 取消订单本身不计惩罚，新增订单按半个变动任务计。
    return float(route_change_weight) * float(changed + missing + 0.5 * new_jobs)


def _final_route_change_penalty(
    final_order: list[str],
    baseline_pos: dict[str, int],
    baseline_lookup: dict[str, str],
    final_lookup: dict[str, str],
    events: list[DynamicEvent],
    route_change_weight: float,
) -> float:
    sequence_component = _sequence_penalty(final_order, baseline_pos) * float(route_change_weight)
    assignment_component = _assignment_change_penalty(
        baseline_lookup=baseline_lookup,
        final_lookup=final_lookup,
        events=events,
        route_change_weight=route_change_weight,
    )
    if assignment_component > 0:
        return float(assignment_component)
    # 若车辆未变但顺序确有调整，保留一个有上限的顺序扰动惩罚，避免排序口径导致过度放大。
    return float(min(sequence_component, float(route_change_weight)))


def _order_to_permutation(dynamic_context: RoutingContext, order: list[str]) -> list[int]:
    key_to_index = {job.job_key: idx for idx, job in enumerate(dynamic_context.jobs)}
    return [key_to_index[key] for key in order if key in key_to_index]


def _build_vehicle_pool_context(
    context: RoutingContext,
    event_time: float,
    jobs: list[DeliveryJob],
    depot_states: list[VehicleState],
) -> RoutingContext:
    counts = Counter(state.vehicle_name for state in depot_states)
    vehicle_types = [replace(vehicle, count=counts.get(vehicle.name, 0)) for vehicle in context.vehicle_types]
    return replace(context, jobs=jobs, vehicle_types=vehicle_types, start_minute=float(event_time))


def _assign_depot_states_to_routes(
    vehicle_context: RoutingContext,
    routes: list[RoutePlan],
    depot_states: list[VehicleState],
) -> list[RoutePlan]:
    state_pool: dict[str, list[VehicleState]] = {}
    for state in depot_states:
        state_pool.setdefault(state.vehicle_name, []).append(state)
    for states in state_pool.values():
        states.sort(key=lambda item: (item.startup_required, item.available_time, item.assigned_vehicle_id))

    assigned_routes = []
    ordered_routes = sorted(routes, key=lambda route: (route.start_time, route.metrics.route_end_time))
    for route in ordered_routes:
        candidates = state_pool.get(route.vehicle.name, [])
        if not candidates:
            return []

        chosen_index = 0
        for idx, state in enumerate(candidates):
            if state.available_time <= route.start_time + 1e-6:
                chosen_index = idx
                break
        state = candidates.pop(chosen_index)
        chosen_start = max(route.start_time, state.available_time)
        assigned_routes.append(
            build_route_plan(
                context=vehicle_context,
                vehicle=route.vehicle,
                jobs=route.jobs,
                variant="dynamic",
                start_node=0,
                start_time=chosen_start,
                assigned_vehicle_id=state.assigned_vehicle_id,
                startup_required=state.startup_required,
            )
        )
    return assigned_routes


def _evaluate_flexible_order(
    context: RoutingContext,
    event_time: float,
    flexible_jobs: list[DeliveryJob],
    order: list[str],
    baseline_pos: dict[str, int],
    depot_states: list[VehicleState],
    config: Problem3Config,
) -> tuple[float, Solution]:
    dynamic_context = _build_vehicle_pool_context(context, event_time, flexible_jobs, depot_states)
    permutation = _order_to_permutation(dynamic_context, order)
    decoded = decode_permutation(
        context=dynamic_context,
        permutation=permutation,
        variant="dynamic",
        tradeoff_lambda=0.0,
        max_segment_length=config.max_segment_length,
    )
    mapped_routes = _assign_depot_states_to_routes(dynamic_context, decoded.routes, depot_states)
    if flexible_jobs and not mapped_routes and decoded.routes:
        decoded.metadata["route_change_penalty"] = float("inf")
        decoded.metadata["objective"] = float("inf")
        return float("inf"), decoded

    scheduled = evaluate_solution(dynamic_context, mapped_routes, "dynamic")
    penalty = _sequence_penalty(order, baseline_pos) * config.route_change_weight
    scheduled.metadata["route_change_penalty"] = penalty
    scheduled.metadata["objective"] = scheduled.total_cost + penalty
    scheduled.metadata["final_order"] = order
    return scheduled.metadata["objective"], scheduled


def _greedy_insert(
    context: RoutingContext,
    event_time: float,
    flexible_jobs: list[DeliveryJob],
    baseline_order: list[str],
    new_keys: list[str],
    baseline_pos: dict[str, int],
    depot_states: list[VehicleState],
    config: Problem3Config,
) -> tuple[list[str], Solution]:
    order = [key for key in baseline_order if key not in new_keys]
    best_solution = None
    for key in new_keys:
        best_order = None
        best_score = float("inf")
        for position in range(len(order) + 1):
            candidate = order[:position] + [key] + order[position:]
            score, solution = _evaluate_flexible_order(
                context,
                event_time,
                flexible_jobs,
                candidate,
                baseline_pos,
                depot_states,
                config,
            )
            if score < best_score:
                best_score = score
                best_order = candidate
                best_solution = solution
        order = best_order if best_order is not None else order

    if best_solution is None:
        _, best_solution = _evaluate_flexible_order(
            context,
            event_time,
            flexible_jobs,
            order,
            baseline_pos,
            depot_states,
            config,
        )
    return order, best_solution


def _lns_search(
    context: RoutingContext,
    event_time: float,
    flexible_jobs: list[DeliveryJob],
    baseline_order: list[str],
    baseline_pos: dict[str, int],
    depot_states: list[VehicleState],
    config: Problem3Config,
) -> tuple[list[str], Solution]:
    rng = random.Random(config.seed)
    current_order = list(baseline_order)
    deadline = time.perf_counter() + max(0.1, float(config.time_limit_seconds))
    best_score, best_solution = _evaluate_flexible_order(
        context,
        event_time,
        flexible_jobs,
        current_order,
        baseline_pos,
        depot_states,
        config,
    )
    best_order = list(current_order)

    for _ in range(config.search_iterations):
        if len(best_order) <= 1 or time.perf_counter() >= deadline:
            break
        candidate_order = list(best_order)
        remove_count = max(1, int(len(candidate_order) * config.destroy_fraction))
        removed_indices = sorted(rng.sample(range(len(candidate_order)), remove_count), reverse=True)
        removed_keys = [candidate_order[idx] for idx in removed_indices]
        for idx in removed_indices:
            del candidate_order[idx]
        removed_keys.sort(key=lambda key: baseline_pos.get(key, len(baseline_pos)))

        for key in removed_keys:
            if time.perf_counter() >= deadline:
                break
            preferred = baseline_pos.get(key, len(candidate_order))
            candidate_positions = {0, len(candidate_order), min(len(candidate_order), preferred)}
            if candidate_order:
                candidate_positions.add(min(len(candidate_order), max(0, preferred - 1)))
                candidate_positions.add(min(len(candidate_order), max(0, preferred + 1)))
                candidate_positions.add(rng.randint(0, len(candidate_order)))

            best_local_order = None
            best_local_score = float("inf")
            for position in sorted(candidate_positions):
                if time.perf_counter() >= deadline:
                    break
                trial = candidate_order[:position] + [key] + candidate_order[position:]
                score, _ = _evaluate_flexible_order(
                    context,
                    event_time,
                    flexible_jobs,
                    trial,
                    baseline_pos,
                    depot_states,
                    config,
                )
                if score < best_local_score:
                    best_local_score = score
                    best_local_order = trial
            candidate_order = best_local_order if best_local_order is not None else candidate_order

        candidate_score, candidate_solution = _evaluate_flexible_order(
            context,
            event_time,
            flexible_jobs,
            candidate_order,
            baseline_pos,
            depot_states,
            config,
        )
        if candidate_score < best_score:
            best_score = candidate_score
            best_solution = candidate_solution
            best_order = candidate_order

    return best_order, best_solution


def _optimize_active_route(
    context: RoutingContext,
    state: VehicleState,
    jobs: list[DeliveryJob],
) -> RoutePlan | None:
    if not jobs:
        return None
    vehicle = next(vehicle for vehicle in context.vehicle_types if vehicle.name == state.vehicle_name)
    route = build_route_plan(
        context=context,
        vehicle=vehicle,
        jobs=jobs,
        variant="dynamic",
        start_node=state.current_node,
        start_time=state.available_time,
        assigned_vehicle_id=state.assigned_vehicle_id,
        startup_required=False,
    )
    return improve_route_two_opt(context, route, "dynamic")


def _executed_prefix_stats(baseline_solution: Solution, event_time: float) -> dict:
    executed_cost = 0.0
    executed_carbon = 0.0
    for route in baseline_solution.routes:
        if route.start_time < event_time:
            executed_cost += route.metrics.start_cost

        completed_distance = 0.0
        for stop in route.metrics.stops:
            if stop.arrival <= event_time:
                completed_distance += stop.distance_from_prev
        completed_ratio = min(1.0, completed_distance / max(route.metrics.total_distance, 1e-9))
        executed_cost += completed_ratio * max(0.0, route.metrics.total_cost - route.metrics.start_cost)
        executed_carbon += completed_ratio * route.metrics.carbon_kg
    return {
        "executed_cost_est": executed_cost,
        "executed_carbon_est": executed_carbon,
    }


def _job_route_lookup(solution: Solution) -> dict[str, str]:
    lookup = {}
    for route in solution.routes:
        for job in route.jobs:
            lookup[job.job_key] = route.assigned_vehicle_id or ""
    return lookup


def _build_new_jobs_from_events(
    context: RoutingContext,
    events: list[DynamicEvent],
    index_start: int,
) -> list[DeliveryJob]:
    new_jobs = []
    next_index = index_start
    for event in events:
        if event.kind != "新增":
            continue
        created = create_virtual_jobs(
            context=context,
            customer_id=event.customer_id,
            weight=event.weight,
            volume=event.volume,
            earliest=event.earliest,
            latest=event.latest,
            prefix="DYN",
            index_start=next_index,
        )
        next_index += len(created)
        new_jobs.extend(created)
    return new_jobs


def _fallback_flexible_solution(
    context: RoutingContext,
    baseline_solution: Solution,
    events: list[DynamicEvent],
    event_time: float,
    active_routes: list[RoutePlan],
    depot_states: list[VehicleState],
    baseline_pos: dict[str, int],
    route_change_weight: float,
) -> tuple[Solution, list[str], float]:
    active_end_by_id = {route.assigned_vehicle_id: route.metrics.route_end_time for route in active_routes if route.assigned_vehicle_id}
    availability_by_id = dict(active_end_by_id)
    routes = []
    used_ids = set(active_end_by_id)
    baseline_future_routes = sorted(
        [route for route in baseline_solution.routes if route.start_time >= event_time],
        key=lambda item: (item.start_time, item.metrics.route_end_time),
    )

    for route in baseline_future_routes:
        updated_jobs, _ = _apply_events_to_jobs(context, route.jobs, events, include_new_jobs=False)
        if not updated_jobs:
            continue
        start_time = max(route.start_time, event_time, availability_by_id.get(route.assigned_vehicle_id or "", 0.0))
        rebuilt = build_route_plan(
            context=context,
            vehicle=route.vehicle,
            jobs=updated_jobs,
            variant="dynamic",
            start_node=0,
            start_time=start_time,
            assigned_vehicle_id=route.assigned_vehicle_id,
            startup_required=route.metrics.start_cost > 0,
        )
        routes.append(improve_route_two_opt(context, rebuilt, "dynamic"))
        if route.assigned_vehicle_id:
            used_ids.add(route.assigned_vehicle_id)
            availability_by_id[route.assigned_vehicle_id] = routes[-1].metrics.route_end_time

    available_states = [state for state in depot_states if state.assigned_vehicle_id not in used_ids]
    repaired_routes = []
    for route in routes:
        if not route.metrics.policy_violation:
            repaired_routes.append(route)
            continue
        replacement_index = None
        replacement_route = None
        for state_index, state in enumerate(available_states):
            if "新能源" not in state.vehicle_name:
                continue
            vehicle = next(vehicle for vehicle in context.vehicle_types if vehicle.name == state.vehicle_name)
            if sum(job.weight for job in route.jobs) - vehicle.capacity_weight > 1e-6:
                continue
            if sum(job.volume for job in route.jobs) - vehicle.capacity_volume > 1e-6:
                continue
            candidate = build_route_plan(
                context=context,
                vehicle=vehicle,
                jobs=route.jobs,
                variant="dynamic",
                start_node=0,
                start_time=max(route.start_time, state.available_time),
                assigned_vehicle_id=state.assigned_vehicle_id,
                startup_required=state.startup_required,
            )
            if candidate.metrics.policy_violation or candidate.metrics.capacity_violation:
                continue
            replacement_index = state_index
            replacement_route = candidate
            break
        if replacement_route is not None:
            del available_states[replacement_index]
            repaired_routes.append(improve_route_two_opt(context, replacement_route, "dynamic"))
        else:
            repaired_routes.append(route)
    routes = repaired_routes

    next_index = max((job.index for route in baseline_solution.routes for job in route.jobs), default=-1) + 1
    new_jobs = _build_new_jobs_from_events(context, events, next_index)

    for job in new_jobs:
        best_option = None
        for route_index, route in enumerate(routes):
            if sum(item.weight for item in route.jobs) + job.weight - route.vehicle.capacity_weight > 1e-6:
                continue
            if sum(item.volume for item in route.jobs) + job.volume - route.vehicle.capacity_volume > 1e-6:
                continue
            for position in range(len(route.jobs) + 1):
                candidate_jobs = route.jobs[:position] + [job] + route.jobs[position:]
                candidate = build_route_plan(
                    context=context,
                    vehicle=route.vehicle,
                    jobs=candidate_jobs,
                    variant="dynamic",
                    start_node=0,
                    start_time=max(route.start_time, active_end_by_id.get(route.assigned_vehicle_id or "", 0.0)),
                    assigned_vehicle_id=route.assigned_vehicle_id,
                    startup_required=route.startup_required,
                )
                if candidate.metrics.policy_violation or candidate.metrics.capacity_violation:
                    continue
                delta = candidate.metrics.total_cost - route.metrics.total_cost
                option = (delta, 0, route_index, candidate)
                if best_option is None or option < best_option:
                    best_option = option

        for state_index, state in enumerate(available_states):
            vehicle = next(vehicle for vehicle in context.vehicle_types if vehicle.name == state.vehicle_name)
            if job.weight - vehicle.capacity_weight > 1e-6 or job.volume - vehicle.capacity_volume > 1e-6:
                continue
            candidate = build_route_plan(
                context=context,
                vehicle=vehicle,
                jobs=[job],
                variant="dynamic",
                start_node=0,
                start_time=max(event_time, state.available_time),
                assigned_vehicle_id=state.assigned_vehicle_id,
                startup_required=state.startup_required,
            )
            if candidate.metrics.policy_violation or candidate.metrics.capacity_violation:
                continue
            option = (candidate.metrics.total_cost, 1, state_index, candidate)
            if best_option is None or option < best_option:
                best_option = option

        if best_option is None:
            infeasible = Solution(routes=routes, total_cost=float("inf"), total_carbon=float("inf"), total_distance=0.0, feasible=False, variant="dynamic")
            return infeasible, [], float("inf")

        _, option_type, target_index, candidate = best_option
        if option_type == 0:
            routes[target_index] = improve_route_two_opt(context, candidate, "dynamic")
        else:
            routes.append(candidate)
            del available_states[target_index]

    scheduled = evaluate_solution(context, routes, "dynamic")
    final_order = [
        job.job_key
        for route in sorted(routes, key=lambda item: (item.start_time, item.metrics.route_end_time))
        for job in route.jobs
    ]
    sequence_penalty = _sequence_penalty(final_order, baseline_pos) * float(route_change_weight)
    scheduled.metadata["route_change_penalty"] = sequence_penalty
    scheduled.metadata["objective"] = scheduled.total_cost + sequence_penalty
    scheduled.metadata["final_order"] = final_order
    return scheduled, final_order, sequence_penalty


def _repair_policy_violations(context: RoutingContext, routes: list[RoutePlan]) -> list[RoutePlan]:
    repaired_routes = list(routes)
    for route_index, route in enumerate(repaired_routes):
        if not route.metrics.policy_violation:
            continue
        best_candidate = None
        best_cost = float("inf")
        for vehicle in context.vehicle_types:
            if not vehicle.is_electric:
                continue
            if route.total_weight - vehicle.capacity_weight > 1e-6:
                continue
            if route.total_volume - vehicle.capacity_volume > 1e-6:
                continue
            candidate = build_route_plan(
                context=context,
                vehicle=vehicle,
                jobs=route.jobs,
                variant="dynamic",
                start_node=route.start_node,
                start_time=route.start_time,
                startup_required=route.startup_required,
            )
            if candidate.metrics.policy_violation or candidate.metrics.capacity_violation:
                continue
            trial_routes = list(repaired_routes)
            trial_routes[route_index] = candidate
            trial_solution = evaluate_solution(context, trial_routes, "dynamic")
            if not trial_solution.feasible and trial_solution.total_cost >= best_cost:
                continue
            best_candidate = candidate
            best_cost = trial_solution.total_cost
        if best_candidate is not None:
            repaired_routes[route_index] = best_candidate
    return repaired_routes


def build_default_events(
    context: RoutingContext,
    baseline_solution: Solution,
    config: Problem3Config,
) -> list[DynamicEvent]:
    remaining_jobs = []
    for route in baseline_solution.routes:
        remaining_jobs.extend(_remaining_jobs_from_route(route, config.event_time))
    remaining_customers = list(dict.fromkeys(job.customer_id for job in remaining_jobs))

    zero_demand = context.customer_frame[context.customer_frame["总重量_kg"] <= 0]["客户编号"].tolist()
    candidate_new = [customer_id for customer_id in zero_demand if customer_id not in remaining_customers]
    add_customer = candidate_new[0] if candidate_new else int(context.customer_frame.iloc[0]["客户编号"])

    cancel_customer = remaining_customers[0] if remaining_customers else 2
    tw_customer = remaining_customers[1] if len(remaining_customers) > 1 else cancel_customer
    addr_customer = remaining_customers[2] if len(remaining_customers) > 2 else tw_customer
    addr_target_candidates = [customer_id for customer_id in candidate_new if customer_id != add_customer and customer_id != addr_customer]
    if not addr_target_candidates:
        addr_target_candidates = [
            int(customer_id)
            for customer_id in context.customer_frame["客户编号"].tolist()
            if int(customer_id) not in {cancel_customer, tw_customer, addr_customer, add_customer}
        ]
    addr_target = addr_target_candidates[0]

    tw_info = context.customer_info(tw_customer)
    narrowed_latest = max(float(tw_info["最早_min"]) + 30.0, float(tw_info["最晚_min"]) - 40.0)

    return [
        DynamicEvent(kind="取消", customer_id=int(cancel_customer), description="取消一位未完成客户"),
        DynamicEvent(
            kind="新增",
            customer_id=int(add_customer),
            weight=1200.0,
            volume=3.6,
            earliest=max(config.event_time + 45.0, float(context.customer_info(add_customer)["最早_min"])),
            latest=max(config.event_time + 120.0, float(context.customer_info(add_customer)["最早_min"]) + 90.0),
            description="新增一笔临时配送订单",
        ),
        DynamicEvent(
            kind="地址变更",
            customer_id=int(addr_customer),
            new_customer_id=int(addr_target),
            description="将一位未完成客户改送至新地址",
        ),
        DynamicEvent(
            kind="时间窗调整",
            customer_id=int(tw_customer),
            earliest=float(tw_info["最早_min"]),
            latest=narrowed_latest,
            description="收紧一位未完成客户的最晚服务时间",
        ),
    ]


def solve_problem3(
    context: RoutingContext,
    baseline_solution: Solution,
    config: Problem3Config | None = None,
    events: list[DynamicEvent] | None = None,
) -> dict:
    solve_started = time.perf_counter()
    config = config or Problem3Config()
    events = events or build_default_events(context, baseline_solution, config)

    active_states, depot_states, flexible_jobs, flexible_order, baseline_vehicle_lookup = _extract_dynamic_context(
        context=context,
        baseline_solution=baseline_solution,
        event_time=config.event_time,
    )

    updated_active_routes: list[RoutePlan] = []
    affected_customers = set()
    for state in active_states:
        updated_jobs, info = _apply_events_to_jobs(context, state.remaining_jobs, events, include_new_jobs=False)
        affected_customers.update(info["affected_customers"])
        route = _optimize_active_route(context, state, updated_jobs)
        if route is not None:
            updated_active_routes.append(route)

    updated_flexible_jobs, flexible_info = _apply_events_to_jobs(context, flexible_jobs, events)
    affected_customers.update(flexible_info["affected_customers"])

    baseline_job_keys = {job.job_key for job in flexible_jobs}
    updated_flexible_order = [key for key in flexible_order if key in {job.job_key for job in updated_flexible_jobs}]
    new_keys = [job.job_key for job in updated_flexible_jobs if job.job_key not in baseline_job_keys]
    updated_flexible_order.extend(key for key in new_keys if key not in updated_flexible_order)
    baseline_pos = {key: idx for idx, key in enumerate(flexible_order)}

    impact_score = max(EVENT_IMPACT_SCORE.get(event.kind, 0.5) for event in events)
    urgency = max(STRATEGY_URGENCY_SCORE.get(event.kind, 0.5) for event in events)
    base_customer_count = max(
        1,
        len(
            {
                job.customer_id
                for state in active_states
                for job in state.remaining_jobs
            }.union({job.customer_id for job in flexible_jobs})
        ),
    )
    disturbance = len(affected_customers) / base_customer_count

    strategy_started = time.perf_counter()
    if updated_flexible_jobs:
        use_fast_repair = (
            len(events) > 1
            or len(updated_flexible_jobs) >= int(config.fast_repair_job_threshold)
            or float(config.time_limit_seconds) <= 1.0
        )
        if urgency < 0.5 and disturbance < 0.2 and len(new_keys) <= 3 and not use_fast_repair:
            final_order, flexible_solution = _greedy_insert(
                context=context,
                event_time=config.event_time,
                flexible_jobs=updated_flexible_jobs,
                baseline_order=updated_flexible_order,
                new_keys=new_keys,
                baseline_pos=baseline_pos,
                depot_states=depot_states,
                config=config,
            )
            strategy = "greedy_insert"
        elif use_fast_repair:
            flexible_solution, final_order, route_change_penalty = _fallback_flexible_solution(
                context=context,
                baseline_solution=baseline_solution,
                events=events,
                event_time=config.event_time,
                active_routes=updated_active_routes,
                depot_states=depot_states,
                baseline_pos=baseline_pos,
                route_change_weight=config.route_change_weight,
            )
            strategy = "fast_baseline_repair"
        else:
            seed_order = list(updated_flexible_order)
            final_order, flexible_solution = _lns_search(
                context=context,
                event_time=config.event_time,
                flexible_jobs=updated_flexible_jobs,
                baseline_order=seed_order,
                baseline_pos=baseline_pos,
                depot_states=depot_states,
                config=config,
            )
            strategy = "lns_reoptimize_limited"
        flexible_routes = flexible_solution.routes
        route_change_penalty = flexible_solution.metadata.get("route_change_penalty", 0.0)
    else:
        flexible_solution = Solution(routes=[], total_cost=0.0, total_carbon=0.0, total_distance=0.0, feasible=True, variant="dynamic")
        flexible_routes = []
        final_order = []
        route_change_penalty = 0.0
        strategy = "active_route_adjustment"

    if updated_flexible_jobs and not flexible_solution.feasible:
        original_strategy = strategy
        flexible_solution, final_order, route_change_penalty = _fallback_flexible_solution(
            context=context,
            baseline_solution=baseline_solution,
            events=events,
            event_time=config.event_time,
            active_routes=updated_active_routes,
            depot_states=depot_states,
            baseline_pos=baseline_pos,
            route_change_weight=config.route_change_weight,
        )
        flexible_routes = flexible_solution.routes
        strategy = f"{original_strategy}+baseline_fallback"

    strategy_elapsed = time.perf_counter() - strategy_started

    combined_routes = _repair_policy_violations(context, updated_active_routes + flexible_routes)
    final_solution = evaluate_solution(context, combined_routes, "dynamic")
    executed_prefix = _executed_prefix_stats(baseline_solution, config.event_time)
    final_job_vehicle = _job_route_lookup(final_solution)
    final_route_change_penalty = _final_route_change_penalty(
        final_order=final_order,
        baseline_pos=baseline_pos,
        baseline_lookup=baseline_vehicle_lookup,
        final_lookup=final_job_vehicle,
        events=events,
        route_change_weight=config.route_change_weight,
    )
    # 最终报告以实际车辆分配/新增任务变化为准，避免 fallback 阶段因路线排序口径放大扰动。
    route_change_penalty = final_route_change_penalty

    final_solution.metadata["events"] = [event.__dict__ for event in events]
    final_solution.metadata["strategy"] = strategy
    final_solution.metadata["urgency"] = urgency
    final_solution.metadata["impact_score"] = impact_score
    final_solution.metadata["disturbance"] = disturbance
    final_solution.metadata["final_order"] = final_order
    final_solution.metadata["route_change_penalty"] = route_change_penalty
    final_solution.metadata["executed_prefix"] = executed_prefix
    final_solution.metadata["remaining_cost"] = final_solution.total_cost
    final_solution.metadata["remaining_carbon"] = final_solution.total_carbon
    final_solution.metadata["full_day_cost_est"] = executed_prefix["executed_cost_est"] + final_solution.total_cost
    final_solution.metadata["full_day_carbon_est"] = executed_prefix["executed_carbon_est"] + final_solution.total_carbon
    final_solution.metadata["active_routes"] = len(updated_active_routes)
    final_solution.metadata["flexible_routes"] = len(flexible_routes)
    final_solution.metadata["baseline_job_vehicle"] = baseline_vehicle_lookup
    final_solution.metadata["final_job_vehicle"] = final_job_vehicle
    final_solution.metadata["vehicle_states"] = [state.__dict__ for state in active_states + depot_states]
    final_solution.metadata["solve_seconds"] = time.perf_counter() - solve_started
    final_solution.metadata["strategy_seconds"] = strategy_elapsed
    final_solution.metadata["fallback_used"] = "+baseline_fallback" in strategy

    return {
        "solution": final_solution,
        "events": events,
        "strategy": strategy,
        "urgency": urgency,
        "impact_score": impact_score,
        "disturbance": disturbance,
        "vehicle_states": active_states + depot_states,
    }


def write_problem3_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    solution = result["solution"]

    pd.DataFrame(route_rows(solution)).to_csv(output_dir / "dynamic_route_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stop_rows(solution)).to_csv(output_dir / "dynamic_route_stops.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([event.__dict__ for event in result["events"]]).to_csv(
        output_dir / "events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(solution.metadata.get("vehicle_states", [])).to_csv(
        output_dir / "vehicle_states.csv",
        index=False,
        encoding="utf-8-sig",
    )

    route_change_rows = []
    baseline_lookup = solution.metadata.get("baseline_job_vehicle", {})
    final_lookup = solution.metadata.get("final_job_vehicle", {})
    all_keys = sorted(set(baseline_lookup) | set(final_lookup))
    for job_key in all_keys:
        route_change_rows.append(
            {
                "job_key": job_key,
                "baseline_vehicle_id": baseline_lookup.get(job_key, ""),
                "dynamic_vehicle_id": final_lookup.get(job_key, ""),
                "changed": baseline_lookup.get(job_key, "") != final_lookup.get(job_key, ""),
            }
        )
    pd.DataFrame(route_change_rows).to_csv(output_dir / "route_changes.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "问题3 动态事件下的实时车辆调度结果",
        f"策略: {result['strategy']}",
        f"策略紧急度(分支): {result['urgency']:.3f}",
        f"事件影响度(论文): {result['impact_score']:.3f}",
        f"扰动度: {result['disturbance']:.3f}",
        f"事件后剩余成本: {solution.metadata.get('remaining_cost', solution.total_cost):.2f}",
        f"事件后剩余碳排放: {solution.metadata.get('remaining_carbon', solution.total_carbon):.2f}",
        f"估计全天累计成本: {solution.metadata.get('full_day_cost_est', solution.total_cost):.2f}",
        f"估计全天累计碳排放: {solution.metadata.get('full_day_carbon_est', solution.total_carbon):.2f}",
        f"活动中车辆路线数: {solution.metadata.get('active_routes', 0)}",
        f"重新分配路线数: {solution.metadata.get('flexible_routes', 0)}",
        f"路线调整惩罚: {solution.metadata.get('route_change_penalty', 0.0):.2f}",
        f"重调度求解耗时(s): {solution.metadata.get('solve_seconds', 0.0):.3f}",
        f"策略阶段耗时(s): {solution.metadata.get('strategy_seconds', 0.0):.3f}",
        f"事件列表: {[event.description or event.kind for event in result['events']]}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def job_customer_id(job_key: str) -> int | None:
    match = JOB_KEY_PATTERN.fullmatch(str(job_key))
    if not match:
        return None
    return int(match.group(1))
