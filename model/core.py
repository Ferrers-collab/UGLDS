from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import heapq
from pathlib import Path
from typing import Iterable, Optional
import math

import numpy as np
import pandas as pd

START_MINUTE = 480.0
SERVICE_MINUTE = 20.0
WAIT_COST_PER_MIN = 20.0 / 60.0
LATE_COST_PER_MIN = 50.0 / 60.0
FUEL_PRICE = 7.61
ELECTRICITY_PRICE = 1.64
CARBON_PRICE = 0.65
FUEL_EMISSION = 2.547
ELECTRICITY_EMISSION = 0.501
LARGE_PENALTY = 1_000_000.0
BAN_START = 480.0
BAN_END = 960.0


@dataclass(frozen=True)
class SpeedBand:
    name: str
    start_min: float
    end_min: float
    mu_kmph: float
    sigma_kmph: float


@dataclass(frozen=True)
class VehicleType:
    name: str
    capacity_weight: float
    capacity_volume: float
    count: int
    start_cost: float
    power_type: str

    @property
    def is_electric(self) -> bool:
        return self.power_type == "新能源"

    @property
    def alpha(self) -> float:
        return 0.35 if self.is_electric else 0.40


@dataclass(frozen=True)
class DeliveryJob:
    index: int
    job_key: str
    customer_id: int
    split_index: int
    split_count: int
    x: float
    y: float
    green_zone: int
    weight: float
    volume: float
    earliest: float
    latest: float


@dataclass
class StopDetail:
    job_key: str
    customer_id: int
    arrival: float
    service_start: float
    departure: float
    wait_minutes: float
    late_minutes: float
    distance_from_prev: float


@dataclass
class RouteMetrics:
    total_distance: float = 0.0
    fuel_liters: float = 0.0
    electricity_kwh: float = 0.0
    carbon_kg: float = 0.0
    waiting_minutes: float = 0.0
    late_minutes: float = 0.0
    waiting_cost: float = 0.0
    late_cost: float = 0.0
    start_cost: float = 0.0
    fuel_cost: float = 0.0
    electricity_cost: float = 0.0
    carbon_cost: float = 0.0
    total_cost: float = 0.0
    route_end_time: float = 0.0
    capacity_violation: bool = False
    policy_violation: bool = False
    infeasibility_penalty: float = 0.0
    stops: list[StopDetail] = field(default_factory=list)


@dataclass
class RoutePlan:
    vehicle: VehicleType
    jobs: list[DeliveryJob]
    metrics: RouteMetrics
    start_node: int = 0
    start_time: float = START_MINUTE
    assigned_vehicle_id: Optional[str] = None
    startup_required: bool = True

    @property
    def total_weight(self) -> float:
        return sum(job.weight for job in self.jobs)

    @property
    def total_volume(self) -> float:
        return sum(job.volume for job in self.jobs)


@dataclass
class Solution:
    routes: list[RoutePlan]
    total_cost: float
    total_carbon: float
    total_distance: float
    feasible: bool
    variant: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RoutingContext:
    root_dir: Path
    data_dir: Path
    customer_frame: pd.DataFrame
    jobs: list[DeliveryJob]
    distance_matrix: np.ndarray
    speed_bands: list[SpeedBand]
    vehicle_types: list[VehicleType]
    start_minute: float = START_MINUTE
    service_minute: float = SERVICE_MINUTE

    def __post_init__(self) -> None:
        self.customer_lookup = self.customer_frame.set_index("客户编号").to_dict("index")
        self.customer_to_index = {0: 0}
        for customer_id in self.customer_frame["客户编号"].tolist():
            self.customer_to_index[int(customer_id)] = int(customer_id)
        self.vehicle_limits = {vehicle.name: vehicle.count for vehicle in self.vehicle_types}
        self.max_capacity_weight = max(vehicle.capacity_weight for vehicle in self.vehicle_types)
        self.max_capacity_volume = max(vehicle.capacity_volume for vehicle in self.vehicle_types)

    def with_jobs(self, jobs: list[DeliveryJob], start_minute: Optional[float] = None) -> "RoutingContext":
        return replace(
            self,
            jobs=jobs,
            start_minute=self.start_minute if start_minute is None else float(start_minute),
        )

    def customer_info(self, customer_id: int) -> dict:
        return self.customer_lookup[int(customer_id)]


REQUIRED_PREPROCESSED_FILES = {
    "customer_attributes.csv",
    "distance_matrix_clean.csv",
    "speed_profile.csv",
    "vehicle_fleet.csv",
}


def _is_preprocessed_dir(path: Path) -> bool:
    return path.is_dir() and REQUIRED_PREPROCESSED_FILES.issubset({item.name for item in path.iterdir()})


def locate_preprocessed_dir(root_dir: Optional[Path] = None) -> Path:
    """定位预处理结果目录。

    早期提交版硬编码 ``yuchuli/预处理结果``，在部分解压工具或非 UTF-8 环境下，
    中文目录会被还原成 ``#U9884#U5904...`` 形式，导致命令行直接运行失败。
    这里先查常规路径，再在项目目录内递归搜索必要 CSV 文件，因此同时兼容：
    - 正常中文目录名；
    - zip 解压后产生的 ``#U`` 转义目录名；
    - 用户从 model 子目录或项目根目录运行脚本的情况。
    """
    project_root = Path(root_dir).resolve() if root_dir else Path(__file__).resolve().parents[1]
    search_roots: list[Path] = []
    for candidate_root in (project_root, project_root.parent, Path.cwd()):
        candidate_root = candidate_root.resolve()
        if candidate_root.exists() and candidate_root not in search_roots:
            search_roots.append(candidate_root)

    direct_candidates = []
    for base in search_roots:
        direct_candidates.extend(
            [
                base / "yuchuli" / "预处理结果",
                base / "yuchuli" / "#U9884#U5904#U7406#U7ed3#U679c",
                base / "预处理结果",
                base / "#U9884#U5904#U7406#U7ed3#U679c",
            ]
        )
    for candidate in direct_candidates:
        if _is_preprocessed_dir(candidate):
            return candidate

    for base in search_roots:
        for csv_path in base.rglob("customer_attributes.csv"):
            candidate = csv_path.parent
            if _is_preprocessed_dir(candidate):
                return candidate

    searched = ", ".join(str(path) for path in search_roots)
    required = ", ".join(sorted(REQUIRED_PREPROCESSED_FILES))
    raise FileNotFoundError(f"未找到预处理结果目录；已搜索: {searched}；需要包含文件: {required}。")


def load_routing_context(
    root_dir: Optional[Path] = None,
    active_only: bool = True,
    split_demands: bool = True,
    start_minute: float = START_MINUTE,
) -> RoutingContext:
    project_root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1]
    data_dir = locate_preprocessed_dir(project_root)

    customer_frame = pd.read_csv(data_dir / "customer_attributes.csv", encoding="utf-8-sig")
    distance_frame = pd.read_csv(data_dir / "distance_matrix_clean.csv", encoding="utf-8-sig")
    speed_frame = pd.read_csv(data_dir / "speed_profile.csv", encoding="utf-8-sig")
    fleet_frame = pd.read_csv(data_dir / "vehicle_fleet.csv", encoding="utf-8-sig")

    customer_frame = customer_frame.sort_values("客户编号").reset_index(drop=True)

    distance_matrix = distance_frame.iloc[:, 1:].to_numpy(dtype=float)

    speed_bands = [
        SpeedBand(
            name=str(row["时段"]),
            start_min=float(row["开始_min"]),
            end_min=float(row["结束_min"]),
            mu_kmph=float(row["mu_kmph"]),
            sigma_kmph=float(row["sigma_kmph"]),
        )
        for _, row in speed_frame.iterrows()
    ]
    speed_bands.sort(key=lambda band: band.start_min)

    vehicle_types = [
        VehicleType(
            name=str(row["车型"]),
            capacity_weight=float(row["载重_kg"]),
            capacity_volume=float(row["容积_m3"]),
            count=int(row["数量"]),
            start_cost=float(row["启动成本_元"]),
            power_type=str(row["动力类型"]),
        )
        for _, row in fleet_frame.iterrows()
    ]

    split_weight_limit = max(vehicle.capacity_weight for vehicle in vehicle_types)
    split_volume_limit = max(vehicle.capacity_volume for vehicle in vehicle_types)

    rows = customer_frame.copy()
    if active_only:
        rows = rows[rows["总重量_kg"] > 0].copy()

    jobs: list[DeliveryJob] = []
    next_index = 0
    for _, row in rows.iterrows():
        weight = float(row["总重量_kg"])
        volume = float(row["总体积_m3"])
        if active_only and weight <= 0:
            continue
        split_count = 1
        if split_demands and weight > 0:
            split_count = max(
                1,
                math.ceil(weight / split_weight_limit),
                math.ceil(volume / split_volume_limit),
            )
        split_weight = weight / split_count if split_count else weight
        split_volume = volume / split_count if split_count else volume
        for split_index in range(split_count):
            jobs.append(
                DeliveryJob(
                    index=next_index,
                    job_key=f"C{int(row['客户编号'])}-S{split_index + 1}",
                    customer_id=int(row["客户编号"]),
                    split_index=split_index + 1,
                    split_count=split_count,
                    x=float(row["X (km)"]),
                    y=float(row["Y (km)"]),
                    green_zone=int(row["绿色区"]),
                    weight=float(split_weight),
                    volume=float(split_volume),
                    earliest=float(row["最早_min"]),
                    latest=float(row["最晚_min"]),
                )
            )
            next_index += 1

    return RoutingContext(
        root_dir=project_root,
        data_dir=data_dir,
        customer_frame=customer_frame,
        jobs=jobs,
        distance_matrix=distance_matrix,
        speed_bands=speed_bands,
        vehicle_types=vehicle_types,
        start_minute=float(start_minute),
    )


def create_virtual_jobs(
    context: RoutingContext,
    customer_id: int,
    weight: float,
    volume: float,
    earliest: Optional[float] = None,
    latest: Optional[float] = None,
    prefix: str = "EXTRA",
    index_start: Optional[int] = None,
) -> list[DeliveryJob]:
    customer = context.customer_info(customer_id)
    split_weight_limit = max(vehicle.capacity_weight for vehicle in context.vehicle_types)
    split_volume_limit = max(vehicle.capacity_volume for vehicle in context.vehicle_types)
    split_count = max(1, math.ceil(weight / split_weight_limit), math.ceil(volume / split_volume_limit))
    split_weight = weight / split_count
    split_volume = volume / split_count
    next_index = len(context.jobs) if index_start is None else int(index_start)
    jobs: list[DeliveryJob] = []
    for split_index in range(split_count):
        jobs.append(
            DeliveryJob(
                index=next_index + split_index,
                job_key=f"{prefix}-C{customer_id}-S{split_index + 1}",
                customer_id=int(customer_id),
                split_index=split_index + 1,
                split_count=split_count,
                x=float(customer["X (km)"]),
                y=float(customer["Y (km)"]),
                green_zone=int(customer["绿色区"]),
                weight=float(split_weight),
                volume=float(split_volume),
                earliest=float(customer["最早_min"] if earliest is None else earliest),
                latest=float(customer["最晚_min"] if latest is None else latest),
            )
        )
    return jobs


def fuel_consumption_per_100km(speed_kmph: float) -> float:
    return 0.0025 * speed_kmph * speed_kmph - 0.2554 * speed_kmph + 31.75


def electric_consumption_per_100km(speed_kmph: float) -> float:
    return 0.0014 * speed_kmph * speed_kmph - 0.12 * speed_kmph + 36.19


def _base_route_cost_without_start(metrics: RouteMetrics) -> float:
    return (
        metrics.fuel_cost
        + metrics.electricity_cost
        + metrics.carbon_cost
        + metrics.waiting_cost
        + metrics.late_cost
        + metrics.infeasibility_penalty
    )


def _set_route_total_cost(metrics: RouteMetrics, start_cost: float) -> None:
    metrics.start_cost = float(start_cost)
    metrics.total_cost = _base_route_cost_without_start(metrics) + metrics.start_cost


def distance_between(context: RoutingContext, from_customer: int, to_customer: int) -> float:
    return float(context.distance_matrix[int(from_customer), int(to_customer)])


def _speed_band_for_time(context: RoutingContext, current_time: float) -> SpeedBand:
    if current_time < context.speed_bands[0].start_min:
        return context.speed_bands[0]
    for band in context.speed_bands:
        if band.start_min <= current_time < band.end_min:
            return band
    return context.speed_bands[-1]


def _sample_speed(
    band: SpeedBand,
    stochastic: bool,
    rng: Optional[np.random.Generator],
) -> float:
    if not stochastic:
        return band.mu_kmph
    generator = rng if rng is not None else np.random.default_rng()
    sampled = float(generator.normal(band.mu_kmph, band.sigma_kmph))
    return float(np.clip(sampled, 8.0, 75.0))


def _simulate_leg(
    context: RoutingContext,
    vehicle: VehicleType,
    departure_time: float,
    distance_km: float,
    load_ratio: float,
    stochastic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    if distance_km <= 1e-9:
        return 0.0, 0.0, 0.0

    current_time = float(departure_time)
    remaining_distance = float(distance_km)
    total_fuel = 0.0
    total_electricity = 0.0

    while remaining_distance > 1e-9:
        band = _speed_band_for_time(context, current_time)
        speed = _sample_speed(band, stochastic=stochastic, rng=rng)
        band_end = band.end_min
        if band_end <= current_time + 1e-6:
            band_end = current_time + 30.0
        available_minutes = max(1.0, band_end - current_time)
        coverable_distance = speed * available_minutes / 60.0
        if coverable_distance <= 1e-9:
            current_time += 1.0
            continue
        leg_distance = min(remaining_distance, coverable_distance)
        leg_minutes = leg_distance / speed * 60.0
        load_factor = 1.0 + vehicle.alpha * max(0.0, min(1.0, load_ratio))
        if vehicle.is_electric:
            per_km = max(0.0, electric_consumption_per_100km(speed) / 100.0) * leg_distance
            total_electricity += per_km * load_factor
        else:
            per_km = max(0.0, fuel_consumption_per_100km(speed) / 100.0) * leg_distance
            total_fuel += per_km * load_factor
        remaining_distance -= leg_distance
        current_time += leg_minutes

    return current_time - departure_time, total_fuel, total_electricity


def evaluate_route(
    context: RoutingContext,
    vehicle: VehicleType,
    jobs: list[DeliveryJob],
    variant: str,
    start_node: int = 0,
    start_time: Optional[float] = None,
    stochastic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> RouteMetrics:
    metrics = RouteMetrics()
    metrics.start_cost = vehicle.start_cost if jobs else 0.0
    if not jobs:
        metrics.total_cost = 0.0
        return metrics

    total_weight = sum(job.weight for job in jobs)
    total_volume = sum(job.volume for job in jobs)
    if total_weight - vehicle.capacity_weight > 1e-6 or total_volume - vehicle.capacity_volume > 1e-6:
        metrics.capacity_violation = True
        metrics.infeasibility_penalty = LARGE_PENALTY
        _set_route_total_cost(metrics, start_cost=0.0)
        return metrics

    current_node = int(start_node)
    current_time = context.start_minute if start_time is None else float(start_time)
    remaining_weight = total_weight
    remaining_volume = total_volume

    for job in jobs:
        leg_distance = distance_between(context, current_node, job.customer_id)
        load_ratio = max(
            remaining_weight / vehicle.capacity_weight,
            remaining_volume / vehicle.capacity_volume,
        )
        travel_minutes, fuel, electricity = _simulate_leg(
            context=context,
            vehicle=vehicle,
            departure_time=current_time,
            distance_km=leg_distance,
            load_ratio=load_ratio,
            stochastic=stochastic,
            rng=rng,
        )
        arrival = current_time + travel_minutes
        wait_minutes = max(0.0, job.earliest - arrival)
        service_start = arrival + wait_minutes
        late_minutes = max(0.0, service_start - job.latest)

        departure = service_start + context.service_minute
        if variant in {"problem2", "dynamic"} and (not vehicle.is_electric) and job.green_zone == 1:
            # 禁行判定按车辆在绿区客户处的停留区间 [arrival, departure]
            # 与禁行区间 [BAN_START, BAN_END] 是否相交
            if arrival < BAN_END and departure > BAN_START:
                metrics.policy_violation = True

        metrics.total_distance += leg_distance
        metrics.fuel_liters += fuel
        metrics.electricity_kwh += electricity
        metrics.waiting_minutes += wait_minutes
        metrics.late_minutes += late_minutes
        metrics.stops.append(
            StopDetail(
                job_key=job.job_key,
                customer_id=job.customer_id,
                arrival=arrival,
                service_start=service_start,
                departure=departure,
                wait_minutes=wait_minutes,
                late_minutes=late_minutes,
                distance_from_prev=leg_distance,
            )
        )

        remaining_weight -= job.weight
        remaining_volume -= job.volume
        current_node = job.customer_id
        current_time = departure

    back_distance = distance_between(context, current_node, 0)
    back_minutes, fuel, electricity = _simulate_leg(
        context=context,
        vehicle=vehicle,
        departure_time=current_time,
        distance_km=back_distance,
        load_ratio=0.0,
        stochastic=stochastic,
        rng=rng,
    )
    metrics.total_distance += back_distance
    metrics.fuel_liters += fuel
    metrics.electricity_kwh += electricity
    metrics.route_end_time = current_time + back_minutes
    metrics.carbon_kg = (
        metrics.fuel_liters * FUEL_EMISSION + metrics.electricity_kwh * ELECTRICITY_EMISSION
    )
    metrics.waiting_cost = metrics.waiting_minutes * WAIT_COST_PER_MIN
    metrics.late_cost = metrics.late_minutes * LATE_COST_PER_MIN
    metrics.fuel_cost = metrics.fuel_liters * FUEL_PRICE
    metrics.electricity_cost = metrics.electricity_kwh * ELECTRICITY_PRICE
    metrics.carbon_cost = metrics.carbon_kg * CARBON_PRICE
    if metrics.policy_violation:
        metrics.infeasibility_penalty = LARGE_PENALTY
    _set_route_total_cost(metrics, start_cost=metrics.start_cost)
    return metrics


def _estimated_route_start_times(
    context: RoutingContext,
    jobs: list[DeliveryJob],
    start_node: int = 0,
) -> list[float]:
    if not jobs:
        return [context.start_minute]

    reference_speed = float(np.mean([band.mu_kmph for band in context.speed_bands])) if context.speed_bands else 35.0
    reference_speed = max(18.0, reference_speed)
    latest_candidate = max(job.latest for job in jobs) + 30.0
    candidates = {float(context.start_minute)}

    current_node = int(start_node)
    cumulative_minutes = 0.0
    for job in jobs:
        leg_distance = distance_between(context, current_node, job.customer_id)
        cumulative_minutes += leg_distance / reference_speed * 60.0
        anchor = job.earliest - cumulative_minutes
        for offset in (-20.0, -10.0, 0.0, 10.0):
            candidate = max(context.start_minute, anchor + offset)
            if candidate <= latest_candidate:
                candidates.add(round(float(candidate), 4))
        cumulative_minutes += context.service_minute
        current_node = job.customer_id

    return sorted(candidates)


def build_route_plan(
    context: RoutingContext,
    vehicle: VehicleType,
    jobs: list[DeliveryJob],
    variant: str,
    start_node: int = 0,
    start_time: Optional[float] = None,
    assigned_vehicle_id: Optional[str] = None,
    startup_required: bool = True,
    stochastic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> RoutePlan:
    chosen_start_time = context.start_minute if start_time is None else float(start_time)
    if jobs and start_time is None and not stochastic:
        best_metrics: Optional[RouteMetrics] = None
        for candidate_start in _estimated_route_start_times(context, jobs, start_node):
            candidate_metrics = evaluate_route(
                context=context,
                vehicle=vehicle,
                jobs=jobs,
                variant=variant,
                start_node=start_node,
                start_time=candidate_start,
                stochastic=False,
                rng=rng,
            )
            if best_metrics is None or candidate_metrics.total_cost + 1e-9 < best_metrics.total_cost:
                best_metrics = candidate_metrics
                chosen_start_time = candidate_start
        metrics = best_metrics if best_metrics is not None else evaluate_route(
            context=context,
            vehicle=vehicle,
            jobs=jobs,
            variant=variant,
            start_node=start_node,
            start_time=chosen_start_time,
            stochastic=False,
            rng=rng,
        )
    else:
        metrics = evaluate_route(
            context=context,
            vehicle=vehicle,
            jobs=jobs,
            variant=variant,
            start_node=start_node,
            start_time=chosen_start_time,
            stochastic=stochastic,
            rng=rng,
        )
    _set_route_total_cost(metrics, start_cost=vehicle.start_cost if jobs and startup_required else 0.0)
    return RoutePlan(
        vehicle=vehicle,
        jobs=list(jobs),
        metrics=metrics,
        start_node=start_node,
        start_time=chosen_start_time,
        assigned_vehicle_id=assigned_vehicle_id,
        startup_required=startup_required,
    )


def route_selection_score(
    route: RoutePlan,
    variant: str,
    tradeoff_lambda: float = 0.0,
) -> float:
    """
    解码阶段对单条候选路线的评分：与主目标一致，不叠加无文献支撑的经验常数。

    - total_cost：已含启动费、能耗、碳价、等待/迟到及 infeasibility_penalty（容量/政策大罚）
    - tradeoff_lambda * carbon_kg：与问题 2 折中目标一致
    - 微量 tie-break：取车队表中「启动成本」的一个可忽略比例，使 DP 在数值并列时略偏好更少分段
      （每多一条非空路线增加极小正项）；比例 1e-6 仅用于打破浮点并列，不影响经济排序主结论。
    """
    _ = variant
    if route.metrics.capacity_violation or route.metrics.policy_violation:
        return LARGE_PENALTY
    base = float(route.metrics.total_cost) + float(tradeoff_lambda) * float(route.metrics.carbon_kg)
    if not route.jobs:
        return base
    trip_tie_break = 1e-6 * float(route.vehicle.start_cost)
    return base + trip_tie_break


def vehicle_allowed_for_jobs(
    vehicle: VehicleType,
    jobs: list[DeliveryJob],
    variant: str,
) -> bool:
    if variant not in {"problem2", "dynamic"}:
        return True
    if vehicle.is_electric:
        return True
    return not jobs_require_electric(jobs, variant)


def jobs_require_electric(
    jobs: list[DeliveryJob],
    variant: str,
) -> bool:
    if variant not in {"problem2", "dynamic"}:
        return False
    green_jobs = [job for job in jobs if job.green_zone == 1]
    if not green_jobs:
        return False
    # 从“时间窗有交集就强制电车”改为“是否存在全体绿区任务均可避开禁行时段的可行窗口”：
    # - 对每个绿区任务，若 earliest + service <= BAN_START，则存在“禁行开始前完成服务”的可能；
    # - 或若 latest >= BAN_END，则存在“禁行结束后开始服务”的可能（可等待至 16:00 后服务）。
    # 仅当存在至少一个绿区任务两者都不满足时，才强制电车。
    for job in green_jobs:
        can_finish_before_ban = (job.earliest + SERVICE_MINUTE) <= BAN_START + 1e-9
        can_start_after_ban = job.latest >= BAN_END - 1e-9
        if not (can_finish_before_ban or can_start_after_ban):
            return True
    return False


def improve_route_two_opt(
    context: RoutingContext,
    route: RoutePlan,
    variant: str,
) -> RoutePlan:
    if len(route.jobs) < 3:
        return route

    best_jobs = list(route.jobs)
    best_metrics = route.metrics
    improved = True
    while improved:
        improved = False
        for left in range(0, len(best_jobs) - 1):
            for right in range(left + 2, len(best_jobs) + 1):
                candidate_jobs = best_jobs[:left] + list(reversed(best_jobs[left:right])) + best_jobs[right:]
                candidate_metrics = evaluate_route(
                    context=context,
                    vehicle=route.vehicle,
                    jobs=candidate_jobs,
                    variant=variant,
                    start_node=route.start_node,
                    start_time=route.start_time,
                )
                if candidate_metrics.total_cost + 1e-6 < best_metrics.total_cost:
                    best_jobs = candidate_jobs
                    best_metrics = candidate_metrics
                    improved = True
                    break
            if improved:
                break

    _set_route_total_cost(
        best_metrics,
        start_cost=route.vehicle.start_cost if best_jobs and route.startup_required else 0.0,
    )
    return RoutePlan(
        vehicle=route.vehicle,
        jobs=best_jobs,
        metrics=best_metrics,
        start_node=route.start_node,
        start_time=route.start_time,
        assigned_vehicle_id=route.assigned_vehicle_id,
        startup_required=route.startup_required,
    )


def _fallback_single_routes(
    context: RoutingContext,
    permutation: list[int],
    variant: str,
    tradeoff_lambda: float,
) -> list[RoutePlan]:
    routes: list[RoutePlan] = []
    for job_index in permutation:
        job = context.jobs[job_index]
        candidate_routes = []
        for vehicle in context.vehicle_types:
            if vehicle.count <= 0:
                continue
            if not vehicle_allowed_for_jobs(vehicle, [job], variant):
                continue
            if job.weight <= vehicle.capacity_weight and job.volume <= vehicle.capacity_volume:
                route = build_route_plan(context, vehicle, [job], variant)
                candidate_routes.append((route_selection_score(route, variant, tradeoff_lambda), route))
        if not candidate_routes:
            route = build_route_plan(context, context.vehicle_types[-1], [job], variant)
            route.metrics.total_cost += LARGE_PENALTY
            routes.append(route)
            continue
        routes.append(min(candidate_routes, key=lambda item: item[0])[1])
    return routes


def repair_vehicle_counts(
    context: RoutingContext,
    routes: list[RoutePlan],
    variant: str,
    tradeoff_lambda: float = 0.0,
) -> tuple[list[RoutePlan], float]:
    fixed_routes = list(routes)
    penalty = 0.0

    while True:
        scheduled_solution = evaluate_solution(context=context, routes=fixed_routes, variant=variant)
        overflow = {
            name: count - context.vehicle_limits.get(name, 0)
            for name, count in scheduled_solution.metadata.get("physical_vehicle_usage", {}).items()
            if count > context.vehicle_limits.get(name, 0)
        }
        if not overflow:
            break

        current_overflow = sum(overflow.values())
        candidate_switches = []
        for over_name in sorted(overflow, key=overflow.get, reverse=True):
            for route_index, route in enumerate(fixed_routes):
                if route.vehicle.name != over_name:
                    continue
                for alternative in context.vehicle_types:
                    if alternative.count <= 0:
                        continue
                    if alternative.name == over_name:
                        continue
                    if not vehicle_allowed_for_jobs(alternative, route.jobs, variant):
                        continue
                    if route.total_weight - alternative.capacity_weight > 1e-6:
                        continue
                    if route.total_volume - alternative.capacity_volume > 1e-6:
                        continue
                    candidate = build_route_plan(
                        context=context,
                        vehicle=alternative,
                        jobs=route.jobs,
                        variant=variant,
                        start_node=route.start_node,
                        start_time=route.start_time,
                        assigned_vehicle_id=route.assigned_vehicle_id,
                        startup_required=route.startup_required,
                    )
                    if candidate.metrics.policy_violation:
                        continue
                    trial_routes = list(fixed_routes)
                    trial_routes[route_index] = candidate
                    trial_solution = evaluate_solution(context=context, routes=trial_routes, variant=variant)
                    trial_overflow = sum(
                        max(0, count - context.vehicle_limits.get(name, 0))
                        for name, count in trial_solution.metadata.get("physical_vehicle_usage", {}).items()
                    )
                    if trial_overflow < current_overflow:
                        candidate_switches.append((trial_overflow, trial_solution.total_cost, route_index, [candidate]))

                if len(route.jobs) >= 2:
                    for split_pos in range(1, len(route.jobs)):
                        left_jobs = route.jobs[:split_pos]
                        right_jobs = route.jobs[split_pos:]
                        left_weight = sum(job.weight for job in left_jobs)
                        left_volume = sum(job.volume for job in left_jobs)
                        right_weight = sum(job.weight for job in right_jobs)
                        right_volume = sum(job.volume for job in right_jobs)
                        for left_vehicle in context.vehicle_types:
                            if left_vehicle.count <= 0:
                                continue
                            if not vehicle_allowed_for_jobs(left_vehicle, left_jobs, variant):
                                continue
                            if left_weight - left_vehicle.capacity_weight > 1e-6:
                                continue
                            if left_volume - left_vehicle.capacity_volume > 1e-6:
                                continue
                            left_route = build_route_plan(
                                context=context,
                                vehicle=left_vehicle,
                                jobs=left_jobs,
                                variant=variant,
                                start_node=route.start_node,
                                start_time=route.start_time,
                                startup_required=route.startup_required,
                            )
                            if left_route.metrics.policy_violation:
                                continue
                            for right_vehicle in context.vehicle_types:
                                if right_vehicle.count <= 0:
                                    continue
                                if not vehicle_allowed_for_jobs(right_vehicle, right_jobs, variant):
                                    continue
                                if right_weight - right_vehicle.capacity_weight > 1e-6:
                                    continue
                                if right_volume - right_vehicle.capacity_volume > 1e-6:
                                    continue
                                right_route = build_route_plan(
                                    context=context,
                                    vehicle=right_vehicle,
                                    jobs=right_jobs,
                                    variant=variant,
                                    start_node=route.start_node,
                                    start_time=route.start_time,
                                )
                                if right_route.metrics.policy_violation:
                                    continue
                                trial_routes = fixed_routes[:route_index] + [left_route, right_route] + fixed_routes[route_index + 1 :]
                                trial_solution = evaluate_solution(context=context, routes=trial_routes, variant=variant)
                                trial_overflow = sum(
                                    max(0, count - context.vehicle_limits.get(name, 0))
                                    for name, count in trial_solution.metadata.get("physical_vehicle_usage", {}).items()
                                )
                                if trial_overflow >= current_overflow:
                                    continue
                                candidate_switches.append(
                                    (
                                        trial_overflow,
                                        trial_solution.total_cost,
                                        route_index,
                                        [left_route, right_route],
                                    )
                                )
        if not candidate_switches:
            penalty += LARGE_PENALTY * current_overflow
            break

        _, _, route_index, candidate = min(candidate_switches, key=lambda item: (item[0], item[1]))
        fixed_routes = fixed_routes[:route_index] + list(candidate) + fixed_routes[route_index + 1 :]

    return fixed_routes, penalty


def decode_permutation(
    context: RoutingContext,
    permutation: Iterable[int],
    variant: str,
    tradeoff_lambda: float = 0.0,
    max_segment_length: int = 20,
) -> Solution:
    order = consolidate_split_order(list(permutation), context.jobs)
    n_jobs = len(order)
    if n_jobs == 0:
        return Solution(routes=[], total_cost=0.0, total_carbon=0.0, total_distance=0.0, feasible=True, variant=variant)

    jobs = [context.jobs[index] for index in order]
    dp = [math.inf] * (n_jobs + 1)
    parent: list[Optional[tuple[int, RoutePlan]]] = [None] * (n_jobs + 1)
    dp[0] = 0.0

    for end in range(1, n_jobs + 1):
        segment_jobs: list[DeliveryJob] = []
        segment_weight = 0.0
        segment_volume = 0.0
        for start in range(end - 1, max(-1, end - max_segment_length - 1), -1):
            job = jobs[start]
            segment_jobs.insert(0, job)
            segment_weight += job.weight
            segment_volume += job.volume
            if segment_weight - context.max_capacity_weight > 1e-6 or segment_volume - context.max_capacity_volume > 1e-6:
                continue
            best_route: Optional[RoutePlan] = None
            best_score = math.inf
            for vehicle in context.vehicle_types:
                if vehicle.count <= 0:
                    continue
                if not vehicle_allowed_for_jobs(vehicle, segment_jobs, variant):
                    continue
                if segment_weight - vehicle.capacity_weight > 1e-6:
                    continue
                if segment_volume - vehicle.capacity_volume > 1e-6:
                    continue
                route = build_route_plan(context, vehicle, segment_jobs, variant)
                score = route_selection_score(route, variant, tradeoff_lambda)
                if score < best_score:
                    best_score = score
                    best_route = route
            if best_route is None:
                continue
            total_score = dp[start] + best_score
            if total_score + 1e-9 < dp[end]:
                dp[end] = total_score
                parent[end] = (start, best_route)

    routes: list[RoutePlan] = []
    cursor = n_jobs
    while cursor > 0 and parent[cursor] is not None:
        start, route = parent[cursor]
        routes.append(route)
        cursor = start
    routes.reverse()

    if cursor != 0:
        routes = _fallback_single_routes(context, order, variant, tradeoff_lambda)

    routes = [improve_route_two_opt(context, route, variant) for route in routes]
    routes, count_penalty = repair_vehicle_counts(
        context=context,
        routes=routes,
        variant=variant,
        tradeoff_lambda=tradeoff_lambda,
    )
    solution = evaluate_solution(context=context, routes=routes, variant=variant)
    solution.metadata["permutation"] = order
    solution.metadata["tradeoff_lambda"] = tradeoff_lambda
    solution.metadata["count_penalty"] = count_penalty
    if count_penalty > 0:
        solution.total_cost += count_penalty
        solution.feasible = False
    return solution


def evaluate_solution(
    context: RoutingContext,
    routes: list[RoutePlan],
    variant: str,
) -> Solution:
    grouped_routes: dict[str, list[RoutePlan]] = {}
    for route in routes:
        grouped_routes.setdefault(route.vehicle.name, []).append(route)

    physical_usage = Counter()
    for vehicle_name, vehicle_routes in grouped_routes.items():
        vehicle_routes.sort(key=lambda item: (item.start_time, item.metrics.route_end_time))
        locked_routes: dict[str, list[RoutePlan]] = {}
        unlocked_routes: list[RoutePlan] = []
        used_ids: set[str] = set()
        next_instance = 1

        for route in vehicle_routes:
            if route.assigned_vehicle_id:
                locked_routes.setdefault(route.assigned_vehicle_id, []).append(route)
                used_ids.add(route.assigned_vehicle_id)
                suffix = route.assigned_vehicle_id.rsplit("-", maxsplit=1)[-1]
                if suffix.isdigit():
                    next_instance = max(next_instance, int(suffix) + 1)
            else:
                unlocked_routes.append(route)

        heap: list[tuple[float, str]] = []
        for vehicle_id, locked_group in locked_routes.items():
            locked_group.sort(key=lambda item: (item.start_time, item.metrics.route_end_time))
            last_end = -math.inf
            for idx, route in enumerate(locked_group):
                startup_cost = route.vehicle.start_cost if route.startup_required and idx == 0 else 0.0
                if route.start_time + 1e-6 < last_end:
                    route.metrics.infeasibility_penalty += LARGE_PENALTY
                route.assigned_vehicle_id = vehicle_id
                _set_route_total_cost(route.metrics, start_cost=startup_cost)
                last_end = route.metrics.route_end_time
            heapq.heappush(heap, (last_end, vehicle_id))

        for route in unlocked_routes:
            if heap and heap[0][0] <= route.start_time + 1e-6:
                _, vehicle_id = heapq.heappop(heap)
                startup_cost = 0.0
            else:
                while f"{vehicle_name}-{next_instance}" in used_ids:
                    next_instance += 1
                vehicle_id = f"{vehicle_name}-{next_instance}"
                next_instance += 1
                used_ids.add(vehicle_id)
                startup_cost = route.vehicle.start_cost if route.startup_required else 0.0
            route.assigned_vehicle_id = vehicle_id
            _set_route_total_cost(route.metrics, start_cost=startup_cost)
            heapq.heappush(heap, (route.metrics.route_end_time, vehicle_id))

        physical_usage[vehicle_name] = max(len(used_ids), len(locked_routes))

    total_cost = sum(route.metrics.total_cost for route in routes)
    total_carbon = sum(route.metrics.carbon_kg for route in routes)
    total_distance = sum(route.metrics.total_distance for route in routes)

    feasible = True
    for route in routes:
        if route.metrics.capacity_violation or route.metrics.policy_violation:
            feasible = False
            break
    for name, count in physical_usage.items():
        if count > context.vehicle_limits.get(name, 0):
            feasible = False
            total_cost += LARGE_PENALTY * (count - context.vehicle_limits.get(name, 0))

    solution = Solution(
        routes=routes,
        total_cost=total_cost,
        total_carbon=total_carbon,
        total_distance=total_distance,
        feasible=feasible,
        variant=variant,
        metadata={},
    )
    solution.metadata["physical_vehicle_usage"] = dict(physical_usage)
    return solution


def monte_carlo_solution_stats(
    context: RoutingContext,
    solution: Solution,
    runs: int = 20,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    costs = []
    carbons = []
    for _ in range(max(1, runs)):
        run_cost = 0.0
        run_carbon = 0.0
        for route in solution.routes:
            metrics = evaluate_route(
                context=context,
                vehicle=route.vehicle,
                jobs=route.jobs,
                variant=solution.variant,
                start_node=route.start_node,
                start_time=route.start_time,
                stochastic=True,
                rng=rng,
            )
            _set_route_total_cost(metrics, start_cost=route.metrics.start_cost)
            run_cost += metrics.total_cost
            run_carbon += metrics.carbon_kg
        costs.append(run_cost)
        carbons.append(run_carbon)
    return {
        "expected_cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "expected_carbon": float(np.mean(carbons)),
        "carbon_std": float(np.std(carbons)),
    }


def consolidate_split_order(perm: list[int], jobs: list[DeliveryJob]) -> list[int]:
    """
    将同一客户拆分的多笔任务在序列中并成连续段，且按 split_index 排序，
    保证同一客户的子任务在任务序列中连续，且与 perm 中“首次出现该客户”的先后一致。
    """
    if not perm:
        return []
    n = len(jobs)
    by_customer: dict[int, list[int]] = {}
    for idx in perm:
        if not (0 <= idx < n):
            continue
        cid = jobs[idx].customer_id
        by_customer.setdefault(cid, []).append(idx)
    for cid, indices in by_customer.items():
        by_customer[cid] = sorted(
            list(dict.fromkeys(indices)),
            key=lambda i: (jobs[i].split_index, jobs[i].index, i),
        )
    seen: set[int] = set()
    out: list[int] = []
    for idx in perm:
        if not (0 <= idx < n):
            continue
        cid = jobs[idx].customer_id
        if cid in seen:
            continue
        seen.add(cid)
        out.extend(by_customer.get(cid, []))
    return out


def base_permutation(context: RoutingContext) -> list[int]:
    def job_key(job: DeliveryJob) -> tuple:
        angle = math.atan2(job.y, job.x)
        demand_score = job.weight + 120.0 * job.volume
        return (
            job.earliest,
            -job.green_zone,
            angle,
            -demand_score,
            job.customer_id,
            job.split_index,
        )

    ordered = sorted(context.jobs, key=job_key)
    raw = [job.index for job in ordered]
    return consolidate_split_order(raw, context.jobs)


def data_quality_audit(context: RoutingContext) -> dict:
    """返回建模前关键数据一致性指标，便于写入结果摘要和论文说明。"""
    frame = context.customer_frame
    green_mask = frame["绿色区"] == 1
    positive_mask = frame["总重量_kg"] > 0
    zero_demand_ids = frame.loc[~positive_mask, "客户编号"].astype(int).tolist()
    green_ids = frame.loc[green_mask, "客户编号"].astype(int).tolist()
    green_positive_ids = frame.loc[green_mask & positive_mask, "客户编号"].astype(int).tolist()
    return {
        "customer_count": int(len(frame)),
        "active_customer_count": int(positive_mask.sum()),
        "zero_demand_customer_count": int((~positive_mask).sum()),
        "zero_demand_customer_ids": zero_demand_ids,
        "green_customer_count": int(green_mask.sum()),
        "green_customer_ids": green_ids,
        "active_green_customer_count": int((green_mask & positive_mask).sum()),
        "active_green_customer_ids": green_positive_ids,
        "green_zone_basis": "附件坐标按距市中心≤10km计算；题面30个绿色区客户与附件计算不一致时，以附件数据为准。",
        "startup_cost_basis": "启动成本按当日实际启用物理车辆计，同一物理车辆多趟配送不重复计启动成本。",
    }


def physical_vehicle_mix(solution: Solution) -> dict[str, int]:
    if solution.metadata.get("physical_vehicle_usage"):
        usage = dict(solution.metadata["physical_vehicle_usage"])
        electric_names = {route.vehicle.name for route in solution.routes if route.vehicle.is_electric}
        electric_total = sum(count for name, count in usage.items() if name in electric_names)
        fuel_total = sum(count for name, count in usage.items() if name not in electric_names)
        return {
            "新能源车": int(electric_total),
            "燃油车": int(fuel_total),
        }
    return vehicle_mix(solution)


def route_rows(solution: Solution) -> list[dict]:
    rows = []
    for route_id, route in enumerate(solution.routes, start=1):
        operating_cost = _base_route_cost_without_start(route.metrics)
        rows.append(
            {
                "route_id": route_id,
                "vehicle_type": route.vehicle.name,
                "assigned_vehicle_id": route.assigned_vehicle_id,
                "power_type": route.vehicle.power_type,
                "num_jobs": len(route.jobs),
                "customers": "->".join(str(job.customer_id) for job in route.jobs),
                "job_keys": "->".join(job.job_key for job in route.jobs),
                "route_start_min": route.start_time,
                "route_end_min": route.metrics.route_end_time,
                "total_weight_kg": route.total_weight,
                "total_volume_m3": route.total_volume,
                "distance_km": route.metrics.total_distance,
                "fuel_liters": route.metrics.fuel_liters,
                "electricity_kwh": route.metrics.electricity_kwh,
                "carbon_kg": route.metrics.carbon_kg,
                "waiting_minutes": route.metrics.waiting_minutes,
                "late_minutes": route.metrics.late_minutes,
                "startup_cost_allocated": route.metrics.start_cost,
                "operating_cost": operating_cost,
                "route_cost": route.metrics.total_cost,
            }
        )
    return rows


def stop_rows(solution: Solution) -> list[dict]:
    rows = []
    for route_id, route in enumerate(solution.routes, start=1):
        for stop_id, detail in enumerate(route.metrics.stops, start=1):
            rows.append(
                {
                    "route_id": route_id,
                    "stop_id": stop_id,
                    "vehicle_type": route.vehicle.name,
                    "job_key": detail.job_key,
                    "customer_id": detail.customer_id,
                    "arrival_min": detail.arrival,
                    "service_start_min": detail.service_start,
                    "departure_min": detail.departure,
                    "wait_min": detail.wait_minutes,
                    "late_min": detail.late_minutes,
                    "distance_from_prev_km": detail.distance_from_prev,
                }
            )
    return rows


def vehicle_mix(solution: Solution) -> dict[str, int]:
    mix = Counter()
    for route in solution.routes:
        if route.vehicle.is_electric:
            mix["新能源车"] += 1
        else:
            mix["燃油车"] += 1
    return dict(mix)
