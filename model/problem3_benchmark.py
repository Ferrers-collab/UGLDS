from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Optional

import numpy as np
import pandas as pd

from .core import load_routing_context, stop_rows
from .core import RoutingContext, Solution
from .problem1 import Problem1Config, solve_problem1
from .problem3 import DynamicEvent, Problem3Config, build_default_events, job_customer_id, solve_problem3


def _remaining_customers_for_event(baseline_solution, event_time: float) -> list[int]:
    customers: list[int] = []
    for route in baseline_solution.routes:
        for job, stop in zip(route.jobs, route.metrics.stops):
            if stop.service_start >= event_time:
                customers.append(job.customer_id)
    return list(dict.fromkeys(customers))


def _scenario_events(context, baseline_solution, config: Problem3Config) -> dict[str, list[DynamicEvent]]:
    remaining_customers = _remaining_customers_for_event(baseline_solution, config.event_time)
    default_events = build_default_events(context, baseline_solution, config)

    cancel_customer = remaining_customers[0] if remaining_customers else 2
    zero_demand = context.customer_frame[context.customer_frame["总重量_kg"] <= 0]["客户编号"].astype(int).tolist()
    add_customer = next((cid for cid in zero_demand if cid not in remaining_customers), int(context.customer_frame.iloc[0]["客户编号"]))
    add_earliest = max(config.event_time + 15.0, float(context.customer_info(add_customer)["最早_min"]))
    add_latest = add_earliest + 35.0

    return {
        "single_cancel_small": [
            DynamicEvent(kind="取消", customer_id=int(cancel_customer), description="单取消小扰动"),
        ],
        "single_add_tight_tw": [
            DynamicEvent(
                kind="新增",
                customer_id=int(add_customer),
                weight=950.0,
                volume=3.1,
                earliest=add_earliest,
                latest=add_latest,
                description="单新增且时间窗较紧",
            ),
        ],
        "multi_event_high": default_events,
    }


def _coverage_rate(result: dict, events: list[DynamicEvent]) -> float:
    solution = result["solution"]
    baseline_lookup = solution.metadata.get("baseline_job_vehicle", {})
    final_lookup = solution.metadata.get("final_job_vehicle", {})
    cancel_customers = {event.customer_id for event in events if event.kind == "取消"}

    required = {
        job_key
        for job_key in baseline_lookup
        if (job_customer_id(job_key) is None or job_customer_id(job_key) not in cancel_customers)
    }
    required.update(job_key for job_key in final_lookup if str(job_key).startswith("DYN-"))

    if not required:
        return 1.0
    served = {job_key for job_key in required if job_key in final_lookup}
    return len(served) / len(required)


def _on_time_rates(result: dict) -> tuple[float, float]:
    rows = pd.DataFrame(stop_rows(result["solution"]))
    if rows.empty:
        return 1.0, 1.0
    stop_rate = float((rows["late_min"] <= 1e-9).mean())
    customer_late = rows.groupby("customer_id", as_index=False)["late_min"].max()
    customer_rate = float((customer_late["late_min"] <= 1e-9).mean())
    return stop_rate, customer_rate


def _route_change_rate(result: dict) -> float:
    solution = result["solution"]
    baseline_lookup = solution.metadata.get("baseline_job_vehicle", {})
    final_lookup = solution.metadata.get("final_job_vehicle", {})
    keys = sorted(set(baseline_lookup) | set(final_lookup))
    if not keys:
        return 0.0
    changed = sum(1 for key in keys if baseline_lookup.get(key, "") != final_lookup.get(key, ""))
    return changed / len(keys)


def run_problem3_benchmarks(
    output_dir: Path,
    runs_per_scenario: int = 3,
    seed: int = 20260425,
    context: Optional[RoutingContext] = None,
    baseline_solution: Optional[Solution] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    context = context or load_routing_context()
    baseline = baseline_solution or solve_problem1(
        context,
        Problem1Config(seed=seed, population_size=60, generations=120),
    )
    config = Problem3Config(seed=seed + 2, search_iterations=8, time_limit_seconds=8.0)
    scenarios = _scenario_events(context, baseline, config)

    run_rows: list[dict] = []
    for scenario_name, events in scenarios.items():
        for run_idx in range(runs_per_scenario):
            run_config = Problem3Config(
                event_time=config.event_time,
                route_change_weight=config.route_change_weight,
                search_iterations=config.search_iterations,
                destroy_fraction=config.destroy_fraction,
                max_segment_length=config.max_segment_length,
                time_limit_seconds=config.time_limit_seconds,
                fast_repair_job_threshold=config.fast_repair_job_threshold,
                seed=config.seed + run_idx,
            )
            result = solve_problem3(context, baseline, run_config, events=events)
            solution = result["solution"]
            stop_rate, customer_rate = _on_time_rates(result)
            full_day_cost = float(solution.metadata.get("full_day_cost_est", solution.total_cost))
            cost_delta_pct = (full_day_cost - baseline.total_cost) / max(1e-9, baseline.total_cost) * 100.0
            run_rows.append(
                {
                    "scenario": scenario_name,
                    "run": run_idx + 1,
                    "strategy": result["strategy"],
                    "urgency": result["urgency"],
                    "impact_score": result.get("impact_score", result["urgency"]),
                    "disturbance": result["disturbance"],
                    "coverage_rate": _coverage_rate(result, events),
                    "on_time_rate_stop": stop_rate,
                    "on_time_rate_customer": customer_rate,
                    "cost_delta_pct_vs_p1": cost_delta_pct,
                    "route_change_rate": _route_change_rate(result),
                    "solve_seconds": float(solution.metadata.get("solve_seconds", 0.0)),
                }
            )

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(output_dir / "benchmark_runs.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict] = []
    for scenario_name, frame in run_df.groupby("scenario"):
        solve_seconds = frame["solve_seconds"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "scenario": scenario_name,
                "selected_strategy": frame["strategy"].mode().iloc[0],
                "coverage_rate_mean": frame["coverage_rate"].mean(),
                "on_time_rate_stop_mean": frame["on_time_rate_stop"].mean(),
                "on_time_rate_customer_mean": frame["on_time_rate_customer"].mean(),
                "cost_delta_pct_vs_p1_mean": frame["cost_delta_pct_vs_p1"].mean(),
                "route_change_rate_mean": frame["route_change_rate"].mean(),
                "solve_seconds_mean": float(mean(solve_seconds)),
                "solve_seconds_p95": float(np.percentile(solve_seconds, 95)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("scenario").reset_index(drop=True)
    summary_df.to_csv(output_dir / "benchmark_summary.csv", index=False, encoding="utf-8-sig")

    strategy_rows: list[dict] = []
    for tag in ("greedy_insert", "fast_baseline_repair", "lns_reoptimize", "lns_reoptimize_limited"):
        frame = run_df[run_df["strategy"].str.startswith(tag)]
        if frame.empty:
            continue
        solve_seconds = frame["solve_seconds"].to_numpy(dtype=float)
        strategy_rows.append(
            {
                "strategy": tag,
                "samples": len(frame),
                "solve_seconds_mean": float(mean(solve_seconds)),
                "solve_seconds_p95": float(np.percentile(solve_seconds, 95)),
            }
        )
    strategy_df = pd.DataFrame(strategy_rows)
    strategy_df.to_csv(output_dir / "benchmark_strategy_latency.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "问题3 实时性与多场景评测摘要",
        f"基线问题1总成本: {baseline.total_cost:.2f}",
        f"每场景重复次数: {runs_per_scenario}",
        f"场景汇总文件: {output_dir / 'benchmark_summary.csv'}",
        f"策略时延文件: {output_dir / 'benchmark_strategy_latency.csv'}",
    ]
    (output_dir / "benchmark_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "baseline_cost": baseline.total_cost,
        "runs": run_df,
        "summary": summary_df,
        "strategy_latency": strategy_df,
        "scenarios": {name: [asdict(event) for event in events] for name, events in scenarios.items()},
    }
